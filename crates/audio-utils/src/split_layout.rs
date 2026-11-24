//! Which time ranges of a session's audio file were recorded in split stereo mic mode. The
//! recorder writes `split-layout.json` next to the audio so later transcription knows where the
//! file holds two speakers (left, right) instead of a mono mix.

use std::io;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

pub const SPLIT_LAYOUT_FILE: &str = "split-layout.json";
const VERSION: u32 = 1;

/// Two captures may begin and end ranges at once: one writer at a time keeps either from losing
/// the other's update.
static WRITE_LOCK: Mutex<()> = Mutex::new(());
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Offsets are ms from the start of the session audio file.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SplitRange {
    pub start_ms: i64,
    /// `None` while still recording, or after a crash: the range runs to the end of the file.
    pub end_ms: Option<i64>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SplitLayout {
    pub version: u32,
    pub ranges: Vec<SplitRange>,
}

enum LoadState {
    Loaded(SplitLayout),
    MissingOrMalformed,
    NewerVersion,
}

impl SplitLayout {
    /// `Ok(None)` when the file is missing or unreadable as a layout.
    pub fn load(session_dir: &Path) -> io::Result<Option<SplitLayout>> {
        match Self::load_state(session_dir)? {
            LoadState::Loaded(layout) => Ok(Some(layout)),
            LoadState::MissingOrMalformed | LoadState::NewerVersion => Ok(None),
        }
    }

    fn load_state(session_dir: &Path) -> io::Result<LoadState> {
        let bytes = match std::fs::read(session_dir.join(SPLIT_LAYOUT_FILE)) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Ok(LoadState::MissingOrMalformed);
            }
            Err(e) => return Err(e),
        };
        match serde_json::from_slice::<SplitLayout>(&bytes) {
            Ok(layout) if layout.version > VERSION => {
                tracing::warn!(version = layout.version, "ignoring newer split layout file");
                Ok(LoadState::NewerVersion)
            }
            Ok(layout) => Ok(LoadState::Loaded(layout)),
            Err(e) => {
                tracing::warn!(error = %e, "ignoring malformed split layout file");
                Ok(LoadState::MissingOrMalformed)
            }
        }
    }

    /// Closes an open range at `start_ms`, then opens a new one there.
    /// Leaves newer-version layout files untouched rather than overwriting them with v1.
    pub fn begin_range(session_dir: &Path, start_ms: i64) -> io::Result<()> {
        let _guard = WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut layout = match Self::load_state(session_dir)? {
            LoadState::Loaded(layout) => layout,
            LoadState::MissingOrMalformed => SplitLayout::default(),
            LoadState::NewerVersion => return Ok(()),
        };
        if let Some(open) = layout.ranges.iter_mut().find(|r| r.end_ms.is_none()) {
            open.end_ms = Some(start_ms.max(open.start_ms));
        }
        layout.ranges.push(SplitRange {
            start_ms,
            end_ms: None,
        });
        layout.save(session_dir)
    }

    /// Closes the open range at `end_ms`. Does nothing when no range is open.
    pub fn end_range(session_dir: &Path, end_ms: i64) -> io::Result<()> {
        let _guard = WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut layout) = Self::load(session_dir)? else {
            return Ok(());
        };
        let Some(open) = layout.ranges.iter_mut().find(|r| r.end_ms.is_none()) else {
            return Ok(());
        };
        open.end_ms = Some(end_ms.max(open.start_ms));
        layout.save(session_dir)
    }

    /// Sorted, disjoint ranges; open ranges end at `file_end_ms`. Empty ranges are dropped and
    /// overlapping ones merged.
    pub fn ranges_ms(&self, file_end_ms: i64) -> Vec<(i64, i64)> {
        let mut ranges: Vec<_> = self
            .ranges
            .iter()
            .map(|r| (r.start_ms, r.end_ms.unwrap_or(file_end_ms)))
            .filter(|(start, end)| end > start)
            .collect();
        ranges.sort();
        let mut merged: Vec<(i64, i64)> = Vec::new();
        for (start, end) in ranges {
            match merged.last_mut() {
                Some(last) if start <= last.1 => last.1 = last.1.max(end),
                _ => merged.push((start, end)),
            }
        }
        merged
    }

    /// True when `ms` lies in a range. An open range has no end, so it contains all later times.
    pub fn contains(&self, ms: i64) -> bool {
        self.ranges
            .iter()
            .any(|r| ms >= r.start_ms && r.end_ms.is_none_or(|end| ms < end))
    }

    fn save(mut self, session_dir: &Path) -> io::Result<()> {
        self.version = VERSION;
        let json = serde_json::to_vec_pretty(&self).map_err(io::Error::other)?;
        let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp = session_dir.join(format!(
            "{SPLIT_LAYOUT_FILE}.{}.{n}.tmp",
            std::process::id()
        ));
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, session_dir.join(SPLIT_LAYOUT_FILE)).inspect_err(|_| {
            let _ = std::fs::remove_file(&tmp);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(start_ms: i64, end_ms: Option<i64>) -> SplitRange {
        SplitRange { start_ms, end_ms }
    }

    #[test]
    fn begin_and_end_ranges_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        SplitLayout::begin_range(dir.path(), 0).unwrap();
        SplitLayout::end_range(dir.path(), 60_000).unwrap();
        SplitLayout::begin_range(dir.path(), 90_000).unwrap();
        let layout = SplitLayout::load(dir.path()).unwrap().unwrap();
        assert_eq!(
            layout,
            SplitLayout {
                version: 1,
                ranges: vec![range(0, Some(60_000)), range(90_000, None)],
            }
        );
        assert_eq!(
            layout.ranges_ms(120_000),
            vec![(0, 60_000), (90_000, 120_000)]
        );
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn begin_closes_an_open_range_at_the_new_start() {
        let dir = tempfile::tempdir().unwrap();
        SplitLayout::begin_range(dir.path(), 1_000).unwrap();
        SplitLayout::begin_range(dir.path(), 5_000).unwrap();
        let layout = SplitLayout::load(dir.path()).unwrap().unwrap();
        assert_eq!(
            layout.ranges,
            vec![range(1_000, Some(5_000)), range(5_000, None)]
        );
    }

    #[test]
    fn missing_file_loads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(SplitLayout::load(dir.path()).unwrap(), None);
    }

    #[test]
    fn malformed_file_loads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(SPLIT_LAYOUT_FILE), "{not json").unwrap();
        assert_eq!(SplitLayout::load(dir.path()).unwrap(), None);
    }

    #[test]
    fn end_without_begin_is_a_no_op() {
        let dir = tempfile::tempdir().unwrap();
        SplitLayout::end_range(dir.path(), 5_000).unwrap();
        assert!(!dir.path().join(SPLIT_LAYOUT_FILE).exists());
        SplitLayout::begin_range(dir.path(), 0).unwrap();
        SplitLayout::end_range(dir.path(), 1_000).unwrap();
        SplitLayout::end_range(dir.path(), 9_000).unwrap();
        let layout = SplitLayout::load(dir.path()).unwrap().unwrap();
        assert_eq!(layout.ranges, vec![range(0, Some(1_000))]);
    }

    #[test]
    fn contains_covers_closed_and_open_ranges() {
        let layout = SplitLayout {
            version: 1,
            ranges: vec![range(0, Some(1_000)), range(5_000, None)],
        };
        assert!(layout.contains(0));
        assert!(layout.contains(999));
        assert!(!layout.contains(1_000));
        assert!(!layout.contains(4_999));
        assert!(layout.contains(5_000));
        assert!(layout.contains(10_000_000));
        assert!(!layout.contains(-1));
    }

    #[test]
    fn ranges_ms_sorts_ranges() {
        let layout = SplitLayout {
            version: 1,
            ranges: vec![range(5_000, None), range(0, Some(1_000))],
        };
        assert_eq!(layout.ranges_ms(9_000), vec![(0, 1_000), (5_000, 9_000)]);
    }

    #[test]
    fn ranges_ms_merges_overlaps_and_drops_empty_ranges() {
        let layout = SplitLayout {
            version: 1,
            ranges: vec![
                range(500, Some(2_000)),
                range(0, Some(1_000)),
                range(3_000, Some(3_000)),
                range(4_000, Some(3_500)),
                range(1_500, None),
            ],
        };
        assert_eq!(layout.ranges_ms(2_500), vec![(0, 2_500)]);
    }

    #[test]
    fn a_range_never_ends_before_it_starts() {
        let dir = tempfile::tempdir().unwrap();
        SplitLayout::begin_range(dir.path(), 5_000).unwrap();
        SplitLayout::end_range(dir.path(), 4_000).unwrap();
        SplitLayout::begin_range(dir.path(), 6_000).unwrap();
        SplitLayout::begin_range(dir.path(), 1_000).unwrap();
        let layout = SplitLayout::load(dir.path()).unwrap().unwrap();
        assert_eq!(layout.ranges[0], range(5_000, Some(5_000)));
        assert_eq!(layout.ranges[1], range(6_000, Some(6_000)));
    }

    #[test]
    fn newer_version_loads_as_none_and_is_not_overwritten_by_begin_range() {
        let dir = tempfile::tempdir().unwrap();
        let raw = r#"{"version":2,"ranges":[{"start_ms":10,"end_ms":20}]}"#;
        std::fs::write(dir.path().join(SPLIT_LAYOUT_FILE), raw).unwrap();
        assert_eq!(SplitLayout::load(dir.path()).unwrap(), None);
        SplitLayout::begin_range(dir.path(), 1_000).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join(SPLIT_LAYOUT_FILE)).unwrap(),
            raw
        );
    }

    #[test]
    fn concurrent_begins_keep_every_range_and_a_stale_tmp_is_harmless() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("split-layout.json.tmp"), "junk").unwrap();
        std::thread::scope(|s| {
            for i in 0..8 {
                let path = dir.path();
                s.spawn(move || SplitLayout::begin_range(path, i * 1_000).unwrap());
            }
        });
        let layout = SplitLayout::load(dir.path()).unwrap().unwrap();
        assert_eq!(layout.ranges.len(), 8);
        assert_eq!(
            layout.ranges.iter().filter(|r| r.end_ms.is_none()).count(),
            1
        );
    }
}
