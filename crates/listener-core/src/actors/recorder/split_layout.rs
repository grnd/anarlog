//! Records which stretch of the session audio file a split stereo capture wrote, so later
//! transcription knows where the file holds two speakers instead of a mono mix.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anlg_audio_utils::split_layout::{SPLIT_LAYOUT_FILE, SplitLayout};

use crate::actors::SAMPLE_RATE;

/// Encoder padding and frame rounding make the file length differ a little from the ranges.
const STALE_SLACK_MS: i64 = 1_000;

pub(super) struct SplitRecording {
    session_dir: PathBuf,
    start_ms: i64,
    frames: u64,
}

impl SplitRecording {
    /// Opens a range at the current end of the session audio. IO errors are logged and
    /// ignored: a missing layout must never fail the recording.
    pub(super) fn begin(session_dir: &Path) -> Self {
        let start_ms = reconcile(session_dir);
        if let Err(error) = SplitLayout::begin_range(session_dir, start_ms) {
            tracing::warn!(?error, "split_layout_begin_failed");
        }
        Self {
            session_dir: session_dir.to_path_buf(),
            start_ms,
            frames: 0,
        }
    }

    /// A capture without split mode appends plain audio, so a range a crash left open must
    /// end where that audio starts.
    pub(super) fn close_open(session_dir: &Path) {
        if !session_dir.join(SPLIT_LAYOUT_FILE).exists() {
            return;
        }
        let end_ms = reconcile(session_dir);
        if let Err(error) = SplitLayout::end_range(session_dir, end_ms) {
            tracing::warn!(?error, "split_layout_close_failed");
        }
    }

    pub(super) fn wrote(&mut self, frames: usize) {
        self.frames += frames as u64;
    }

    /// Closes the range after a clean stop. A crash leaves it open, which reads as "until
    /// the end of the file".
    pub(super) fn end(self) {
        let end_ms = self.start_ms + (self.frames * 1000 / SAMPLE_RATE as u64) as i64;
        if let Err(error) = SplitLayout::end_range(&self.session_dir, end_ms) {
            tracing::warn!(?error, "split_layout_end_failed");
        }
    }
}

/// Returns the length of the existing session audio in ms. A layout that the audio no longer
/// matches (audio deleted, or replaced by a shorter import) is dropped: its offsets would
/// point into the wrong audio.
fn reconcile(session_dir: &Path) -> i64 {
    let audio_ms = mp3_duration_ms(&session_dir.join("audio.mp3")) as i64;
    let stale = match SplitLayout::load(session_dir) {
        Ok(Some(layout)) => {
            audio_ms == 0
                || layout
                    .ranges
                    .iter()
                    .filter_map(|range| range.end_ms)
                    .any(|end| end > audio_ms + STALE_SLACK_MS)
        }
        _ => false,
    };
    if stale {
        let _ = std::fs::remove_file(session_dir.join(SPLIT_LAYOUT_FILE));
    }
    audio_ms
}

/// Length of an MP3 file, summed from its frame headers; 0 when it is missing or unreadable.
fn mp3_duration_ms(path: &Path) -> u64 {
    let Ok(mut file) = File::open(path) else {
        return 0;
    };
    let Ok(file_len) = file.metadata().map(|m| m.len()) else {
        return 0;
    };
    if file_len < 4 {
        return 0;
    }
    let mut head = [0u8; 10];
    let mut pos = 0u64;
    if file.read_exact(&mut head).is_ok() && &head[..3] == b"ID3" {
        let size = head[6..10]
            .iter()
            .fold(0u64, |acc, b| (acc << 7) | u64::from(b & 0x7f));
        pos = 10 + size;
    }
    let mut duration_us: u128 = 0;
    while pos + 4 <= file_len {
        let mut header = [0u8; 4];
        if file.seek(SeekFrom::Start(pos)).is_err() || file.read_exact(&mut header).is_err() {
            break;
        }
        if let Some((frame_len, frame_samples, sample_rate)) = parse_frame(header)
            && pos + frame_len <= file_len
        {
            duration_us += u128::from(frame_samples) * 1_000_000 / u128::from(sample_rate);
            pos += frame_len;
            continue;
        }
        // Sync loss (e.g. a trailing ID3v1 tag or truncated frame before an appended capture):
        // scan forward for the next valid frame sync.
        let Some(next) = find_next_frame(&mut file, pos + 1, file_len) else {
            break;
        };
        pos = next;
    }
    (duration_us / 1_000) as u64
}

fn find_next_frame(file: &mut File, start: u64, file_len: u64) -> Option<u64> {
    const BUF_SIZE: usize = 8192;
    let mut buf = [0u8; BUF_SIZE];
    let mut offset = start;
    while offset + 4 <= file_len {
        file.seek(SeekFrom::Start(offset)).ok()?;
        let n = file.read(&mut buf).ok()?;
        if n < 4 {
            return None;
        }
        for i in 0..=n - 4 {
            if buf[i] != 0xFF {
                continue;
            }
            let header = [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]];
            let Some((frame_len, _, _)) = parse_frame(header) else {
                continue;
            };
            let candidate = offset + i as u64;
            let next_pos = candidate + frame_len;
            if next_pos > file_len {
                continue;
            }
            if next_pos == file_len {
                return Some(candidate);
            }
            if next_pos + 4 <= file_len {
                let mut next_header = [0u8; 4];
                if file.seek(SeekFrom::Start(next_pos)).is_ok()
                    && file.read_exact(&mut next_header).is_ok()
                    && parse_frame(next_header).is_some()
                {
                    return Some(candidate);
                }
            }
        }
        if offset + n as u64 >= file_len {
            break;
        }
        offset += (n - 3) as u64;
    }
    None
}

/// Layer III only: `(frame length in bytes, samples per frame, sample rate)`.
fn parse_frame(h: [u8; 4]) -> Option<(u64, u64, u64)> {
    if h[0] != 0xFF || h[1] & 0xE0 != 0xE0 || h[1] >> 1 & 0b11 != 0b01 {
        return None;
    }
    let version = h[1] >> 3 & 0b11;
    let bitrate_index = (h[2] >> 4) as usize;
    let rate_index = (h[2] >> 2 & 0b11) as usize;
    if version == 0b01 || bitrate_index == 0 || bitrate_index == 15 || rate_index == 3 {
        return None;
    }
    let mpeg1 = version == 0b11;
    const V1: [u64; 15] = [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
    ];
    const V2: [u64; 15] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];
    let bitrate = if mpeg1 { V1 } else { V2 }[bitrate_index] * 1000;
    let base_rate = [44_100, 48_000, 32_000][rate_index];
    let sample_rate = match version {
        0b11 => base_rate,
        0b10 => base_rate / 2,
        _ => base_rate / 4,
    };
    let padding = u64::from(h[2] >> 1 & 1);
    let samples = if mpeg1 { 1152 } else { 576 };
    Some((
        samples / 8 * bitrate / sample_rate + padding,
        samples,
        sample_rate,
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::chunks::{ChunkedSink, PersistConfig};
    use super::*;

    /// One capture as the writer thread runs it: record, finish the sink, close the range.
    fn record_capture(dir: &Path, split: bool, secs: usize) {
        let config = PersistConfig::new(Arc::new(|_| {}));
        let mut sink = ChunkedSink::new(dir, 1, 0, true, config).unwrap();
        let mut range = split.then(|| SplitRecording::begin(dir));
        let second = vec![0.1_f32; SAMPLE_RATE as usize];
        for _ in 0..secs {
            sink.write(&second, &second).unwrap();
            if let Some(range) = range.as_mut() {
                range.wrote(second.len());
            }
        }
        sink.finish().unwrap();
        if let Some(range) = range {
            range.end();
        }
    }

    #[test]
    fn split_captures_record_one_closed_range_each_after_the_existing_audio() {
        let dir = tempfile::tempdir().unwrap();

        record_capture(dir.path(), true, 60);
        let first = SplitLayout::load(dir.path()).unwrap().unwrap();
        assert_eq!(first.ranges.len(), 1);
        assert_eq!(first.ranges[0].start_ms, 0);
        let first_end = first.ranges[0]
            .end_ms
            .expect("range closed on a clean stop");
        assert!((first_end - 60_000).abs() <= 100, "ended at {first_end}");

        // A later capture is appended to the same audio file, so its range starts after it.
        record_capture(dir.path(), true, 2);
        let layout = SplitLayout::load(dir.path()).unwrap().unwrap();
        assert_eq!(layout.ranges.len(), 2);
        let second = &layout.ranges[1];
        assert!(
            (second.start_ms - 60_000).abs() <= 150,
            "second range starts at {}",
            second.start_ms
        );
        assert!(second.end_ms.is_some_and(|end| end > second.start_ms));
    }

    #[test]
    fn a_crashed_open_range_ends_where_a_later_plain_capture_starts() {
        let dir = tempfile::tempdir().unwrap();
        record_capture(dir.path(), true, 3);
        // Simulate a crash: reopen the range as it stood mid-recording.
        let mut layout = SplitLayout::load(dir.path()).unwrap().unwrap();
        layout.ranges[0].end_ms = None;
        std::fs::write(
            dir.path().join(SPLIT_LAYOUT_FILE),
            serde_json::to_vec(&layout).unwrap(),
        )
        .unwrap();

        SplitRecording::close_open(dir.path());
        record_capture(dir.path(), false, 2);

        let layout = SplitLayout::load(dir.path()).unwrap().unwrap();
        let end = layout.ranges[0].end_ms.expect("closed");
        assert!((end - 3_000).abs() <= 150, "closed at {end}");
        assert!(!layout.contains(3_500));
    }

    #[test]
    fn a_layout_is_dropped_when_its_audio_is_gone_or_replaced() {
        let dir = tempfile::tempdir().unwrap();
        record_capture(dir.path(), true, 5);
        std::fs::remove_file(dir.path().join("audio.mp3")).unwrap();

        record_capture(dir.path(), true, 1);
        let layout = SplitLayout::load(dir.path()).unwrap().unwrap();
        assert_eq!(layout.ranges.len(), 1);
        assert_eq!(layout.ranges[0].start_ms, 0);

        // Replaced by a shorter file: the old range no longer fits.
        SplitLayout::begin_range(dir.path(), 0).unwrap();
        SplitLayout::end_range(dir.path(), 60_000).unwrap();
        record_capture(dir.path(), true, 1);
        let layout = SplitLayout::load(dir.path()).unwrap().unwrap();
        assert_eq!(layout.ranges.len(), 1, "{layout:?}");
    }

    #[test]
    fn a_non_split_capture_writes_no_split_layout() {
        let dir = tempfile::tempdir().unwrap();
        record_capture(dir.path(), false, 1);
        assert!(dir.path().join("audio.mp3").exists());
        assert!(SplitLayout::load(dir.path()).unwrap().is_none());
    }

    #[test]
    fn mp3_duration_handles_mixed_sample_rates_and_mid_file_tags() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audio.mp3");
        // MPEG1 Layer III, 128 kbps, 44.1 kHz, no padding: 417 bytes, 1152 samples (~26.12 ms)
        let frame_44k = [0xFFu8, 0xFB, 0x90, 0x00];
        let (len_44k, _, _) = parse_frame(frame_44k).unwrap();
        // MPEG2.5 Layer III, 32 kbps, 16 kHz (or MPEG2 16 kHz: version=0b10, rate_index=2 -> 16 kHz, 32 kbps -> 144 bytes, 576 samples = 36 ms)
        let frame_16k = [0xFFu8, 0xF3, 0x48, 0x00];
        let (len_16k, _, rate_16k) = parse_frame(frame_16k).unwrap();
        assert_eq!(rate_16k, 16_000);

        let mut file = File::create(&path).unwrap();
        // 100 frames at 44.1 kHz (~2612 ms)
        for _ in 0..100 {
            file.write_all(&frame_44k).unwrap();
            file.write_all(&vec![0u8; len_44k as usize - 4]).unwrap();
        }
        // Trailing 128-byte ID3v1 tag left before an appended capture
        let mut id3v1 = [0u8; 128];
        id3v1[..3].copy_from_slice(b"TAG");
        file.write_all(&id3v1).unwrap();
        // 50 frames at 16 kHz (1800 ms)
        for _ in 0..50 {
            file.write_all(&frame_16k).unwrap();
            file.write_all(&vec![0u8; len_16k as usize - 4]).unwrap();
        }
        drop(file);

        let duration = mp3_duration_ms(&path);
        // Expected: ~2612 + 1800 = ~4412 ms (NOT 115200+28800 / 16 = 9000 ms, and NOT 2612 ms).
        assert!(
            (duration as i64 - 4_412).abs() <= 20,
            "expected ~4412 ms, got {duration} ms"
        );
    }
}
