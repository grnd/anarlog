//! Cross-channel bleed detection for two lavalier mics recorded as one stereo input: each mic
//! also hears the other speaker, only quieter. How much quieter (the coupling) depends on how far
//! apart the mics sit, so it is measured from the recording itself: while one speaker talks
//! alone, the other mic's level relative to theirs is pure bleed. A word is bleed when its own
//! channel carries no more than that predicted leak of the other channel.

const FRAME_MS: i64 = 20;
/// Mean square below which a frame is too quiet to be anyone speaking (~-50 dBFS).
const NOISE_FLOOR: f64 = 1e-5;
/// A frame measures the coupling only when the other channel dominates by at least 10 dB.
const SOLO_DOMINANCE: f64 = 10.0;
/// Solo frames needed before a coupling estimate exists (0.5 s).
const MIN_SOLO_FRAMES: usize = 25;
/// A word is bleed when its own channel is within 5 dB (power factor √10) of the predicted
/// leak. Measured on a 2.5-hour two-lavalier recording: 5 dB removed 87% of bleed duplicates
/// vs 75% at 3 dB, with no louder original dropped.
const BLEED_TOLERANCE: f64 = 3.162_277_660_168_379_5;
/// How far (5 s) a word's window end may be from the cached estimate's before it is recomputed.
const REFRESH_FRAMES: usize = 250;
/// Only the 30 minutes of frames up to a word's end feed its coupling estimate. This bounds the
/// cost of each refresh on multi-hour sessions; coupling is a property of mic placement, which a
/// 30-minute window tracks. Ending the window at the word, not at the latest audio, judges a
/// word in a whole recording as live filtering judged it then.
const COUPLING_WINDOW_FRAMES: usize = 90_000;

/// How far (300 ms) apart in time the two channels' copies of one word may be: each channel is
/// transcribed on its own, so the same speech gets slightly different timestamps.
pub const MATCH_SLACK_MS: i64 = 300;

fn normalize(word: &str) -> Vec<char> {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// True when two words are the same word as heard on two mics: equal ignoring case and
/// punctuation, or one edit apart when both have 4+ characters (inflections, small ASR slips).
/// Words with no letters or digits never match.
pub fn same_word(a: &str, b: &str) -> bool {
    let (a, b) = (normalize(a), normalize(b));
    if a.is_empty() || b.is_empty() {
        return false;
    }
    a == b || (a.len() >= 4 && b.len() >= 4 && within_one_edit(&a, &b))
}

fn within_one_edit(a: &[char], b: &[char]) -> bool {
    if a.len().abs_diff(b.len()) > 1 {
        return false;
    }
    let prefix = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    let (a, b) = (&a[prefix..], &b[prefix..]);
    match a.len().cmp(&b.len()) {
        std::cmp::Ordering::Equal => a.get(1..) == b.get(1..),
        std::cmp::Ordering::Less => a == &b[1..],
        std::cmp::Ordering::Greater => &a[1..] == b,
    }
}

/// True when `other` (word, start_ms, end_ms) holds the same word as `text` within
/// `MATCH_SLACK_MS` of `[start_ms, end_ms]`. A loud-enough word is bleed only with such a twin.
pub fn has_matching_word<'a>(
    text: &str,
    start_ms: i64,
    end_ms: i64,
    other: impl IntoIterator<Item = (&'a str, i64, i64)>,
) -> bool {
    other.into_iter().any(|(word, other_start, other_end)| {
        other_start <= end_ms.saturating_add(MATCH_SLACK_MS)
            && other_end >= start_ms.saturating_sub(MATCH_SLACK_MS)
            && same_word(text, word)
    })
}

#[derive(Debug, Clone, Default)]
struct Frame {
    sum_sq: f64,
    count: u32,
}

impl Frame {
    fn mean_square(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum_sq / f64::from(self.count))
    }
}

/// Per-20 ms energy of one channel, indexed by session time. Writes to the same time
/// accumulate, which keeps replayed audio from skewing one channel against the other.
#[derive(Debug, Clone)]
pub struct EnergyTrack {
    sample_rate: u32,
    frames: Vec<Frame>,
    /// End of the latest written sample, so a word running past it inside the last frame is
    /// not judged on the fraction of that frame recorded so far.
    written_end_ms: i64,
    /// Frames `clear_from` cut partway, which a replay starting inside them refills only from
    /// that point, each with the time up to which it has been rewritten contiguously from its
    /// start. Spans reaching past that have no energy, as a fraction of a frame is not a word's.
    incomplete: Vec<(usize, i64)>,
    /// All audio is written (batch), so a word ending past `written_end_ms` is judged on the
    /// part that exists.
    complete: bool,
}

impl EnergyTrack {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            frames: Vec::new(),
            written_end_ms: 0,
            incomplete: Vec::new(),
            complete: false,
        }
    }

    pub fn mark_complete(&mut self) {
        self.complete = true;
    }

    pub fn write(&mut self, start_ms: i64, samples: impl IntoIterator<Item = f32>) {
        if self.sample_rate == 0 {
            return;
        }
        let rate = i64::from(self.sample_rate);
        let mut written = 0i64;
        for (i, sample) in samples.into_iter().enumerate() {
            written = i as i64 + 1;
            if !sample.is_finite() {
                continue;
            }
            let at_ms = start_ms + (i as i64 * 1000) / rate;
            if at_ms < 0 {
                continue;
            }
            let index = (at_ms / FRAME_MS) as usize;
            if index >= self.frames.len() {
                self.frames.resize(index + 1, Frame::default());
            }
            let frame = &mut self.frames[index];
            frame.sum_sq += f64::from(sample) * f64::from(sample);
            frame.count += 1;
        }
        if written > 0 {
            let end_ms = start_ms + written * 1000 / rate;
            self.written_end_ms = self.written_end_ms.max(end_ms);
            // A cut frame is whole again only once writes have rewritten all of it, from its
            // start; a write that leaves a gap before its own start does not count.
            self.incomplete.retain_mut(|(frame, filled_to)| {
                if start_ms <= *filled_to && end_ms > *filled_to {
                    *filled_to = end_ms;
                }
                *filled_to < (*frame as i64 + 1) * FRAME_MS
            });
        }
    }

    pub fn write_i16_le(&mut self, start_ms: i64, bytes: &[u8]) {
        self.write(
            start_ms,
            bytes
                .chunks_exact(2)
                .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0),
        );
    }

    /// Forgets everything from the frame holding `ms` on, so audio rewritten from there does
    /// not pile onto different audio already recorded at those times. When `ms` is not on a
    /// frame boundary, that frame is left incomplete: rewriting starts partway into it.
    pub fn clear_from(&mut self, ms: i64) {
        let kept = ms.max(0) / FRAME_MS;
        self.frames.truncate(kept as usize);
        self.incomplete.retain(|&(frame, _)| (frame as i64) < kept);
        if ms > 0 && ms % FRAME_MS != 0 {
            self.incomplete.push((kept as usize, kept * FRAME_MS));
        }
        self.written_end_ms = self.written_end_ms.min(kept * FRAME_MS);
        self.complete = false;
    }

    /// Mean square over `[start_ms, end_ms)`, or `None` unless audio covers every frame of the
    /// span and runs to its end: a partly recorded word would be judged on a fragment of its
    /// energy. Once the track is complete, a span running past the end is clamped to it when at
    /// least one frame of it remains: provider timestamps often overshoot the file by a little.
    pub fn mean_square(&self, start_ms: i64, end_ms: i64) -> Option<f64> {
        let end_ms = if self.complete
            && end_ms > self.written_end_ms
            && self.written_end_ms - start_ms >= FRAME_MS
        {
            self.written_end_ms
        } else {
            end_ms
        };
        if end_ms <= start_ms || end_ms <= 0 || end_ms > self.written_end_ms {
            return None;
        }
        let first = (start_ms.max(0) / FRAME_MS) as usize;
        let last = ((end_ms - 1) / FRAME_MS) as usize;
        let frames = self.frames.get(first..=last)?;
        if frames.iter().any(|f| f.count == 0)
            || self
                .incomplete
                .iter()
                .any(|&(frame, filled_to)| (first..=last).contains(&frame) && end_ms > filled_to)
        {
            return None;
        }
        let (sum_sq, count) = frames.iter().fold((0.0, 0u64), |(sum, n), f| {
            (sum + f.sum_sq, n + u64::from(f.count))
        });
        Some(sum_sq / count as f64)
    }
}

#[derive(Debug, Clone, Copy)]
struct Coupling {
    /// End frame (exclusive) of the window this estimate was measured over.
    end: usize,
    /// Median power ratio of the listening mic to the speaking one while only the latter talks.
    ratio: f64,
}

/// Energy of both sides of a split stereo mic (left = channel 0's speaker, right = channel 1's)
/// and the bleed decision over them.
#[derive(Debug, Clone)]
pub struct SplitEnergy {
    left: EnergyTrack,
    right: EnergyTrack,
    /// Cached estimate per channel: `[0]` is the left mic's leak from the right speaker, `[1]`
    /// the right mic's from the left. Only measured estimates are cached.
    coupling: [Option<Coupling>; 2],
    /// Latest estimate per channel for words whose window ends before `coupling`'s. Out-of-order
    /// live words reuse it instead of rescanning the window for each one.
    older_coupling: [Option<Coupling>; 2],
    /// Full-window scans done, to test that out-of-order words do not rescan.
    #[cfg(test)]
    scans: usize,
}

impl SplitEnergy {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            left: EnergyTrack::new(sample_rate),
            right: EnergyTrack::new(sample_rate),
            coupling: [None, None],
            older_coupling: [None, None],
            #[cfg(test)]
            scans: 0,
        }
    }

    /// The audio is fully written (batch): a word that starts before the written end but ends
    /// after it is judged on the part that exists.
    pub fn mark_complete(&mut self) {
        self.left.mark_complete();
        self.right.mark_complete();
    }

    pub fn write(
        &mut self,
        start_ms: i64,
        left: impl IntoIterator<Item = f32>,
        right: impl IntoIterator<Item = f32>,
    ) {
        self.left.write(start_ms, left);
        self.right.write(start_ms, right);
    }

    pub fn write_i16_le(&mut self, start_ms: i64, left: &[u8], right: &[u8]) {
        self.left.write_i16_le(start_ms, left);
        self.right.write_i16_le(start_ms, right);
    }

    pub fn clear_from(&mut self, ms: i64) {
        self.left.clear_from(ms);
        self.right.clear_from(ms);
        self.coupling = [None, None];
        self.older_coupling = [None, None];
    }

    /// True when a word on `channel` over `[start_ms, end_ms)` is only the other mic's speaker
    /// leaking in: its own channel is no louder than the measured coupling predicts from the
    /// other channel, plus 5 dB. Keeps the word when the coupling is not yet measured, when
    /// either channel has no audio for the span, and on channels other than 0 and 1.
    pub fn is_bleed(&mut self, channel: usize, start_ms: i64, end_ms: i64) -> bool {
        if channel > 1 {
            return false;
        }
        let coupling = self.coupling(channel, end_ms);
        let (own, other) = match channel {
            0 => (&self.left, &self.right),
            _ => (&self.right, &self.left),
        };
        match (
            coupling,
            own.mean_square(start_ms, end_ms),
            other.mean_square(start_ms, end_ms),
        ) {
            (Some(k), Some(own), Some(other)) => {
                other >= NOISE_FLOOR && own <= k * other * BLEED_TOLERANCE
            }
            _ => false,
        }
    }

    /// The estimate for `channel` over the window ending at the frame holding `end_ms` (or at
    /// the latest audio, if earlier). A measured estimate is cached so a long session does not
    /// rescan every frame per word: it is reused for words whose window ends at or after the
    /// cached one and within `REFRESH_FRAMES` of it, so words near the live edge refresh as new
    /// audio arrives and a whole recording judged in time order refreshes once per 5 s of it. A
    /// word whose window ends before the cached one is never judged by an estimate that saw
    /// later audio than it: it uses a second cached estimate for its own window, so out-of-order words close together share one scan. A
    /// missing estimate is never cached: it is cheap to retry while little audio exists, and
    /// solo speech may have arrived since.
    fn coupling(&mut self, channel: usize, end_ms: i64) -> Option<f64> {
        let (own, other) = match channel {
            0 => (&self.left, &self.right),
            _ => (&self.right, &self.left),
        };
        let frames = own.frames.len().min(other.frames.len());
        let end = if end_ms <= 0 {
            0
        } else {
            (((end_ms - 1) / FRAME_MS) as usize + 1).min(frames)
        };
        let reusable = |c: &Coupling| (c.end..c.end + REFRESH_FRAMES).contains(&end);
        let cached = self.coupling[channel];
        if let Some(cached) = cached.filter(reusable) {
            return Some(cached.ratio);
        }
        let older = cached.is_some_and(|cached| end < cached.end);
        if older && let Some(c) = self.older_coupling[channel].filter(reusable) {
            return Some(c.ratio);
        }
        #[cfg(test)]
        {
            self.scans += 1;
        }
        let ratio = estimate_coupling(own, other, end)?;
        // A word from before the cached window gets its own estimate, kept apart: the live
        // edge moves forward, and its cached estimate should stay there.
        let slot = if older {
            &mut self.older_coupling[channel]
        } else {
            &mut self.coupling[channel]
        };
        *slot = Some(Coupling { end, ratio });
        Some(ratio)
    }
}

/// The `COUPLING_WINDOW_FRAMES` frames ending at (excluding) frame `end`.
fn coupling_window(end: usize) -> std::ops::Range<usize> {
    end.saturating_sub(COUPLING_WINDOW_FRAMES)..end
}

/// Median of `own / other` over the window's frames where only `other`'s speaker talks, or
/// `None` until there are enough of them. `end` must not exceed either track's frames.
fn estimate_coupling(own: &EnergyTrack, other: &EnergyTrack, end: usize) -> Option<f64> {
    let window = coupling_window(end);
    let mut ratios: Vec<f64> = own.frames[window.clone()]
        .iter()
        .zip(&other.frames[window])
        .filter_map(|(own, other)| {
            let (own, other) = (own.mean_square()?, other.mean_square()?);
            (other >= NOISE_FLOOR && other >= own * SOLO_DOMINANCE).then(|| own / other)
        })
        .collect();
    if ratios.len() < MIN_SOLO_FRAMES {
        return None;
    }
    let middle = ratios.len() / 2;
    let (_, median, _) = ratios.select_nth_unstable_by(middle, f64::total_cmp);
    Some(*median)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;

    fn tone(amplitude: f32, ms: usize) -> Vec<f32> {
        (0..RATE as usize * ms / 1000)
            .map(|i| amplitude * ((i as f32) * 0.07).sin())
            .collect()
    }

    #[test]
    fn equally_loud_overlapping_speech_keeps_both_channels() {
        let mut energy = SplitEnergy::new(RATE);
        energy.write(0, tone(0.4, 1000), tone(0.4, 1000));
        assert!(!energy.is_bleed(0, 200, 600));
        assert!(!energy.is_bleed(1, 200, 600));
    }

    #[test]
    fn partly_covered_span_has_no_energy_and_keeps_the_word() {
        let mut energy = split(1000, 0);
        assert!(energy.is_bleed(1, 200, 600));
        assert_eq!(energy.right.mean_square(900, 1200), None);
        assert!(!energy.is_bleed(1, 900, 1200));
    }

    #[test]
    fn word_running_past_the_written_audio_within_a_frame_keeps_the_word() {
        let mut energy = split(1000, 0);
        let a = wave(A, 0.07, 1);
        energy.write(1000, a.clone(), a.iter().map(|s| s * BLEED_15_DB));
        // 1,001 ms written: the frame at 1,000 ms holds only 1 ms of audio.
        assert!(energy.right.mean_square(1000, 1001).is_some());
        assert_eq!(energy.right.mean_square(1000, 1019), None);
        assert!(!energy.is_bleed(1, 700, 1019));
        assert!(energy.is_bleed(1, 700, 1000));
    }

    #[test]
    fn clear_from_shrinks_the_written_extent() {
        let mut track = EnergyTrack::new(RATE);
        track.write(0, tone(0.5, 1000));
        track.clear_from(510);
        assert!(track.mean_square(400, 500).is_some());
        assert_eq!(track.mean_square(400, 510), None);
        track.write(500, tone(0.5, 5));
        assert!(track.mean_square(400, 505).is_some());
        assert_eq!(track.mean_square(400, 506), None);
    }

    #[test]
    fn unaligned_clear_leaves_the_boundary_frame_incomplete() {
        let mut track = EnergyTrack::new(RATE);
        track.write(0, tone(0.5, 1000));
        track.clear_from(510);
        track.write(510, tone(0.5, 490));
        // Frame 500-520 was replayed only from 510 ms.
        assert_eq!(track.mean_square(480, 560), None);
        assert_eq!(track.mean_square(500, 510), None);
        assert!(track.mean_square(600, 900).is_some());
        assert!(track.mean_square(400, 500).is_some());
        // Clearing earlier forgets the incomplete frame along with its audio.
        track.clear_from(300);
        track.write(300, tone(0.5, 700));
        assert!(track.mean_square(480, 560).is_some());
    }

    #[test]
    fn short_write_into_a_cut_frame_does_not_complete_it() {
        let mut track = EnergyTrack::new(RATE);
        track.write(0, tone(0.5, 1000));
        track.clear_from(510);
        track.write(500, tone(0.5, 1));
        track.write(510, tone(0.5, 490));
        // 501-510 ms of frame 500-520 was never rewritten.
        assert_eq!(track.mean_square(500, 520), None);
        assert_eq!(track.mean_square(500, 510), None);
        assert!(track.mean_square(600, 900).is_some());
    }

    #[test]
    fn write_covering_the_whole_cut_frame_completes_it() {
        let mut track = EnergyTrack::new(RATE);
        track.write(0, tone(0.5, 1000));
        track.clear_from(510);
        track.write(510, tone(0.5, 490));
        assert_eq!(track.mean_square(500, 520), None);
        track.write(500, tone(0.5, 20));
        assert!(track.mean_square(500, 520).is_some());
    }

    #[test]
    fn coupling_window_never_exceeds_the_cap() {
        let ten_hours = 10 * 60 * 60 * 1000 / FRAME_MS as usize;
        let window = coupling_window(ten_hours);
        assert_eq!(window.len(), COUPLING_WINDOW_FRAMES);
        assert_eq!(window.end, ten_hours);
        assert_eq!(coupling_window(10), 0..10);
    }

    #[test]
    fn coupling_measured_before_the_window_is_no_longer_used() {
        // One sample per 20 ms frame.
        let mut energy = SplitEnergy::new(50);
        let solo = vec![0.141f32; 1000];
        energy.write(
            0,
            solo.iter().copied(),
            solo.iter().map(|s| s * BLEED_15_DB),
        );
        let loud = vec![0.4f32; COUPLING_WINDOW_FRAMES];
        energy.write(20_000, loud.iter().copied(), loud.iter().copied());
        // A short bleed-shaped span after the loud stretch: too few solo frames on its own.
        let end_ms = 20_000 + COUPLING_WINDOW_FRAMES as i64 * FRAME_MS;
        let word = [0.141f32; 20];
        energy.write(
            end_ms,
            word.iter().copied(),
            word.iter().map(|s| s * BLEED_15_DB),
        );
        // Its window ends at the word, and the 1000 solo frames lie before it.
        assert!(!energy.is_bleed(1, end_ms, end_ms + 400));
        // A word among the solo frames is still judged by them.
        assert!(energy.is_bleed(1, 200, 600));
    }

    #[test]
    fn coupling_is_measured_up_to_each_words_own_time() {
        // A talks alone for 30 s. The right mic hears A 20 dB down for the first 10 s, then
        // 12 dB down once the mics are moved closer. A right-channel word at 14 dB below A
        // is 6 dB above the early leak but within 5 dB of the later one.
        const LEAK_EARLY: f32 = 0.1;
        const LEAK_LATE: f32 = 0.251;
        const WORD: f32 = 0.2;
        let a = wave(A, 0.07, 30_000);
        let per_ms = RATE as usize / 1000;
        let gain_at = |i: usize| {
            let ms = i / per_ms;
            if (5_000..5_400).contains(&ms) || (25_000..25_400).contains(&ms) {
                WORD
            } else if ms < 10_000 {
                LEAK_EARLY
            } else {
                LEAK_LATE
            }
        };
        let mut energy = SplitEnergy::new(RATE);
        energy.write(
            0,
            a.clone(),
            a.iter().enumerate().map(|(i, s)| s * gain_at(i)),
        );
        // The later word is bleed under the later coupling, so the early one would be too
        // if it were judged against the recording's tail.
        assert!(energy.is_bleed(1, 25_000, 25_400));
        assert!(!energy.is_bleed(1, 5_000, 5_400));
    }

    #[test]
    fn words_in_time_order_refresh_the_estimate_at_most_once_per_five_seconds() {
        // One sample per 20 ms frame; 60 s of A alone.
        let mut energy = SplitEnergy::new(50);
        let solo = vec![0.141f32; 3000];
        energy.write(
            0,
            solo.iter().copied(),
            solo.iter().map(|s| s * BLEED_15_DB),
        );
        let mut refreshes = 0;
        let mut last_end = None;
        // From 1 s on, once there is enough solo speech for an estimate.
        for word in 0..295 {
            let start_ms = 1000 + word * 200;
            assert!(energy.is_bleed(1, start_ms, start_ms + 200));
            let end = energy.coupling[1].unwrap().end;
            if last_end != Some(end) {
                refreshes += 1;
                last_end = Some(end);
            }
        }
        assert!(refreshes <= 60_000 / 5_000 + 1, "{refreshes} refreshes");
    }

    #[test]
    fn missing_energy_keeps_the_word() {
        let mut energy = split(1000, 0);
        assert!(!energy.is_bleed(1, 5_000, 5_400));
        assert!(!energy.is_bleed(1, 600, 600));
        assert!(!energy.is_bleed(2, 200, 600));
    }

    // Speaker A at -20 dBFS RMS on the left mic; B at -28 dBFS on the right;
    // each mic hears the other speaker 15 dB down.
    const A: f32 = 0.141;
    const B: f32 = 0.056;
    const BLEED_15_DB: f32 = 0.178;

    fn wave(amplitude: f32, step: f32, ms: usize) -> Vec<f32> {
        (0..RATE as usize * ms / 1000)
            .map(|i| amplitude * ((i as f32) * step).sin())
            .collect()
    }

    fn mix(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    /// `solo_ms` of A alone, then `overlap_ms` of A and B talking at once.
    fn split(solo_ms: usize, overlap_ms: usize) -> SplitEnergy {
        let mut energy = SplitEnergy::new(RATE);
        write_split(&mut energy, solo_ms, overlap_ms);
        energy
    }

    fn write_split(energy: &mut SplitEnergy, solo_ms: usize, overlap_ms: usize) {
        let a = wave(A, 0.07, solo_ms);
        energy.write(0, a.clone(), a.iter().map(|s| s * BLEED_15_DB));
        let a = wave(A, 0.07, overlap_ms);
        let b = wave(B, 0.31, overlap_ms);
        let scaled = |w: &[f32]| w.iter().map(|s| s * BLEED_15_DB).collect::<Vec<_>>();
        energy.write(solo_ms as i64, mix(&a, &scaled(&b)), mix(&b, &scaled(&a)));
    }

    #[test]
    fn pure_bleed_is_dropped_on_the_quiet_channel() {
        let mut energy = split(1000, 0);
        assert!(energy.is_bleed(1, 200, 600));
        assert!(!energy.is_bleed(0, 200, 600));
    }

    #[test]
    fn quieter_speaker_overlapping_a_louder_one_is_kept() {
        let mut energy = split(1000, 1000);
        assert!(!energy.is_bleed(1, 1200, 1600));
        assert!(!energy.is_bleed(0, 1200, 1600));
        assert!(energy.is_bleed(1, 200, 600));
    }

    #[test]
    fn allowance_above_the_leak_is_five_db() {
        // Right mic carries A's leak (15 dB down) boosted by `boost_db`, as a word would
        // that is a little louder than the measured coupling predicts.
        let with_boost = |boost_db: f32| {
            let mut energy = split(1000, 0);
            let gain = BLEED_15_DB * 10f32.powf(boost_db / 20.0);
            let a = wave(A, 0.07, 500);
            energy.write(1000, a.clone(), a.iter().map(|s| s * gain));
            energy
        };
        assert!(with_boost(4.0).is_bleed(1, 1100, 1400));
        assert!(!with_boost(6.0).is_bleed(1, 1100, 1400));
    }

    #[test]
    fn close_seated_mics_have_no_estimate_and_keep_words() {
        // Each mic hears the other speaker only 6 dB down.
        let mut energy = SplitEnergy::new(RATE);
        let a = wave(A, 0.07, 1000);
        energy.write(0, a.clone(), a.iter().map(|s| s * 0.5));
        assert!(!energy.is_bleed(1, 200, 600));
    }

    #[test]
    fn too_little_solo_speech_keeps_words() {
        let mut energy = split(300, 0);
        assert!(!energy.is_bleed(1, 0, 300));
    }

    #[test]
    fn cached_estimate_refreshes_once_solo_speech_arrives() {
        let mut energy = SplitEnergy::new(RATE);
        energy.write(0, tone(0.4, 1000), tone(0.4, 1000));
        assert!(!energy.is_bleed(1, 200, 600));
        let a = wave(A, 0.07, 6000);
        energy.write(1000, a.clone(), a.iter().map(|s| s * BLEED_15_DB));
        // A word at the new live edge, more than `REFRESH_FRAMES` past the cached window.
        assert!(energy.is_bleed(1, 6400, 6800));
    }

    #[test]
    fn missing_estimate_is_recomputed_once_solo_speech_arrives() {
        let mut energy = SplitEnergy::new(RATE);
        let a = wave(A, 0.07, 200);
        energy.write(0, a.clone(), a.iter().map(|s| s * BLEED_15_DB));
        // Only 10 solo frames so far: no estimate, so the word is kept.
        assert!(!energy.is_bleed(1, 0, 200));
        let a = wave(A, 0.07, 800);
        energy.write(200, a.clone(), a.iter().map(|s| s * BLEED_15_DB));
        // 50 solo frames now exist, well within the refresh distance of the first query.
        assert!(energy.is_bleed(1, 400, 1000));
    }

    #[test]
    fn older_word_is_judged_by_the_coupling_up_to_its_own_time() {
        // One sample per 20 ms frame. A talks alone throughout. The right mic hears A 20 dB
        // down for the first 500 frames, then 12 dB down once the mics are moved closer. The
        // median over a window flips to the later coupling once it holds more later frames
        // than earlier ones, i.e. after frame 1000. A right-channel word 10 dB below A is
        // bleed under the later coupling but 5+ dB above the earlier one.
        const LEAK_EARLY: f32 = 0.1;
        const LEAK_LATE: f32 = 0.251;
        const WORD: f32 = 0.3;
        let a = vec![0.141f32; 1100];
        let right: Vec<f32> = a
            .iter()
            .enumerate()
            .map(|(frame, s)| {
                let gain = match frame {
                    820..840 | 1050..1070 => WORD,
                    ..500 => LEAK_EARLY,
                    _ => LEAK_LATE,
                };
                s * gain
            })
            .collect();
        let mut energy = SplitEnergy::new(50);
        energy.write(0, a.iter().copied(), right);
        // The later word caches the later coupling, 230 frames past the older word's window.
        assert!(energy.is_bleed(1, 21_000, 21_400));
        // The older word must not borrow it: judged on its own window it is not bleed.
        assert!(!energy.is_bleed(1, 16_400, 16_800));
    }

    #[test]
    fn duplicate_writes_do_not_flip_decisions() {
        let mut energy = split(1000, 1000);
        write_split(&mut energy, 1000, 1000);
        assert!(energy.is_bleed(1, 200, 600));
        assert!(!energy.is_bleed(1, 1200, 1600));
        assert!(!energy.is_bleed(0, 1200, 1600));
    }

    #[test]
    fn silence_on_both_channels_keeps_words() {
        let mut energy = split(1000, 0);
        energy.write(1000, vec![0.0; 8000], vec![0.0; 8000]);
        assert!(!energy.is_bleed(1, 1100, 1400));
    }

    #[test]
    fn clear_from_lets_rewritten_audio_replace_what_was_there() {
        let mut track = EnergyTrack::new(RATE);
        track.write(0, tone(0.5, 1000));
        track.clear_from(500);
        track.write(500, tone(0.05, 500));
        let early = track.mean_square(0, 400).unwrap();
        let late = track.mean_square(600, 1000).unwrap();
        assert!((early - 0.125).abs() < 0.01);
        assert!((late - 0.00125).abs() < 0.0001);
    }

    #[test]
    fn clear_from_drops_the_coupling_measured_on_cleared_audio() {
        let mut energy = split(1000, 0);
        assert!(energy.is_bleed(1, 200, 600));
        energy.clear_from(0);
        // Too little solo speech for a new estimate.
        write_split(&mut energy, 300, 0);
        assert!(!energy.is_bleed(1, 0, 300));
    }

    #[test]
    fn i16_writes_match_float_writes() {
        let samples = tone(0.5, 100);
        let bytes: Vec<u8> = samples
            .iter()
            .flat_map(|s| ((s * 32767.0) as i16).to_le_bytes())
            .collect();
        let mut from_f32 = EnergyTrack::new(RATE);
        let mut from_i16 = EnergyTrack::new(RATE);
        from_f32.write(40, samples);
        from_i16.write_i16_le(40, &bytes);
        let a = from_f32.mean_square(40, 140).unwrap();
        let b = from_i16.mean_square(40, 140).unwrap();
        assert!((a - b).abs() / a < 0.01);
    }

    #[test]
    fn same_word_ignores_case_and_punctuation() {
        assert!(same_word("Right,", "right"));
        assert!(same_word("Привет", "привет"));
        assert!(same_word("a", "A."));
        assert!(same_word("你", "你"));
        assert!(!same_word("你", "好"));
        assert!(same_word("Café", "cafe")); // one edit, both 4+ chars
    }

    #[test]
    fn same_word_allows_one_edit_only_for_longer_words() {
        assert!(same_word("meeting", "meetings"));
        assert!(same_word("meeting", "meating"));
        assert!(same_word("meeting", "eeting"));
        assert!(same_word("also", "alse"));
        assert!(!same_word("meeting", "meatings"));
        assert!(!same_word("cat", "cats"));
        assert!(!same_word("no", "yes"));
        assert!(!same_word("the", "then"));
    }

    #[test]
    fn same_word_never_matches_empty_words() {
        assert!(!same_word("...", "..."));
        assert!(!same_word("", ""));
        assert!(!same_word("", "a"));
    }

    #[test]
    fn matching_word_needs_overlap_within_the_slack() {
        let hit = |s, e| has_matching_word("hello", 1000, 1400, [("Hello.", s, e)]);
        assert!(hit(1050, 1450));
        assert!(!has_matching_word(
            "hello",
            1000,
            1400,
            [("hello", 3000, 3400)]
        ));
        // Only the slack overlaps: ends 300 ms before the start, starts 300 ms after the end.
        assert!(hit(500, 700));
        assert!(hit(1700, 2000));
        assert!(!hit(500, 699));
        assert!(!hit(1701, 2000));
        assert!(!has_matching_word(
            "hello",
            1000,
            1400,
            [("world", 1000, 1400)]
        ));
        assert!(!has_matching_word("hello", 1000, 1400, std::iter::empty()));
    }

    #[test]
    fn complete_track_judges_a_word_overshooting_the_file_end() {
        let mut track = EnergyTrack::new(RATE);
        track.write(0, tone(0.5, 1000));
        // Live: a word past the written end is not judged.
        assert_eq!(track.mean_square(800, 1040), None);
        track.mark_complete();
        assert!(track.mean_square(800, 1040).is_some());
        assert_eq!(track.mean_square(800, 1040), track.mean_square(800, 1000));
        // Less than one frame remains, or the word starts after the end: still not judged.
        assert_eq!(track.mean_square(990, 1040), None);
        assert_eq!(track.mean_square(1000, 1040), None);
        // clear_from reopens the track.
        track.clear_from(500);
        track.write(500, tone(0.5, 400));
        assert_eq!(track.mean_square(800, 940), None);
    }

    #[test]
    fn complete_split_energy_judges_the_final_word() {
        let mut energy = split(1000, 0);
        assert!(!energy.is_bleed(1, 700, 1040));
        energy.mark_complete();
        assert!(energy.is_bleed(1, 700, 1040));
    }

    #[test]
    fn out_of_order_words_do_not_rescan_or_change_the_answer() {
        // One sample per 20 ms frame; 60 s of A alone.
        let mut energy = SplitEnergy::new(50);
        let solo = vec![0.141f32; 3000];
        energy.write(
            0,
            solo.iter().copied(),
            solo.iter().map(|s| s * BLEED_15_DB),
        );
        // The live edge is cached first.
        assert!(energy.is_bleed(1, 59_000, 59_400));
        let scans = energy.scans;
        let in_order: Vec<bool> = (0..20)
            .map(|i| energy.is_bleed(1, 30_000 + i * 50, 30_200 + i * 50))
            .collect();
        assert!(in_order.iter().all(|&b| b));
        // One scan for the older window; the rest reuse it.
        assert_eq!(energy.scans, scans + 1);
        for _ in 0..50 {
            for end in [30_400, 30_100, 33_000, 30_050] {
                assert!(energy.is_bleed(1, end - 200, end));
            }
        }
        assert!(energy.scans <= scans + 3, "{} scans", energy.scans - scans);
    }

    #[test]
    fn zero_sample_rate_and_non_finite_samples_are_ignored() {
        let mut zero = EnergyTrack::new(0);
        zero.write(0, [0.5f32; 160]);
        assert_eq!(zero.mean_square(0, 20), None);

        let mut energy = SplitEnergy::new(RATE);
        energy.write(0, tone(0.5, 1000), tone(0.5 * BLEED_15_DB, 1000));
        // An Inf or NaN sample on the other channel must not poison frames or drop real speech.
        let mut bad = vec![0.0f32; 320];
        bad[0] = f32::INFINITY;
        bad[1] = f32::NAN;
        energy.write(1000, tone(0.5, 20), bad);
        assert!(!energy.is_bleed(0, 1000, 1020));
    }
}
