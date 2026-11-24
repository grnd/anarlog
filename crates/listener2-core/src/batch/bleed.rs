//! Drops words that are one lavalier's speaker leaking into the other mic on a
//! split stereo recording (left = channel 0 speaker, right = channel 1 speaker).

use anlg_audio_utils::{SplitEnergy, has_matching_word};
use anlg_transcribe_core::TARGET_SAMPLE_RATE;

/// Time ranges (ms from the start of the file) recorded in split mode.
pub(super) type Ranges = [(i64, i64)];

pub(super) fn in_ranges(ranges: &Ranges, start_ms: i64, end_ms: i64) -> bool {
    let mid = start_ms + (end_ms - start_ms) / 2;
    ranges.iter().any(|&(from, to)| mid >= from && mid < to)
}

/// The side (0 = left, 1 = right) a word belongs to. A word's own `channel` wins when
/// set; otherwise its transcript channel's position is its identity.
pub(super) fn word_side(index: usize, word: &owhisper_interface::batch::Word) -> usize {
    if word.channel > 0 {
        word.channel as usize
    } else {
        index
    }
}

pub(super) fn ms(seconds: f64) -> i64 {
    (seconds * 1000.0).round() as i64
}

/// True when a lone channel never names a side: a mixdown, whose words cannot be
/// attributed to either person.
pub(super) fn is_mixdown(response: &owhisper_interface::batch::Response) -> bool {
    let any_explicit = response
        .results
        .channels
        .iter()
        .flat_map(|c| &c.alternatives)
        .flat_map(|a| &a.words)
        .any(|w| w.channel > 0);
    response.results.channels.len() < 2 && !any_explicit
}

/// Separator for a transcript rebuilt from words: CJK text has no spaces between words.
pub(super) fn transcript_separator(original: &str) -> &'static str {
    if original.trim().contains(' ') {
        " "
    } else {
        ""
    }
}

/// Energy of each side of a stereo recording, from the audio inside `ranges` only, so
/// earlier non-split audio never calibrates the coupling; `None` when it is not stereo.
pub(super) fn stereo_energy(
    file_path: &str,
    ranges: &Ranges,
) -> Result<Option<SplitEnergy>, String> {
    let source = anlg_audio_utils::source_from_path(file_path).map_err(|e| e.to_string())?;
    if u16::from(anlg_audio_utils::Source::channels(&source)) < 2 {
        return Ok(None);
    }

    let rate = i64::from(TARGET_SAMPLE_RATE);
    let sample_ranges: Vec<(i64, i64)> = ranges
        .iter()
        .map(|&(from, to)| {
            (
                from.saturating_mul(rate) / 1000,
                to.saturating_mul(rate) / 1000,
            )
        })
        .collect();
    let mut energy = SplitEnergy::new(TARGET_SAMPLE_RATE);
    let mut written = 0i64;
    anlg_audio_utils::for_each_resampled_channel_block::<_, anlg_audio_utils::Error>(
        source,
        TARGET_SAMPLE_RATE,
        |channels| {
            let block_end = written + channels[0].len() as i64;
            for &(from, to) in &sample_ranges {
                let (a, b) = (from.max(written), to.min(block_end));
                if a >= b {
                    continue;
                }
                let (lo, hi) = ((a - written) as usize, (b - written) as usize);
                energy.write(
                    a * 1000 / rate,
                    channels[0][lo..hi].iter().copied(),
                    channels[1][lo..hi].iter().copied(),
                );
            }
            written = block_end;
            Ok(())
        },
    )
    .map_err(|e| e.to_string())?;
    energy.mark_complete();
    Ok(Some(energy))
}

/// Removes words on either channel that are only the other mic's bleed: inside `ranges`,
/// quieter than the other mic predicts AND the other channel holds the same word at about
/// the same time. `transcript` is rebuilt for any alternative that lost words, since the
/// summary is read directly downstream.
pub(super) fn drop_cross_channel_bleed(
    response: &mut owhisper_interface::batch::Response,
    energy: &mut SplitEnergy,
    ranges: &Ranges,
) {
    // A lone channel whose words never name a side is a mixdown: whose words
    // they are is unknown, so keep them all.
    if is_mixdown(response) {
        return;
    }
    // Judge words in time order across channels, so the coupling estimate, measured up to
    // each word, is recomputed at most once per few seconds of recording.
    let mut spans = Vec::new();
    let mut by_side: [Vec<(&str, i64, i64)>; 2] = [Vec::new(), Vec::new()];
    for (index, channel) in response.results.channels.iter().enumerate() {
        for (alt, alternative) in channel.alternatives.iter().enumerate() {
            for (at, word) in alternative.words.iter().enumerate() {
                let side = word_side(index, word);
                let (start_ms, end_ms) = (ms(word.start), ms(word.end));
                if side > 1 {
                    continue;
                }
                by_side[side].push((word.word.as_str(), start_ms, end_ms));
                if in_ranges(ranges, start_ms, end_ms) {
                    spans.push((end_ms, start_ms, side, (index, alt, at), word.word.as_str()));
                }
            }
        }
    }
    spans.sort_unstable_by_key(|&(end_ms, start_ms, ..)| (end_ms, start_ms));
    let mut bleed = std::collections::HashSet::new();
    for (end_ms, start_ms, side, key, text) in spans {
        if energy.is_bleed(side, start_ms, end_ms)
            && has_matching_word(text, start_ms, end_ms, by_side[1 - side].iter().copied())
        {
            bleed.insert(key);
        }
    }
    if bleed.is_empty() {
        return;
    }

    for (index, channel) in response.results.channels.iter_mut().enumerate() {
        for (alt, alternative) in channel.alternatives.iter_mut().enumerate() {
            let before = alternative.words.len();
            let separator = transcript_separator(&alternative.transcript);
            let mut at = 0;
            alternative.words.retain(|_| {
                let keep = !bleed.contains(&(index, alt, at));
                at += 1;
                keep
            });
            if alternative.words.len() != before {
                alternative.transcript = alternative
                    .words
                    .iter()
                    .map(|w| w.punctuated_word.as_deref().unwrap_or(&w.word))
                    .collect::<Vec<_>>()
                    .join(separator);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use owhisper_interface::batch::{Alternatives, Channel, Response, Results, Word};

    const ALL: &Ranges = &[(0, i64::MAX)];

    fn write_stereo_wav(path: &std::path::Path, left_amp: f32, right_amp: f32) {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..16_000 {
            let tone = (i as f32 * 0.1).sin();
            writer.write_sample(tone * left_amp).unwrap();
            writer.write_sample(tone * right_amp).unwrap();
        }
        writer.finalize().unwrap();
    }

    fn word(channel: i32) -> Word {
        Word {
            word: "hello".to_string(),
            start: 0.2,
            end: 0.6,
            confidence: 1.0,
            channel,
            speaker: None,
            punctuated_word: Some("Hello".to_string()),
        }
    }

    fn channel(channel: i32) -> Channel {
        Channel {
            alternatives: vec![Alternatives {
                transcript: "Hello".to_string(),
                confidence: 1.0,
                words: vec![word(channel)],
            }],
        }
    }

    fn response(channels: Vec<Channel>) -> Response {
        Response {
            metadata: serde_json::Value::Null,
            results: Results { channels },
        }
    }

    #[test]
    fn drops_the_quiet_channels_word() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("split.wav");
        write_stereo_wav(&path, 0.5, 0.05);
        let mut energy = stereo_energy(path.to_str().unwrap(), ALL).unwrap().unwrap();

        let mut response = response(vec![channel(0), channel(1)]);
        drop_cross_channel_bleed(&mut response, &mut energy, ALL);

        let channels = &response.results.channels;
        assert_eq!(channels[0].alternatives[0].words.len(), 1);
        assert_eq!(channels[0].alternatives[0].transcript, "Hello");
        assert!(channels[1].alternatives[0].words.is_empty());
        assert_eq!(channels[1].alternatives[0].transcript, "");
    }

    #[test]
    fn uses_channel_position_when_words_report_zero() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("split.wav");
        write_stereo_wav(&path, 0.5, 0.05);
        let mut energy = stereo_energy(path.to_str().unwrap(), ALL).unwrap().unwrap();

        // The second transcript channel's words carry the default channel 0.
        let mut response = response(vec![channel(0), channel(0)]);
        drop_cross_channel_bleed(&mut response, &mut energy, ALL);

        assert_eq!(response.results.channels[0].alternatives[0].words.len(), 1);
        assert!(
            response.results.channels[1].alternatives[0]
                .words
                .is_empty()
        );
    }

    fn tracks() -> (SplitEnergy, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("split.wav");
        write_stereo_wav(&path, 0.5, 0.05);
        let energy = stereo_energy(path.to_str().unwrap(), ALL).unwrap().unwrap();
        (energy, dir)
    }

    #[test]
    fn single_channel_of_zero_words_is_left_untouched() {
        let (mut energy, _dir) = tracks();
        let mut response = response(vec![Channel {
            alternatives: vec![Alternatives {
                transcript: "Hello Hello".to_string(),
                confidence: 1.0,
                words: vec![word(0), word(0)],
            }],
        }]);
        drop_cross_channel_bleed(&mut response, &mut energy, ALL);
        let alternative = &response.results.channels[0].alternatives[0];
        assert_eq!(alternative.words.len(), 2);
        assert_eq!(alternative.transcript, "Hello Hello");
    }

    #[test]
    fn single_channel_with_explicit_sides_is_still_filtered() {
        let (mut energy, _dir) = tracks();
        let mut response = response(vec![Channel {
            alternatives: vec![Alternatives {
                transcript: "Hello Hello".to_string(),
                confidence: 1.0,
                words: vec![word(0), word(1)],
            }],
        }]);
        drop_cross_channel_bleed(&mut response, &mut energy, ALL);
        let alternative = &response.results.channels[0].alternatives[0];
        assert_eq!(alternative.words.len(), 1);
        assert_eq!(alternative.words[0].channel, 0);
    }

    #[test]
    fn quieter_speaker_talking_over_the_louder_one_is_kept() {
        // 1 s of the left speaker alone (right hears them 15 dB down), then
        // 1 s where the right speaker, 8 dB quieter, talks over them.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("split.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..32_000 {
            let a = 0.141 * (i as f32 * 0.07).sin();
            let b = if i < 16_000 {
                0.0
            } else {
                0.056 * (i as f32 * 0.31).sin()
            };
            writer.write_sample(a + 0.178 * b).unwrap();
            writer.write_sample(b + 0.178 * a).unwrap();
        }
        writer.finalize().unwrap();
        let mut energy = stereo_energy(path.to_str().unwrap(), ALL).unwrap().unwrap();

        let at = |channel: i32, start: f64| Word {
            start,
            end: start + 0.4,
            ..word(channel)
        };
        let mut response = response(vec![Channel {
            alternatives: vec![Alternatives {
                transcript: "Hello Hello".to_string(),
                confidence: 1.0,
                words: vec![at(0, 0.2), at(1, 0.2), at(1, 1.2)],
            }],
        }]);
        drop_cross_channel_bleed(&mut response, &mut energy, ALL);
        let words = &response.results.channels[0].alternatives[0].words;
        assert_eq!(words.len(), 2);
        assert_eq!((words[0].channel, words[1].start), (0, 1.2));
    }

    #[test]
    fn mono_recording_has_no_tracks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mono.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for _ in 0..1600 {
            writer.write_sample(1000i16).unwrap();
        }
        writer.finalize().unwrap();
        assert!(
            stereo_energy(path.to_str().unwrap(), ALL)
                .unwrap()
                .is_none()
        );
    }

    fn word_at(channel: i32, text: &str, start: f64) -> Word {
        Word {
            word: text.to_string(),
            start,
            end: start + 0.4,
            punctuated_word: Some(text.to_string()),
            ..word(channel)
        }
    }

    fn one_channel(words: Vec<Word>, transcript: &str) -> Response {
        response(vec![Channel {
            alternatives: vec![Alternatives {
                transcript: transcript.to_string(),
                confidence: 1.0,
                words,
            }],
        }])
    }

    #[test]
    fn quiet_independent_word_is_kept() {
        let (mut energy, _dir) = tracks();
        // The quiet side says something the loud side never said.
        let mut response = one_channel(
            vec![word_at(0, "hello", 0.2), word_at(1, "different", 0.2)],
            "hello different",
        );
        drop_cross_channel_bleed(&mut response, &mut energy, ALL);
        assert_eq!(response.results.channels[0].alternatives[0].words.len(), 2);
    }

    #[test]
    fn word_outside_the_split_ranges_is_never_judged() {
        let (mut energy, _dir) = tracks();
        let mut response = response(vec![channel(0), channel(1)]);
        // The matching quiet copy sits at 0.2-0.6 s, outside this range.
        drop_cross_channel_bleed(&mut response, &mut energy, &[(5_000, 6_000)]);
        assert_eq!(response.results.channels[1].alternatives[0].words.len(), 1);
    }

    #[test]
    fn earlier_non_split_audio_does_not_calibrate_the_coupling() {
        // 1 s of the left speaker alone with the right mic hearing them 20 dB down, then
        // 1 s of the right speaker alone with the left mic hearing them barely at all.
        // Only the second second is split, so the first must not feed the coupling.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..32_000 {
            let tone = (i as f32 * 0.1).sin();
            let (l, r) = if i < 16_000 {
                (0.5 * tone, 0.05 * tone)
            } else {
                (0.0, 0.5 * tone)
            };
            writer.write_sample(l).unwrap();
            writer.write_sample(r).unwrap();
        }
        writer.finalize().unwrap();

        let mut all = stereo_energy(path.to_str().unwrap(), ALL).unwrap().unwrap();
        let mut split = stereo_energy(path.to_str().unwrap(), &[(1_000, 2_000)])
            .unwrap()
            .unwrap();
        // With the whole file calibrating, the quiet right mic at 0.2 s is bleed; with only
        // the second second written, no coupling exists, so nothing is judged bleed.
        assert!(all.is_bleed(1, 200, 600));
        assert!(!split.is_bleed(1, 200, 600));
    }

    #[test]
    fn transcript_is_rebuilt_without_spaces_for_cjk() {
        let (mut energy, _dir) = tracks();
        let mut response = response(vec![
            one_channel(
                vec![word_at(0, "你好", 0.2), word_at(0, "世界", 0.7)],
                "你好世界",
            )
            .results
            .channels
            .remove(0),
            one_channel(vec![word_at(1, "你好", 0.2)], "你好")
                .results
                .channels
                .remove(0),
        ]);
        drop_cross_channel_bleed(&mut response, &mut energy, ALL);
        assert_eq!(
            response.results.channels[0].alternatives[0].transcript,
            "你好世界"
        );
        assert!(
            response.results.channels[1].alternatives[0]
                .words
                .is_empty()
        );
        assert_eq!(response.results.channels[1].alternatives[0].transcript, "");
        // The channel that lost a word rejoins the rest with no spaces.
        let mut response = one_channel(
            vec![
                word_at(0, "你好", 0.2),
                word_at(1, "你好", 0.2),
                word_at(0, "世界", 0.7),
            ],
            "你好你好世界",
        );
        drop_cross_channel_bleed(&mut response, &mut energy, ALL);
        assert_eq!(
            response.results.channels[0].alternatives[0].transcript,
            "你好世界"
        );
    }

    #[test]
    fn transcript_is_rebuilt_with_spaces_otherwise() {
        let (mut energy, _dir) = tracks();
        let mut response = one_channel(
            vec![
                word_at(0, "hello", 0.2),
                word_at(1, "hello", 0.2),
                word_at(0, "world", 0.7),
            ],
            "hello hello world",
        );
        drop_cross_channel_bleed(&mut response, &mut energy, ALL);
        assert_eq!(
            response.results.channels[0].alternatives[0].transcript,
            "hello world"
        );
    }
}
