//! Drops words that are one lavalier's speaker leaking into the other mic on a
//! split stereo recording (left = channel 0 speaker, right = channel 1 speaker).

use anlg_audio_utils::SplitEnergy;
use anlg_transcribe_core::TARGET_SAMPLE_RATE;

/// Energy of each side of a stereo recording; `None` when it is not stereo.
pub(super) fn stereo_energy(file_path: &str) -> Result<Option<SplitEnergy>, String> {
    let source = anlg_audio_utils::source_from_path(file_path).map_err(|e| e.to_string())?;
    if u16::from(anlg_audio_utils::Source::channels(&source)) < 2 {
        return Ok(None);
    }

    let mut energy = SplitEnergy::new(TARGET_SAMPLE_RATE);
    let mut written = 0usize;
    anlg_audio_utils::for_each_resampled_channel_block::<_, anlg_audio_utils::Error>(
        source,
        TARGET_SAMPLE_RATE,
        |channels| {
            let start_ms = (written as u64 * 1000 / u64::from(TARGET_SAMPLE_RATE)) as i64;
            energy.write(
                start_ms,
                channels[0].iter().copied(),
                channels[1].iter().copied(),
            );
            written += channels[0].len();
            Ok(())
        },
    )
    .map_err(|e| e.to_string())?;
    Ok(Some(energy))
}

/// Removes words on either channel that are only the other mic's bleed.
/// A word's own `channel` wins when set; otherwise its transcript channel's
/// position is its identity. `transcript` is rebuilt for any alternative that
/// lost words, since the summary is read directly downstream.
pub(super) fn drop_cross_channel_bleed(
    response: &mut owhisper_interface::batch::Response,
    energy: &mut SplitEnergy,
) {
    // A lone channel whose words never name a side is a mixdown: whose words
    // they are is unknown, so keep them all.
    let any_explicit = response
        .results
        .channels
        .iter()
        .flat_map(|c| &c.alternatives)
        .flat_map(|a| &a.words)
        .any(|w| w.channel > 0);
    if response.results.channels.len() < 2 && !any_explicit {
        return;
    }
    // Judge words in time order across channels, so the coupling estimate, measured up to
    // each word, is recomputed at most once per few seconds of recording.
    let mut spans = Vec::new();
    for (index, channel) in response.results.channels.iter().enumerate() {
        for (alt, alternative) in channel.alternatives.iter().enumerate() {
            for (at, word) in alternative.words.iter().enumerate() {
                let side = if word.channel > 0 {
                    word.channel as usize
                } else {
                    index
                };
                let start_ms = (word.start * 1000.0).round() as i64;
                let end_ms = (word.end * 1000.0).round() as i64;
                spans.push((end_ms, start_ms, side, (index, alt, at)));
            }
        }
    }
    spans.sort_unstable_by_key(|&(end_ms, start_ms, ..)| (end_ms, start_ms));
    let mut bleed = std::collections::HashSet::new();
    for (end_ms, start_ms, side, key) in spans {
        if energy.is_bleed(side, start_ms, end_ms) {
            bleed.insert(key);
        }
    }
    if bleed.is_empty() {
        return;
    }

    for (index, channel) in response.results.channels.iter_mut().enumerate() {
        for (alt, alternative) in channel.alternatives.iter_mut().enumerate() {
            let before = alternative.words.len();
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
                    .join(" ");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use owhisper_interface::batch::{Alternatives, Channel, Response, Results, Word};

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
        let mut energy = stereo_energy(path.to_str().unwrap()).unwrap().unwrap();

        let mut response = response(vec![channel(0), channel(1)]);
        drop_cross_channel_bleed(&mut response, &mut energy);

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
        let mut energy = stereo_energy(path.to_str().unwrap()).unwrap().unwrap();

        // The second transcript channel's words carry the default channel 0.
        let mut response = response(vec![channel(0), channel(0)]);
        drop_cross_channel_bleed(&mut response, &mut energy);

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
        let energy = stereo_energy(path.to_str().unwrap()).unwrap().unwrap();
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
        drop_cross_channel_bleed(&mut response, &mut energy);
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
        drop_cross_channel_bleed(&mut response, &mut energy);
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
        let mut energy = stereo_energy(path.to_str().unwrap()).unwrap().unwrap();

        let at = |channel: i32, start: f64| Word {
            start,
            end: start + 0.4,
            ..word(channel)
        };
        let mut response = response(vec![Channel {
            alternatives: vec![Alternatives {
                transcript: "Hello Hello".to_string(),
                confidence: 1.0,
                words: vec![at(1, 0.2), at(1, 1.2)],
            }],
        }]);
        drop_cross_channel_bleed(&mut response, &mut energy);
        let words = &response.results.channels[0].alternatives[0].words;
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].start, 1.2);
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
        assert!(stereo_energy(path.to_str().unwrap()).unwrap().is_none());
    }
}
