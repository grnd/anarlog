//! Split stereo mic for providers that return every word on one mixed channel:
//! each side of the stereo file is transcribed alone, so the words keep which
//! person spoke instead of collapsing into one speaker.

use std::future::Future;
use std::io::BufWriter;
use std::path::PathBuf;

use anlg_transcribe_core::TARGET_SAMPLE_RATE;
use owhisper_client::AdapterKind;
use owhisper_interface::batch::{Alternatives, Channel, Response, Results, Word};

use super::bleed::{Ranges, in_ranges, is_mixdown, ms, word_side};
use super::{BatchParams, BatchRunOutput, upload};

/// Adapters whose batch output puts words on `MIXED_CAPTURE_CHANNEL` (or on channel 0
/// with no side) whatever the input layout, so a stereo file loses which side spoke.
/// Classified from the `MIXED_CAPTURE_CHANNEL` uses in `owhisper-client`: Soniox,
/// AssemblyAI, Cartesia, ElevenLabs, Pyannote, Meta, Mistral, SmallestAI, OpenAI, and
/// GoogleGenerativeAi always or conditionally relabel words as mixed, and the
/// OpenAI-compatible family (Groq, Together, SiliconFlow, Zai, OpenRouter, Bedrock, AWS
/// Transcribe) mixes the file down before upload. Fireworks and Speechmatics label every
/// word channel 0 and Wispr Flow mixes down. Adapters not listed (Argmax, Deepgram,
/// Gladia, ...) either report per-channel words or were not confirmed to mix, so they keep
/// the single stereo upload.
pub(super) fn is_mixed_channel_adapter(kind: AdapterKind) -> bool {
    matches!(
        kind,
        AdapterKind::Soniox
            | AdapterKind::AssemblyAI
            | AdapterKind::Cartesia
            | AdapterKind::ElevenLabs
            | AdapterKind::Pyannote
            | AdapterKind::Meta
            | AdapterKind::Mistral
            | AdapterKind::SmallestAI
            | AdapterKind::OpenAI
            | AdapterKind::GoogleGenerativeAi
            | AdapterKind::AmazonBedrock
            | AdapterKind::Groq
            | AdapterKind::Together
            | AdapterKind::SiliconFlow
            | AdapterKind::Zai
            | AdapterKind::OpenRouter
            | AdapterKind::AwsTranscribe
            | AdapterKind::Fireworks
            | AdapterKind::Speechmatics
            | AdapterKind::WisprFlow
    )
}

/// One mono file per side of a stereo input. Audio outside the split ranges is
/// silence, so the provider only hears the split part and timestamps stay on the
/// input's timeline.
pub(super) struct SideFiles {
    _dir: tempfile::TempDir,
    pub(super) left: PathBuf,
    pub(super) right: PathBuf,
}

pub(super) fn write_side_files(file_path: &str, ranges: &Ranges) -> Result<SideFiles, String> {
    let source = anlg_audio_utils::source_from_path(file_path).map_err(|e| e.to_string())?;
    if u16::from(anlg_audio_utils::Source::channels(&source)) < 2 {
        return Err("split transcription needs a stereo recording".to_string());
    }
    let dir = upload::temporary_audio_directory(file_path).map_err(|e| e.to_string())?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let paths = [dir.path().join("left.wav"), dir.path().join("right.wav")];
    let mut writers = paths
        .iter()
        .map(|path| {
            std::fs::File::create(path)
                .map_err(anlg_audio_utils::Error::from)
                .and_then(|file| {
                    hound::WavWriter::new(BufWriter::new(file), spec).map_err(Into::into)
                })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e: anlg_audio_utils::Error| e.to_string())?;

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
    let mut written = 0i64;
    anlg_audio_utils::for_each_resampled_channel_block::<_, anlg_audio_utils::Error>(
        source,
        TARGET_SAMPLE_RATE,
        |channels| {
            for (side, writer) in writers.iter_mut().enumerate() {
                let masked: Vec<f32> = channels[side]
                    .iter()
                    .enumerate()
                    .map(|(i, &sample)| {
                        let at = written + i as i64;
                        let inside = sample_ranges
                            .iter()
                            .any(|&(from, to)| at >= from && at < to);
                        if inside { sample } else { 0.0 }
                    })
                    .collect();
                for sample in anlg_audio_utils::f32_to_i16_samples(&masked) {
                    writer
                        .write_sample(sample)
                        .map_err(anlg_audio_utils::Error::from)?;
                }
            }
            written += channels[0].len() as i64;
            Ok(())
        },
    )
    .map_err(|e| e.to_string())?;
    for writer in writers {
        writer.finalize().map_err(|e| e.to_string())?;
    }
    let [left, right] = paths;
    Ok(SideFiles {
        _dir: dir,
        left,
        right,
    })
}

/// Transcribes each side of the input alone with `run` (called with the params for a
/// file and the channel count of that file) and merges the words by start time with
/// channel 0 (left) and 1 (right). `whole_file` is true when the split ranges cover
/// the whole input; otherwise the input is also transcribed as is, for the audio
/// outside the ranges.
pub(super) async fn transcribe_sides<F, Fut>(
    params: BatchParams,
    ranges: &Ranges,
    whole_file: bool,
    run: F,
) -> crate::Result<BatchRunOutput>
where
    F: Fn(BatchParams, u8) -> Fut,
    Fut: Future<Output = crate::Result<BatchRunOutput>>,
{
    let source = params.file_path.clone();
    let owned_ranges = ranges.to_vec();
    let files = tokio::task::spawn_blocking(move || write_side_files(&source, &owned_ranges))
        .await
        .map_err(|_| side_failure("The split recording could not be prepared."))?
        .map_err(|error| {
            tracing::error!(error, "split_side_files_failed");
            side_failure("The split recording could not be prepared.")
        })?;

    let mut sides = Vec::with_capacity(2);
    for path in [&files.left, &files.right] {
        let mut side_params = params.clone();
        side_params.file_path = path.to_string_lossy().into_owned();
        sides.push(run(side_params, 1).await?.response);
    }
    let whole = if whole_file {
        None
    } else {
        Some(run(params.clone(), 2).await?.response)
    };

    let [left, right]: [Response; 2] = sides.try_into().expect("two sides");
    Ok(BatchRunOutput {
        session_id: params.session_id,
        mode: super::BatchRunMode::Direct,
        response: merge_sides(whole, [left, right], ranges),
    })
}

fn side_failure(message: &str) -> crate::Error {
    crate::BatchFailure::DirectRequestFailed {
        provider: "split".to_string(),
        message: message.to_string(),
    }
    .into()
}

fn first_alternative(response: &Response) -> Option<&Alternatives> {
    response
        .results
        .channels
        .first()
        .and_then(|channel| channel.alternatives.first())
}

fn merged_separator(whole: Option<&Response>, sides: &[Response; 2]) -> &'static str {
    let transcripts: Vec<&str> = whole
        .into_iter()
        .chain(sides.iter())
        .filter_map(first_alternative)
        .map(|a| a.transcript.trim())
        .filter(|t| !t.is_empty())
        .collect();
    if transcripts.iter().any(|t| t.contains(' ')) {
        return " ";
    }
    // Single-word transcripts have no internal spaces; only omit spaces for CJK text.
    if transcripts
        .iter()
        .flat_map(|t| t.chars())
        .any(|c| c.is_ascii_alphanumeric())
    {
        " "
    } else {
        ""
    }
}

/// Side words inside the ranges, labelled with their side, merged into `whole`'s words
/// outside the ranges (or on their own), sorted by start time.
pub(super) fn merge_sides(
    whole: Option<Response>,
    sides: [Response; 2],
    ranges: &Ranges,
) -> Response {
    let separator = merged_separator(whole.as_ref(), &sides);
    let mut metadata = serde_json::Value::Null;
    let mut words: Vec<Word> = Vec::new();
    if let Some(whole) = whole {
        metadata = whole.metadata;
        for channel in whole.results.channels {
            for alternative in channel.alternatives.into_iter().take(1) {
                words.extend(
                    alternative
                        .words
                        .into_iter()
                        .filter(|w| !in_ranges(ranges, ms(w.start), ms(w.end))),
                );
            }
        }
    }
    for (side, response) in sides.into_iter().enumerate() {
        if metadata.is_null() {
            metadata = response.metadata.clone();
        }
        for channel in response.results.channels {
            for alternative in channel.alternatives.into_iter().take(1) {
                words.extend(alternative.words.into_iter().filter_map(|mut word| {
                    let inside = in_ranges(ranges, ms(word.start), ms(word.end));
                    word.channel = side as i32;
                    inside.then_some(word)
                }));
            }
        }
    }
    words.sort_by(|a, b| a.start.total_cmp(&b.start));
    let transcript = words
        .iter()
        .map(|w| w.punctuated_word.as_deref().unwrap_or(&w.word))
        .collect::<Vec<_>>()
        .join(separator);
    Response {
        metadata: if metadata.is_null() {
            serde_json::json!({})
        } else {
            metadata
        },
        results: Results {
            channels: vec![Channel {
                alternatives: vec![Alternatives {
                    transcript,
                    confidence: 1.0,
                    words,
                }],
            }],
        },
    }
}

/// Inside the split ranges each channel is exactly one speaker (channel 0 is speaker 0,
/// channel 1 is speaker 1), whatever a diarizer made of one lavalier. Words outside the
/// ranges keep their own labels. A mixdown names no side, so it is left alone unless
/// `sides_split` is true (left-only side-split responses carry explicit `channel == 0`).
pub(super) fn pin_split_speakers(response: &mut Response, ranges: &Ranges, sides_split: bool) {
    if ranges.is_empty() || (!sides_split && is_mixdown(response)) {
        return;
    }
    for (index, channel) in response.results.channels.iter_mut().enumerate() {
        for word in channel
            .alternatives
            .iter_mut()
            .flat_map(|alternative| alternative.words.iter_mut())
        {
            let side = word_side(index, word);
            if side <= 1 && in_ranges(ranges, ms(word.start), ms(word.end)) {
                word.speaker = Some(side);
            }
        }
    }
}

/// Pinned ids are 0 and 1 per channel, so a diarizer id on a word outside the ranges
/// moves up by 2 to stay a different person. Call once, after the final pin.
pub(super) fn offset_unsplit_speakers(response: &mut Response, ranges: &Ranges, sides_split: bool) {
    if ranges.is_empty() || (!sides_split && is_mixdown(response)) {
        return;
    }
    for (index, channel) in response.results.channels.iter_mut().enumerate() {
        for word in channel
            .alternatives
            .iter_mut()
            .flat_map(|alternative| alternative.words.iter_mut())
        {
            let inside = in_ranges(ranges, ms(word.start), ms(word.end));
            if word_side(index, word) <= 1 && !inside {
                word.speaker = word.speaker.map(|speaker| speaker + SPLIT_SPEAKER_OFFSET);
            }
        }
    }
}

const SPLIT_SPEAKER_OFFSET: usize = 2;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::{BatchProvider, BatchRunMode};

    fn word(text: &str, start: f64, channel: i32) -> Word {
        Word {
            word: text.to_string(),
            start,
            end: start + 0.3,
            confidence: 1.0,
            channel,
            speaker: Some(3),
            punctuated_word: Some(text.to_string()),
        }
    }

    fn response(words: Vec<Word>) -> Response {
        let transcript = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Response {
            metadata: serde_json::Value::Null,
            results: Results {
                channels: vec![Channel {
                    alternatives: vec![Alternatives {
                        transcript,
                        confidence: 1.0,
                        words,
                    }],
                }],
            },
        }
    }

    fn write_stereo(path: &std::path::Path, frames: i16) {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..frames {
            writer.write_sample(i + 1).unwrap();
            writer.write_sample(-(i + 1)).unwrap();
        }
        writer.finalize().unwrap();
    }

    fn read_mono(path: &std::path::Path) -> Vec<i16> {
        let mut reader = hound::WavReader::open(path).unwrap();
        assert_eq!(reader.spec().channels, 1);
        reader.samples::<i16>().map(Result::unwrap).collect()
    }

    #[test]
    fn stereo_is_split_into_two_mono_files_sample_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("audio.wav");
        write_stereo(&input, 16_000);

        let files = write_side_files(input.to_str().unwrap(), &[(0, i64::MAX)]).unwrap();
        let left = read_mono(&files.left);
        let right = read_mono(&files.right);
        assert_eq!(left.len(), 16_000);
        assert_eq!(left, (1..=16_000).collect::<Vec<i16>>());
        assert_eq!(right, (1..=16_000).map(|v| -v).collect::<Vec<i16>>());
    }

    #[test]
    fn audio_outside_the_ranges_is_silent() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("audio.wav");
        write_stereo(&input, 16_000);

        // 250 ms to 500 ms is samples 4000..8000.
        let files = write_side_files(input.to_str().unwrap(), &[(250, 500)]).unwrap();
        let left = read_mono(&files.left);
        assert_eq!(left.len(), 16_000);
        assert!(left[..4000].iter().all(|&s| s == 0));
        assert_eq!(left[4000..8000], (4001..=8000).collect::<Vec<i16>>()[..]);
        assert!(left[8000..].iter().all(|&s| s == 0));
    }

    #[test]
    fn mono_input_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("mono.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&input, spec).unwrap();
        writer.write_sample(1i16).unwrap();
        writer.finalize().unwrap();
        assert!(write_side_files(input.to_str().unwrap(), &[(0, 1000)]).is_err());
    }

    fn params(path: &str) -> BatchParams {
        crate::batch::test_fixtures::params(BatchProvider::Soniox, "https://example.invalid", path)
    }

    #[tokio::test]
    async fn sides_are_merged_by_start_time_with_channel_labels() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("audio.wav");
        write_stereo(&input, 16_000);

        let seen = std::sync::Mutex::new(Vec::new());
        let output = transcribe_sides(
            params(input.to_str().unwrap()),
            &[(0, i64::MAX)],
            true,
            |side_params, channels| {
                let name = std::path::Path::new(&side_params.file_path)
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned();
                seen.lock().unwrap().push((name.clone(), channels));
                let words = if name == "left.wav" {
                    vec![word("one", 0.0, 2), word("three", 0.5, 2)]
                } else {
                    vec![word("two", 0.25, 2), word("four", 0.75, 2)]
                };
                async move {
                    Ok(BatchRunOutput {
                        session_id: "session".to_string(),
                        mode: BatchRunMode::Direct,
                        response: response(words),
                    })
                }
            },
        )
        .await
        .unwrap();

        assert_eq!(
            *seen.lock().unwrap(),
            vec![("left.wav".to_string(), 1), ("right.wav".to_string(), 1)]
        );
        let alternative = &output.response.results.channels[0].alternatives[0];
        let order: Vec<_> = alternative
            .words
            .iter()
            .map(|w| (w.word.as_str(), w.channel))
            .collect();
        assert_eq!(order, [("one", 0), ("two", 1), ("three", 0), ("four", 1)]);
        assert_eq!(alternative.transcript, "one two three four");
    }

    #[test]
    fn partial_ranges_keep_the_whole_files_words_outside_them() {
        let whole = response(vec![word("before", 0.0, 2), word("inside", 5.0, 2)]);
        let left = response(vec![word("hi", 5.0, 2), word("ghost", 0.0, 2)]);
        let right = response(vec![word("yo", 5.5, 2)]);
        let merged = merge_sides(Some(whole), [left, right], &[(4_000, 8_000)]);
        let words: Vec<_> = merged.results.channels[0].alternatives[0]
            .words
            .iter()
            .map(|w| (w.word.as_str(), w.channel))
            .collect();
        assert_eq!(words, [("before", 2), ("hi", 0), ("yo", 1)]);
    }

    #[test]
    fn mixed_channel_adapters_are_listed() {
        assert!(is_mixed_channel_adapter(AdapterKind::Soniox));
        assert!(is_mixed_channel_adapter(AdapterKind::AssemblyAI));
        assert!(!is_mixed_channel_adapter(AdapterKind::Deepgram));
        assert!(is_mixed_channel_adapter(AdapterKind::Speechmatics));
        assert!(is_mixed_channel_adapter(AdapterKind::AwsTranscribe));
    }

    #[test]
    fn each_channel_is_one_speaker_only_inside_the_ranges() {
        let mut response = Response {
            metadata: serde_json::Value::Null,
            results: Results {
                channels: vec![Channel {
                    alternatives: vec![Alternatives {
                        transcript: String::new(),
                        confidence: 1.0,
                        words: vec![word("a", 1.0, 0), word("b", 1.0, 1), word("c", 9.0, 0)],
                    }],
                }],
            },
        };
        pin_split_speakers(&mut response, &[(0, 5_000)], false);
        let speakers: Vec<_> = response.results.channels[0].alternatives[0]
            .words
            .iter()
            .map(|w| w.speaker)
            .collect();
        assert_eq!(speakers, [Some(0), Some(1), Some(3)]);
    }

    #[test]
    fn unsplit_speakers_do_not_collide_with_pinned_ones() {
        let mut response = response(vec![
            word("a", 1.0, 0),
            word("b", 9.0, 0),
            word("c", 9.0, 1),
        ]);
        for w in &mut response.results.channels[0].alternatives[0].words {
            w.speaker = Some(0);
        }
        pin_split_speakers(&mut response, &[(0, 5_000)], false);
        offset_unsplit_speakers(&mut response, &[(0, 5_000)], false);
        let speakers: Vec<_> = response.results.channels[0].alternatives[0]
            .words
            .iter()
            .map(|w| w.speaker)
            .collect();
        assert_eq!(speakers, [Some(0), Some(2), Some(2)]);
    }

    #[test]
    fn left_only_split_response_pins_left_speaker() {
        let mut merged = merge_sides(
            None,
            [response(vec![word("hello", 1.0, 2)]), response(vec![])],
            &[(0, 5_000)],
        );
        pin_split_speakers(&mut merged, &[(0, 5_000)], true);
        assert_eq!(
            merged.results.channels[0].alternatives[0].words[0].speaker,
            Some(0)
        );
    }

    #[test]
    fn merge_sides_keeps_cjk_unspaced_and_single_word_english_spaced() {
        let cjk = merge_sides(
            None,
            [
                response(vec![word("你好", 0.0, 2)]),
                response(vec![word("世界", 0.5, 2)]),
            ],
            &[(0, 5_000)],
        );
        assert_eq!(
            cjk.results.channels[0].alternatives[0].transcript,
            "你好世界"
        );

        let english = merge_sides(
            Some(response(vec![word("Hello", 0.0, 2)])),
            [response(vec![word("world", 5.0, 2)]), response(vec![])],
            &[(4_000, 8_000)],
        );
        assert_eq!(
            english.results.channels[0].alternatives[0].transcript,
            "Hello world"
        );
    }
}
