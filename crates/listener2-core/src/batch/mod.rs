mod accumulator;
mod bleed;
mod consolidate;
mod diarize;
mod progressive;
mod simple;
mod split_sides;
#[cfg(test)]
mod test_fixtures;
mod upload;

use std::sync::Arc;

use owhisper_client::{AdapterKind, OpenAIAdapter};

use crate::{BatchEvent, BatchRuntime};

use consolidate::consolidate_hosted_speakers;
pub use diarize::KnownSpeaker;
use diarize::apply_local_diarization;
use progressive::run_progressive_batch_session;
use simple::{run_apple_speech_batch, run_direct_batch_for_adapter_kind, run_soniqo_batch};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, strum::Display, strum::EnumString)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum BatchProvider {
    Argmax,
    #[serde(rename = "whispercpp")]
    #[strum(serialize = "whispercpp")]
    WhisperLocal,
    Deepgram,
    Soniox,
    AssemblyAI,
    Fireworks,
    OpenAI,
    OpenRouter,
    SiliconFlow,
    Zai,
    Gladia,
    ElevenLabs,
    Pyannote,
    DashScope,
    Mistral,
    Meta,
    #[serde(alias = "hyprnote")]
    Anarlog,
    Am,
    Soniqo,
    AppleSpeech,
    AquaVoice,
    Cartesia,
    Cohere,
    #[serde(rename = "aws_transcribe")]
    #[strum(serialize = "aws_transcribe")]
    AwsTranscribe,
    #[serde(rename = "azure_speech")]
    #[strum(serialize = "azure_speech")]
    AzureSpeech,
    #[serde(rename = "google_cloud")]
    #[strum(serialize = "google_cloud")]
    GoogleCloud,
    #[serde(rename = "google_generative_ai")]
    #[strum(serialize = "google_generative_ai")]
    GoogleGenerativeAi,
    Groq,
    RevAi,
    Speechmatics,
    Together,
    Xai,
    SmallestAI,
    WisprFlow,
}

impl BatchProvider {
    pub fn to_adapter_kind(&self) -> Option<AdapterKind> {
        match self {
            Self::Argmax => Some(AdapterKind::Argmax),
            Self::Deepgram => Some(AdapterKind::Deepgram),
            Self::Soniox => Some(AdapterKind::Soniox),
            Self::AssemblyAI => Some(AdapterKind::AssemblyAI),
            Self::Fireworks => Some(AdapterKind::Fireworks),
            Self::OpenAI => Some(AdapterKind::OpenAI),
            Self::OpenRouter => Some(AdapterKind::OpenRouter),
            Self::SiliconFlow => Some(AdapterKind::SiliconFlow),
            Self::Zai => Some(AdapterKind::Zai),
            Self::Gladia => Some(AdapterKind::Gladia),
            Self::ElevenLabs => Some(AdapterKind::ElevenLabs),
            Self::Pyannote => Some(AdapterKind::Pyannote),
            Self::Mistral => Some(AdapterKind::Mistral),
            Self::Meta => Some(AdapterKind::Meta),
            Self::Anarlog => Some(AdapterKind::Anarlog),
            Self::AquaVoice => Some(AdapterKind::AquaVoice),
            Self::Cartesia => Some(AdapterKind::Cartesia),
            Self::Cohere => Some(AdapterKind::Cohere),
            Self::AwsTranscribe => Some(AdapterKind::AwsTranscribe),
            Self::AzureSpeech => Some(AdapterKind::AzureSpeech),
            Self::GoogleCloud => Some(AdapterKind::GoogleCloud),
            Self::GoogleGenerativeAi => Some(AdapterKind::GoogleGenerativeAi),
            Self::Groq => Some(AdapterKind::Groq),
            Self::RevAi => Some(AdapterKind::RevAi),
            Self::Speechmatics => Some(AdapterKind::Speechmatics),
            Self::Together => Some(AdapterKind::Together),
            Self::Xai => Some(AdapterKind::Xai),
            Self::SmallestAI => Some(AdapterKind::SmallestAI),
            Self::WisprFlow => Some(AdapterKind::WisprFlow),
            Self::Am | Self::WhisperLocal | Self::Soniqo | Self::AppleSpeech | Self::DashScope => {
                None
            }
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct BatchParams {
    pub session_id: String,
    pub provider: BatchProvider,
    pub file_path: String,
    #[serde(default)]
    pub model: Option<String>,
    pub base_url: String,
    pub api_key: String,
    #[serde(default)]
    pub languages: Vec<anlg_language::Language>,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub num_speakers: Option<u32>,
    #[serde(default)]
    pub min_speakers: Option<u32>,
    #[serde(default)]
    pub max_speakers: Option<u32>,
    /// Voiceprints of expected participants; consulted on device to label
    /// local transcripts and to consolidate hosted ones, never sent to a
    /// provider.
    #[serde(default)]
    pub known_speakers: Vec<KnownSpeaker>,
    /// The whole input file was recorded with a split stereo mic (a capture chunk).
    /// The session audio file instead lists its split ranges in `split-layout.json`.
    #[serde(default)]
    pub split_stereo_mic: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(rename_all = "snake_case")]
pub enum BatchRunMode {
    Direct,
    Streamed,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
pub struct BatchRunOutput {
    pub session_id: String,
    pub mode: BatchRunMode,
    pub response: owhisper_interface::batch::Response,
}

pub async fn run_batch(
    runtime: Arc<dyn BatchRuntime>,
    params: BatchParams,
) -> crate::Result<BatchRunOutput> {
    runtime.emit(BatchEvent::BatchStarted {
        session_id: params.session_id.clone(),
    });

    let session_id = params.session_id.clone();
    let result = run_batch_inner(runtime.clone(), params).await;

    if let Err(error) = &result {
        let (code, message) = match error {
            crate::Error::BatchFailed(failure) => (failure.code(), failure.to_string()),
            _ => (crate::BatchErrorCode::Unknown, error.to_string()),
        };

        runtime.emit(BatchEvent::BatchFailed {
            session_id,
            code,
            error: message,
        });
    } else {
        let output = result.as_ref().unwrap();

        runtime.emit(BatchEvent::BatchResponse {
            session_id: output.session_id.clone(),
            response: output.response.clone(),
            mode: output.mode,
        });
        runtime.emit(BatchEvent::BatchCompleted {
            session_id: output.session_id.clone(),
        });
    }

    result
}

/// Whether this run transcribes on-device, in which case speaker labels come
/// from the local diarizer.
pub fn uses_local_diarization(params: &BatchParams) -> bool {
    diarize::is_local_batch(params)
}

pub fn expects_progressive_batch(params: &BatchParams) -> bool {
    match params.provider {
        BatchProvider::Am | BatchProvider::OpenAI => {
            let listen_params = owhisper_interface::ListenParams {
                model: params.model.clone(),
                languages: params.languages.clone(),
                ..Default::default()
            };
            let adapter_kind = resolve_batch_adapter_kind(params, &listen_params);
            if !supports_progressive_batch(adapter_kind, listen_params.model.as_deref()) {
                return false;
            }
            !(split_sides::is_mixed_channel_adapter(adapter_kind) && has_split_ranges(params))
        }
        BatchProvider::WhisperLocal => true,
        _ => false,
    }
}

fn has_split_ranges(params: &BatchParams) -> bool {
    if params.split_stereo_mic {
        return true;
    }
    load_session_split_layout(std::path::Path::new(&params.file_path))
        .is_some_and(|layout| !layout.ranges.is_empty())
}

async fn run_batch_inner(
    runtime: Arc<dyn BatchRuntime>,
    params: BatchParams,
) -> crate::Result<BatchRunOutput> {
    let metadata_joined = tokio::task::spawn_blocking({
        let path = params.file_path.clone();
        move || anlg_audio_utils::audio_file_metadata(path)
    })
    .await;

    let metadata_result = match metadata_joined {
        Ok(result) => result,
        Err(_err) => {
            tracing::error!(error.type = "audio_metadata_task_join_failed", "audio_metadata_task_join_failed");
            return Err(crate::BatchFailure::AudioMetadataJoinFailed.into());
        }
    };

    let metadata = match metadata_result {
        Ok(metadata) => metadata,
        Err(err) => {
            let raw_error = err.to_string();
            let message = format_user_friendly_error(&raw_error);
            log_batch_failure("audio_metadata_read", &raw_error);
            return Err(crate::BatchFailure::AudioMetadataReadFailed { message }.into());
        }
    };

    let listen_params = build_listen_params(&params, metadata.channels, metadata.sample_rate);
    let (ranges, whole_file) = split_ranges(&params).await;

    let post_process = (runtime.clone(), params.clone(), listen_params.clone());
    let mut sides_split = false;
    let mut output = match params.provider {
        BatchProvider::Am | BatchProvider::OpenAI => {
            let adapter_kind = resolve_batch_adapter_kind(&params, &listen_params);
            if use_progressive_path(adapter_kind, &listen_params, &ranges) {
                run_progressive_batch_session(runtime, params, listen_params).await
            } else {
                sides_split = will_split_sides(adapter_kind, &listen_params, &ranges);
                run_direct(adapter_kind, params, listen_params, &ranges, whole_file).await
            }
        }
        BatchProvider::WhisperLocal => {
            run_progressive_batch_session(runtime, params, listen_params).await
        }
        BatchProvider::Soniqo => run_soniqo_batch(runtime, params, listen_params).await,
        BatchProvider::AppleSpeech => run_apple_speech_batch(runtime, params, listen_params).await,
        BatchProvider::DashScope => Err(crate::BatchFailure::BatchCapabilityUnsupported {
            provider: batch_provider_label(BatchProvider::DashScope),
        }
        .into()),
        ref provider => {
            let adapter_kind = provider
                .to_adapter_kind()
                .expect("all non-special BatchProvider variants have an AdapterKind mapping");
            sides_split = will_split_sides(adapter_kind, &listen_params, &ranges);
            run_direct(adapter_kind, params, listen_params, &ranges, whole_file).await
        }
    }?;

    let (runtime, params, listen_params) = post_process;
    if !ranges.is_empty() {
        drop_bleed(&params.file_path, &mut output.response, &ranges).await;
    }
    let all_words_pinned =
        !ranges.is_empty() && whole_file && (sides_split || !bleed::is_mixdown(&output.response));
    if !all_words_pinned {
        apply_local_diarization(runtime.clone(), &params, &listen_params, &mut output).await;
        consolidate_hosted_speakers(runtime, &params, &listen_params, &mut output).await;
    }
    if !ranges.is_empty() {
        split_sides::pin_split_speakers(&mut output.response, &ranges, sides_split);
        split_sides::offset_unsplit_speakers(&mut output.response, &ranges, sides_split);
    }
    Ok(output)
}

/// Time ranges (ms from the start of the input) recorded in split mode, plus whether those
/// ranges cover the whole file. Empty means a normal recording, with no split handling.
async fn split_ranges(params: &BatchParams) -> (Vec<(i64, i64)>, bool) {
    let path = std::path::PathBuf::from(&params.file_path);
    let whole = params.split_stereo_mic;
    tokio::task::spawn_blocking(move || split_ranges_blocking(&path, whole))
        .await
        .unwrap_or_default()
}

fn load_session_split_layout(
    path: &std::path::Path,
) -> Option<anlg_audio_utils::split_layout::SplitLayout> {
    let is_session_audio = path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| matches!(name, "audio.wav" | "audio.mp3" | "audio.ogg"));
    match path.parent() {
        Some(dir) if is_session_audio => anlg_audio_utils::split_layout::SplitLayout::load(dir)
            .unwrap_or_else(|error| {
                tracing::warn!(error = %error, "split_layout_load_failed");
                None
            }),
        _ => None,
    }
}

/// Small tolerance for comparing split range endpoints against decoded audio duration so a
/// 1 ms MP3 frame rounding difference does not trigger a redundant full-file transcription.
const WHOLE_FILE_SLACK_MS: i64 = 500;

fn split_ranges_blocking(path: &std::path::Path, whole: bool) -> (Vec<(i64, i64)>, bool) {
    let layout = load_session_split_layout(path);
    if !whole && layout.is_none() {
        return (Vec::new(), false);
    }
    let end_ms = file_duration_ms(&path.to_string_lossy());
    let mut ranges = layout.map(|l| l.ranges_ms(end_ms)).unwrap_or_default();
    if whole {
        ranges.push((0, end_ms));
    }
    ranges.sort_unstable();
    let mut merged: Vec<(i64, i64)> = Vec::new();
    for (from, to) in ranges {
        match merged.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => merged.push((from, to)),
        }
    }
    merged.retain(|&(from, to)| to > from);
    let whole_file = merged.len() == 1
        && merged[0].0 <= WHOLE_FILE_SLACK_MS
        && merged[0].1 >= end_ms.saturating_sub(WHOLE_FILE_SLACK_MS);
    (merged, whole_file)
}

fn will_split_sides(
    adapter_kind: AdapterKind,
    listen_params: &owhisper_interface::ListenParams,
    ranges: &[(i64, i64)],
) -> bool {
    !ranges.is_empty()
        && listen_params.channels >= 2
        && split_sides::is_mixed_channel_adapter(adapter_kind)
}

/// Runs the direct batch for `adapter_kind`. A provider that returns every word on one mixed
/// channel would lose which person spoke on a split stereo mic, so each side goes alone.
async fn run_direct(
    adapter_kind: AdapterKind,
    params: BatchParams,
    listen_params: owhisper_interface::ListenParams,
    ranges: &[(i64, i64)],
    whole_file: bool,
) -> crate::Result<BatchRunOutput> {
    if !will_split_sides(adapter_kind, &listen_params, ranges) {
        return run_direct_batch_for_adapter_kind(adapter_kind, params, listen_params).await;
    }
    split_sides::transcribe_sides(params, ranges, whole_file, |side_params, channels| {
        let listen_params = owhisper_interface::ListenParams {
            channels,
            ..listen_params.clone()
        };
        run_direct_batch_for_adapter_kind(adapter_kind, side_params, listen_params)
    })
    .await
}

/// The progressive path cannot transcribe each side alone, so a split recording on a
/// mixed-channel provider takes the direct path, which does.
fn use_progressive_path(
    adapter_kind: AdapterKind,
    listen_params: &owhisper_interface::ListenParams,
    ranges: &[(i64, i64)],
) -> bool {
    supports_progressive_batch(adapter_kind, listen_params.model.as_deref())
        && !will_split_sides(adapter_kind, listen_params, ranges)
}

/// `i64::MAX` when the duration is unknown, so open ranges run to the end of the file.
fn file_duration_ms(file_path: &str) -> i64 {
    upload::audio_duration(file_path).map_or(i64::MAX, |duration| duration.as_millis() as i64)
}

/// Loading the energy and filtering the words share one blocking task, so neither stalls
/// the async runtime on a long recording.
async fn drop_bleed(
    file_path: &str,
    response: &mut owhisper_interface::batch::Response,
    ranges: &[(i64, i64)],
) {
    let path = file_path.to_string();
    let ranges = ranges.to_vec();
    // A copy goes into the task, so a panic there cannot lose the transcript.
    let owned = response.clone();
    let joined = tokio::task::spawn_blocking(move || {
        let mut owned = owned;
        let result = match bleed::stereo_energy(&path, &ranges) {
            Ok(Some(mut energy)) => {
                bleed::drop_cross_channel_bleed(&mut owned, &mut energy, &ranges);
                Ok(())
            }
            Ok(None) => Ok(()),
            Err(error) => Err(error),
        };
        (owned, result)
    })
    .await;
    match joined {
        Ok((owned, result)) => {
            *response = owned;
            if let Err(error) = result {
                tracing::warn!(error, "batch_bleed_filter_failed");
            }
        }
        Err(error) => tracing::warn!(error = %error, "batch_bleed_filter_failed"),
    }
}

fn resolve_batch_adapter_kind(
    params: &BatchParams,
    listen_params: &owhisper_interface::ListenParams,
) -> AdapterKind {
    let resolved = AdapterKind::from_url_and_languages(
        &params.base_url,
        &listen_params.languages,
        listen_params.model.as_deref(),
    );
    if matches!(params.provider, BatchProvider::OpenAI) && resolved != AdapterKind::AmazonBedrock {
        AdapterKind::OpenAI
    } else {
        resolved
    }
}

fn supports_progressive_batch(adapter_kind: AdapterKind, model: Option<&str>) -> bool {
    match adapter_kind {
        AdapterKind::Argmax => true,
        AdapterKind::OpenAI => OpenAIAdapter::supports_progressive_batch_model(model),
        _ => false,
    }
}

fn build_listen_params(
    params: &BatchParams,
    channels: u8,
    sample_rate: u32,
) -> owhisper_interface::ListenParams {
    owhisper_interface::ListenParams {
        model: params.model.clone(),
        channels,
        sample_rate,
        languages: params.languages.clone(),
        keywords: params.keywords.clone(),
        num_speakers: params.num_speakers,
        min_speakers: params.min_speakers,
        max_speakers: params.max_speakers,
        mic_num_speakers: None,
        custom_query: None,
    }
}

pub(super) fn batch_provider_label(provider: BatchProvider) -> String {
    provider.to_string()
}

pub(super) fn session_span(session_id: &str) -> tracing::Span {
    let _ = session_id;
    tracing::info_span!("session")
}

pub(super) fn log_batch_failure(operation: &str, error: &str) {
    if anlg_user_error::is_user_error_text(error) {
        tracing::warn!(
            anarlog.operation = operation,
            error.type = "user_error",
            "batch_operation_rejected"
        );
    } else {
        tracing::error!(
            anarlog.operation = operation,
            error.type = "batch_operation_failed",
            "batch_operation_failed"
        );
    }
}

pub(super) fn format_user_friendly_error(error: &str) -> String {
    let error_lower = error.to_lowercase();

    if error_lower.contains("401") || error_lower.contains("unauthorized") {
        return "Authentication failed. Please check your API key in settings.".to_string();
    }
    if error_lower.contains("403") || error_lower.contains("forbidden") {
        return "Access denied. Your API key may not have permission for this operation."
            .to_string();
    }
    if error_lower.contains("429") || error_lower.contains("rate limit") {
        return "Rate limit exceeded. Wait a few minutes or check your provider plan's quota, then try again.".to_string();
    }
    if error_lower.contains("timeout") {
        return "Connection timed out. Please check your internet connection and try again."
            .to_string();
    }
    if error_lower.contains("connection refused")
        || error_lower.contains("failed to connect")
        || error_lower.contains("network")
    {
        return "Could not connect to the transcription service. Please check your internet connection.".to_string();
    }
    if error_lower.contains("413")
        || error_lower.contains("payload too large")
        || error_lower.contains("file too large")
        || error_lower.contains("upload limit")
        || error_lower.contains("audio duration")
    {
        return "This recording is too large for the selected transcription provider. Try another provider or split the recording.".to_string();
    }
    if error_lower.contains("invalid audio")
        || error_lower.contains("unsupported format")
        || error_lower.contains("codec")
    {
        return "The audio file format is not supported. Please try a different file.".to_string();
    }
    if error_lower.contains("file not found") || error_lower.contains("no such file") {
        return "Audio file not found. The recording may have been moved or deleted.".to_string();
    }

    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listen_params(model: Option<&str>) -> owhisper_interface::ListenParams {
        owhisper_interface::ListenParams {
            model: model.map(ToOwned::to_owned),
            languages: vec![anlg_language::ISO639::En.into()],
            ..Default::default()
        }
    }

    fn batch_params(provider: BatchProvider, base_url: &str) -> BatchParams {
        BatchParams {
            session_id: "session".to_string(),
            provider,
            file_path: "/tmp/audio.wav".to_string(),
            model: None,
            base_url: base_url.to_string(),
            api_key: "key".to_string(),
            languages: vec![anlg_language::ISO639::En.into()],
            keywords: vec![],
            num_speakers: None,
            min_speakers: None,
            max_speakers: None,
            known_speakers: vec![],
            split_stereo_mic: false,
        }
    }

    #[test]
    fn build_listen_params_preserves_speaker_options() {
        let mut params = batch_params(BatchProvider::Pyannote, "https://api.pyannote.ai");
        params.num_speakers = Some(3);
        params.min_speakers = Some(2);
        params.max_speakers = Some(4);

        let listen_params = build_listen_params(&params, 2, 48_000);
        assert_eq!(listen_params.num_speakers, Some(3));
        assert_eq!(listen_params.min_speakers, Some(2));
        assert_eq!(listen_params.max_speakers, Some(4));
        assert_eq!(listen_params.channels, 2);
        assert_eq!(listen_params.sample_rate, 48_000);
        assert!(listen_params.custom_query.is_none());
    }

    #[test]
    fn am_routes_each_backend_to_its_batch_mode() {
        let cases = [
            (
                "https://api.pyannote.ai",
                None,
                AdapterKind::Pyannote,
                false,
            ),
            (
                "https://api.deepgram.com/v1",
                None,
                AdapterKind::Deepgram,
                false,
            ),
            ("http://localhost:50060/v1", None, AdapterKind::Argmax, true),
            (
                "https://api.openai.com/v1",
                Some("gpt-4o-transcribe"),
                AdapterKind::OpenAI,
                true,
            ),
            (
                "https://api.openai.com/v1",
                Some("gpt-4o-transcribe-diarize"),
                AdapterKind::OpenAI,
                false,
            ),
        ];

        for (base_url, model, expected_adapter_kind, expected_progressive) in cases {
            let params = batch_params(BatchProvider::Am, base_url);
            let adapter_kind = resolve_batch_adapter_kind(&params, &listen_params(model));

            assert_eq!(adapter_kind, expected_adapter_kind);
            assert_eq!(
                supports_progressive_batch(adapter_kind, model),
                expected_progressive
            );
        }
    }

    #[test]
    fn only_local_batches_expect_progressive() {
        let cases = [
            (BatchProvider::Anarlog, "https://api.anarlog.so/stt", false),
            (BatchProvider::Am, "https://api.anarlog.so/stt", false),
            (BatchProvider::Am, "http://localhost:50060/v1", true),
            (BatchProvider::OpenAI, "http://localhost:9000/v1", true),
            (
                BatchProvider::OpenAI,
                "http://localhost:9000/v1?provider=amazon_bedrock",
                false,
            ),
        ];

        for (provider, base_url, expected) in cases {
            assert_eq!(
                expects_progressive_batch(&batch_params(provider, base_url)),
                expected
            );
        }
    }

    #[test]
    fn split_sessions_on_mixed_channel_providers_skip_the_progressive_path() {
        let stereo = owhisper_interface::ListenParams {
            channels: 2,
            ..listen_params(Some("gpt-4o-transcribe"))
        };
        let split = [(0, 10_000)];

        // OpenAI mixes channels and is progressive: a split session goes to the direct
        // path, where `run_direct` transcribes each side.
        assert!(!use_progressive_path(AdapterKind::OpenAI, &stereo, &split));
        assert!(split_sides::is_mixed_channel_adapter(AdapterKind::OpenAI));
        // A normal session on the same provider stays progressive.
        assert!(use_progressive_path(AdapterKind::OpenAI, &stereo, &[]));
        // A provider that keeps channels apart stays progressive.
        assert!(use_progressive_path(AdapterKind::Argmax, &stereo, &split));
        // Mono input cannot be split into sides.
        let mono = owhisper_interface::ListenParams {
            channels: 1,
            ..stereo.clone()
        };
        assert!(use_progressive_path(AdapterKind::OpenAI, &mono, &split));
        // A provider that is not progressive stays direct either way.
        assert!(!use_progressive_path(AdapterKind::Soniox, &stereo, &split));
    }

    fn write_mono_wav(path: &std::path::Path, seconds: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for _ in 0..16_000 * seconds {
            writer.write_sample(0i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn normal_recordings_have_no_split_ranges() {
        let dir = tempfile::tempdir().unwrap();
        let audio = dir.path().join("audio.wav");
        write_mono_wav(&audio, 2);
        assert_eq!(split_ranges_blocking(&audio, false), (Vec::new(), false));
    }

    #[test]
    fn session_audio_reads_the_split_layout() {
        let dir = tempfile::tempdir().unwrap();
        let audio = dir.path().join("audio.wav");
        write_mono_wav(&audio, 10);
        anlg_audio_utils::split_layout::SplitLayout::begin_range(dir.path(), 1_000).unwrap();
        anlg_audio_utils::split_layout::SplitLayout::end_range(dir.path(), 3_000).unwrap();
        anlg_audio_utils::split_layout::SplitLayout::begin_range(dir.path(), 6_000).unwrap();
        assert_eq!(
            split_ranges_blocking(&audio, false),
            (vec![(1_000, 3_000), (6_000, 10_000)], false)
        );
        // Another file in the same folder is not the session audio.
        let chunk = dir.path().join("chunk.wav");
        write_mono_wav(&chunk, 2);
        assert_eq!(split_ranges_blocking(&chunk, false), (Vec::new(), false));
        // A whole-file flag covers everything.
        assert_eq!(
            split_ranges_blocking(&audio, true),
            (vec![(0, 10_000)], true)
        );
    }

    #[test]
    fn split_openai_batches_do_not_expect_progressive_timeout() {
        let mut params = batch_params(BatchProvider::OpenAI, "http://localhost:9000/v1");
        assert!(expects_progressive_batch(&params));
        params.split_stereo_mic = true;
        assert!(!expects_progressive_batch(&params));

        let dir = tempfile::tempdir().unwrap();
        let audio = dir.path().join("audio.wav");
        write_mono_wav(&audio, 2);
        anlg_audio_utils::split_layout::SplitLayout::begin_range(dir.path(), 0).unwrap();
        params.split_stereo_mic = false;
        params.file_path = audio.to_string_lossy().into_owned();
        assert!(!expects_progressive_batch(&params));
    }

    #[test]
    fn provider_size_limit_errors_are_explained() {
        let errors = [
            r#"UnexpectedStatus { status: 400, body: "Audio file exceeds the 25 MB multipart upload limit." }"#,
            r#"UnexpectedStatus { status: 400, body: "{\"error\":{\"message\":\"audio duration 1500.012 seconds is longer than 1400 seconds which is the maximum for this model\"}}" }"#,
        ];

        for error in errors {
            let message = format_user_friendly_error(error);
            assert!(message.starts_with("This recording is too large"));
        }
    }
}
