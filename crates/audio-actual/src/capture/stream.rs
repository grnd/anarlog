use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::task::{Context, Poll};
use std::{panic::AssertUnwindSafe, panic::catch_unwind};

use anlg_aec::AEC;
use anlg_audio_sync::{SyncProbe, SyncProbeConfig, SyncProbeEvent, SyncProbeState};
use anlg_resampler::ResampleExtDynamicNew;
use futures_util::{Stream, StreamExt};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;

use super::recovery::{ChannelItem, OpenedChannel, run_channel};
use anlg_audio::{CaptureChannel, CaptureConfig, CaptureEvent, CaptureFrame, CaptureStream, Error};

use crate::mic::MicInput;
use crate::speaker::SpeakerInput;

use super::joiner::{AudioPair, Joiner};

pub(crate) type ChunkStream =
    Pin<Box<dyn Stream<Item = Result<Vec<f32>, anlg_resampler::Error>> + Send>>;

const AUDIO_SYNC_PROBE_ENV: &str = "AUDIO_SYNC_PROBE";
const AEC_MAX_REFERENCE_LAG_MS: u32 = 600;
const AEC_MIN_REFERENCE_OVERLAP_MS: u32 = 400;
const AEC_MIN_REFERENCE_RMS: f32 = 1e-4;
const AEC_MIN_MIC_RMS: f32 = 1e-4;
const AEC_MIN_REFERENCE_CORRELATION: f32 = 0.12;
const AEC_MAX_LINEAR_GAIN: f32 = 1.25;
const AEC_LINEAR_GAIN_SMOOTHING: f32 = 0.12;
const AEC_DOUBLE_TALK_RESIDUAL_RATIO: f32 = 0.08;
const AEC_ROBUST_GAIN_SEGMENTS: usize = 8;
const AEC_ROBUST_GAIN_MIN_SEGMENTS: usize = 3;

struct CaptureStreamInner {
    inner: ReceiverStream<Result<CaptureEvent, Error>>,
    cancel_token: CancellationToken,
    task: JoinHandle<()>,
}

impl Stream for CaptureStreamInner {
    type Item = Result<CaptureEvent, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl Drop for CaptureStreamInner {
    fn drop(&mut self) {
        self.cancel_token.cancel();
        self.task.abort();
    }
}

pub(crate) fn setup_mic_stream(
    sample_rate: u32,
    chunk_size: usize,
    mic_device: Option<String>,
) -> Result<ChunkStream, Error> {
    let mic = MicInput::new(mic_device).map_err(|_| Error::MicOpenFailed)?;
    mic.stream()?
        .resampled_chunks(sample_rate, chunk_size)
        .map(|stream| Box::pin(stream) as ChunkStream)
        .map_err(|_| Error::MicStreamSetupFailed)
}

pub(crate) fn setup_speaker_stream(
    sample_rate: u32,
    chunk_size: usize,
) -> Result<ChunkStream, Error> {
    let speaker = SpeakerInput::new()
        .map_err(|error| Error::SpeakerStreamInitializationFailed(error.to_string()))?;
    speaker
        .stream()
        .map_err(|error| Error::SpeakerStreamInitializationFailed(error.to_string()))?
        .resampled_chunks(sample_rate, chunk_size)
        .map(|stream| Box::pin(stream) as ChunkStream)
        .map_err(|_| Error::SpeakerStreamSetupFailed)
}

pub(crate) fn open_recovering(config: CaptureConfig) -> CaptureStream {
    let cancel_token = CancellationToken::new();
    let (tx, rx) = tokio::sync::mpsc::channel(32);
    let (mic_retry, mic_rx) = tokio::sync::mpsc::unbounded_channel();
    let (speaker_retry, speaker_rx) = tokio::sync::mpsc::unbounded_channel();
    let cancel = cancel_token.clone();
    let aec_enabled = Arc::new(AtomicBool::new(config.enable_aec));
    let aec_config = aec_enabled.clone();
    let speaker_output_retry = speaker_retry.clone();
    let task = tokio::spawn(async move {
        let (chunks_tx, chunks_rx) = tokio::sync::mpsc::channel(32);
        let rate = config.sample_rate;
        let size = config.chunk_size;
        let mic = run_channel(
            CaptureChannel::Mic,
            Arc::new(move |device, excluded| {
                if crate::mic::startup_pending() {
                    return Err(Error::MicStreamInitializationFailed(
                        "previous microphone startup is still pending".into(),
                    ));
                }
                let mut excluded = excluded.to_vec();
                let first = MicInput::new_excluding(device.clone(), &excluded).or_else(|error| {
                    if excluded.is_empty() {
                        Err(error)
                    } else {
                        excluded.clear();
                        MicInput::new_excluding(device.clone(), &excluded)
                    }
                });
                let mut input = Some(first?);
                loop {
                    let input = match input.take() {
                        Some(input) => input,
                        None => MicInput::new_excluding(device.clone(), &excluded)?,
                    };
                    let name = input.device_name();
                    match input.stream() {
                        Ok(stream) => {
                            return Ok(OpenedChannel {
                                stream: Box::pin(
                                    stream
                                        .resampled_chunks(rate, size)
                                        .map_err(|_| Error::MicStreamSetupFailed)?,
                                ),
                                device: Some(name),
                            });
                        }
                        Err(error) => {
                            excluded.push(name);
                            if crate::mic::startup_pending() {
                                return Err(error);
                            }
                        }
                    }
                }
            }),
            config.mic_device,
            mic_rx,
            chunks_tx.clone(),
            cancel.clone(),
        );
        let speaker = run_channel(
            CaptureChannel::Speaker,
            Arc::new(move |_, _| {
                Ok(OpenedChannel {
                    stream: setup_speaker_stream(rate, size)?,
                    device: None,
                })
            }),
            None,
            speaker_rx,
            chunks_tx,
            cancel.clone(),
        );
        let output = run_dual_loop(tx, cancel.clone(), rate, size, aec_config, chunks_rx);
        tokio::join!(mic, speaker, output);
    });
    CaptureStream::with_events(CaptureStreamInner {
        inner: ReceiverStream::new(rx),
        cancel_token,
        task,
    })
    .with_retry(move |device, force_mic| {
        let _ = mic_retry.send((device.clone(), force_mic));
        let _ = speaker_retry.send((device, false));
    })
    .with_output_change(move |enable_aec| {
        aec_enabled.store(enable_aec, Ordering::Release);
        let _ = speaker_output_retry.send((None, true));
    })
}

/// Captures a stereo mic's left and right channels as the mic and speaker tracks. Both tracks
/// come from one device, so they run as a single recovering channel: they become ready, fail,
/// retry and recover together, and an outage pads both with the same silence.
pub(crate) fn open_split_recovering(
    device: Option<String>,
    sample_rate: u32,
    chunk_size: usize,
) -> CaptureStream {
    let cancel_token = CancellationToken::new();
    let (tx, rx) = tokio::sync::mpsc::channel(32);
    let (retry, retry_rx) = tokio::sync::mpsc::unbounded_channel();
    let requested = Mutex::new(device.clone());
    let cancel = cancel_token.clone();
    let task = tokio::spawn(async move {
        let (pairs_tx, pairs_rx) = tokio::sync::mpsc::channel(32);
        let pinned = Mutex::new(None);
        let device = run_channel(
            CaptureChannel::Mic,
            Arc::new(move |device, _excluded| {
                let ((left, right), name) = open_split_mic(device, &pinned)?;
                let resample = |stream: crate::mic::MicStream| {
                    stream
                        .resampled_chunks(sample_rate, chunk_size)
                        .map(|stream| Box::pin(stream) as ChunkStream)
                        .map_err(|_| Error::MicStreamSetupFailed)
                };
                Ok(OpenedChannel {
                    stream: Box::pin(LockstepPairs::new(resample(left)?, resample(right)?)),
                    device: Some(name),
                })
            }),
            device,
            retry_rx,
            pairs_tx,
            cancel.clone(),
        );
        let output = run_split_loop(tx, cancel.clone(), sample_rate, chunk_size, pairs_rx);
        tokio::join!(device, output);
    });
    CaptureStream::with_events(CaptureStreamInner {
        inner: ReceiverStream::new(rx),
        cancel_token,
        task,
    })
    .with_retry(move |device, force| {
        let _ = retry.send((split_retry_device(&requested, device), force));
    })
}

/// A split capture never follows the default input, so a retry that names no device asks for the
/// one last requested; a healthy capture then sees no change and keeps running.
fn split_retry_device(requested: &Mutex<Option<String>>, device: Option<String>) -> Option<String> {
    let mut requested = requested
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if device.is_some() {
        *requested = device;
    }
    requested.clone()
}

/// Opens the stereo mic a split capture records from. Its channels are two speakers, so recovery
/// never moves to another microphone: a capture started on the default input stays on the device
/// it first opened, and while that device is absent the open fails and is retried.
fn open_split_mic(
    device: Option<String>,
    pinned: &Mutex<Option<String>>,
) -> Result<((crate::mic::MicStream, crate::mic::MicStream), String), Error> {
    if crate::mic::startup_pending() {
        return Err(Error::MicStreamInitializationFailed(
            "previous microphone startup is still pending".into(),
        ));
    }
    open_pinned(
        device,
        pinned,
        |target| {
            let input = MicInput::new(target)?;
            Ok((input.device_name(), input))
        },
        MicInput::stream_split,
    )
}

fn open_pinned<D, S>(
    device: Option<String>,
    pinned: &Mutex<Option<String>>,
    open: impl FnOnce(Option<String>) -> Result<(String, D), Error>,
    start: impl FnOnce(&D) -> Result<S, Error>,
) -> Result<(S, String), Error> {
    let mut pinned = pinned
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let target = device.or_else(|| pinned.clone());
    let (name, input) = open(target.clone())?;
    if let Some(target) = target.filter(|target| *target != name) {
        return Err(Error::MicStreamInitializationFailed(format!(
            "split stereo microphone {target} is unavailable"
        )));
    }
    let streams = start(&input)?;
    *pinned = Some(name.clone());
    Ok((streams, name))
}

/// Joins a split mic's left and right chunk streams in lockstep: a late side is waited for,
/// never silence-filled, and the pair stream ends or fails as soon as either side does.
struct LockstepPairs {
    left: ChunkStream,
    right: ChunkStream,
    joiner: Joiner,
}

impl LockstepPairs {
    fn new(left: ChunkStream, right: ChunkStream) -> Self {
        Self {
            left,
            right,
            joiner: Joiner::lockstep(),
        }
    }
}

impl Stream for LockstepPairs {
    type Item = Result<AudioPair, anlg_resampler::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        loop {
            if let Some(pair) = this.joiner.pop_pair() {
                return Poll::Ready(Some(Ok(pair)));
            }
            let left = this.left.as_mut().poll_next(cx);
            let right = this.right.as_mut().poll_next(cx);
            let mut progressed = false;
            for (polled, is_left) in [(left, true), (right, false)] {
                match polled {
                    Poll::Ready(Some(Ok(data))) => {
                        progressed = true;
                        if is_left {
                            this.joiner.push_mic(data);
                        } else {
                            this.joiner.push_speaker(data);
                        }
                    }
                    Poll::Ready(Some(Err(error))) => return Poll::Ready(Some(Err(error))),
                    Poll::Ready(None) => return Poll::Ready(None),
                    Poll::Pending => {}
                }
            }
            if !progressed {
                return Poll::Pending;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CaptureSide {
    Mic,
    Speaker,
}

pub(crate) fn open_single(chunk_stream: ChunkStream, side: CaptureSide) -> CaptureStream {
    let cancel_token = CancellationToken::new();
    let (tx, rx) = tokio::sync::mpsc::channel(32);
    let task = tokio::spawn(run_single_loop(
        tx,
        cancel_token.clone(),
        chunk_stream,
        side,
    ));

    CaptureStream::with_events(CaptureStreamInner {
        inner: ReceiverStream::new(rx),
        cancel_token,
        task,
    })
}

async fn run_dual_loop(
    tx: tokio::sync::mpsc::Sender<Result<CaptureEvent, Error>>,
    cancel_token: CancellationToken,
    sample_rate: u32,
    chunk_size: usize,
    aec_enabled: Arc<AtomicBool>,
    mut chunks: tokio::sync::mpsc::Receiver<ChannelItem>,
) {
    let mut joiner = Joiner::new();
    let mut enable_aec = aec_enabled.load(Ordering::Acquire);
    let mut aec = if enable_aec { build_aec() } else { None };
    let mut linear_echo_gain = None;
    let mut aec_reference = if aec.is_some() {
        Some(AecReferenceAligner::new(sample_rate))
    } else {
        None
    };

    let mut mic_ready = false;
    let mut speaker_ready = false;
    let mut last_frame_at = None;
    let mut gap = None;
    let mut clock = tokio::time::interval(std::time::Duration::from_secs_f64(
        chunk_size as f64 / sample_rate as f64,
    ));
    clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        let item = tokio::select! {
            _ = cancel_token.cancelled() => return,
            _ = clock.tick(), if gap.is_some() => {
                if !emit_gap(&tx, &cancel_token, sample_rate, chunk_size, gap.as_mut().unwrap()).await {
                    return;
                }
                continue;
            },
            item = chunks.recv() => item,
        };
        let Some(item) = item else {
            return;
        };
        let updated_aec = aec_enabled.load(Ordering::Acquire);
        if updated_aec != enable_aec {
            enable_aec = updated_aec;
            aec = if enable_aec { build_aec() } else { None };
            linear_echo_gain = None;
            aec_reference = aec.as_ref().map(|_| AecReferenceAligner::new(sample_rate));
        }
        match item {
            ChannelItem::Ready(channel, device) => {
                if let Some(mut gap) = gap.take()
                    && !emit_gap(&tx, &cancel_token, sample_rate, chunk_size, &mut gap).await
                {
                    return;
                }
                match channel {
                    CaptureChannel::Mic => mic_ready = true,
                    CaptureChannel::Speaker => speaker_ready = true,
                }
                if tx
                    .send(Ok(CaptureEvent::ChannelReady { channel, device }))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            ChannelItem::Failed(channel, error) => {
                match channel {
                    CaptureChannel::Mic => {
                        mic_ready = false;
                        joiner.clear_mic();
                    }
                    CaptureChannel::Speaker => {
                        speaker_ready = false;
                        joiner.clear_speaker();
                    }
                }
                if !mic_ready && !speaker_ready && gap.is_none() {
                    gap = last_frame_at.map(|at| (at, 0));
                }
                if let Some(aec) = aec.as_mut() {
                    aec.reset();
                }
                linear_echo_gain = None;
                aec_reference = aec.as_ref().map(|_| AecReferenceAligner::new(sample_rate));
                if tx
                    .send(Ok(CaptureEvent::ChannelFailed { channel, error }))
                    .await
                    .is_err()
                {
                    return;
                }
            }
            ChannelItem::Chunk(channel, data) => match channel {
                CaptureChannel::Mic => joiner.push_mic(data),
                CaptureChannel::Speaker => joiner.push_speaker(data),
            },
        }
        while let Some((raw_mic, raw_speaker)) = joiner.pop_available_pair(mic_ready, speaker_ready)
        {
            last_frame_at = Some(tokio::time::Instant::now());
            let raw_mic = Arc::<[f32]>::from(raw_mic);
            let raw_speaker = Arc::<[f32]>::from(raw_speaker);
            let aec_mic = if mic_ready && speaker_ready {
                let aligned = aec_reference
                    .as_mut()
                    .map(|aligner| aligner.align(&raw_speaker, &raw_mic))
                    .unwrap_or_else(|| AecAlignedPair {
                        mic: Arc::clone(&raw_mic),
                        speaker: Arc::clone(&raw_speaker),
                        alignment_changed: false,
                    });
                if aligned.alignment_changed {
                    if let Some(processor) = aec.as_mut() {
                        processor.reset();
                    }
                    linear_echo_gain = None;
                }
                process_aec(
                    &mut aec,
                    &mut linear_echo_gain,
                    &aligned.mic,
                    &aligned.speaker,
                )
            } else {
                None
            };
            if tx
                .send(Ok(CaptureEvent::Frame(CaptureFrame {
                    raw_mic,
                    raw_speaker,
                    aec_mic,
                })))
                .await
                .is_err()
            {
                return;
            }
        }
    }
}

// A split mic has one device channel: its pairs are emitted as they arrive, and when it fails
// the elapsed outage is padded on both tracks alike, as for a dual-channel outage.
async fn run_split_loop(
    tx: tokio::sync::mpsc::Sender<Result<CaptureEvent, Error>>,
    cancel_token: CancellationToken,
    sample_rate: u32,
    chunk_size: usize,
    mut pairs: tokio::sync::mpsc::Receiver<ChannelItem<AudioPair>>,
) {
    let mut last_frame_at = None;
    let mut gap = None;
    let mut clock = tokio::time::interval(std::time::Duration::from_secs_f64(
        chunk_size as f64 / sample_rate as f64,
    ));
    clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        let item = tokio::select! {
            _ = cancel_token.cancelled() => return,
            _ = clock.tick(), if gap.is_some() => {
                if !emit_gap(&tx, &cancel_token, sample_rate, chunk_size, gap.as_mut().unwrap()).await {
                    return;
                }
                continue;
            },
            item = pairs.recv() => item,
        };
        let Some(item) = item else {
            return;
        };
        let event = match item {
            ChannelItem::Ready(channel, device) => {
                if let Some(mut gap) = gap.take()
                    && !emit_gap(&tx, &cancel_token, sample_rate, chunk_size, &mut gap).await
                {
                    return;
                }
                CaptureEvent::ChannelReady { channel, device }
            }
            ChannelItem::Failed(channel, error) => {
                if gap.is_none() {
                    gap = last_frame_at.map(|at| (at, 0));
                }
                CaptureEvent::ChannelFailed { channel, error }
            }
            ChannelItem::Chunk(_, (left, right)) => {
                last_frame_at = Some(tokio::time::Instant::now());
                CaptureEvent::Frame(CaptureFrame {
                    raw_mic: Arc::from(left),
                    raw_speaker: Arc::from(right),
                    aec_mic: None,
                })
            }
        };
        if tx.send(Ok(event)).await.is_err() {
            return;
        }
    }
}

async fn emit_gap(
    tx: &tokio::sync::mpsc::Sender<Result<CaptureEvent, Error>>,
    cancel: &CancellationToken,
    sample_rate: u32,
    chunk_size: usize,
    gap: &mut (tokio::time::Instant, usize),
) -> bool {
    let elapsed_samples = (gap.0.elapsed().as_secs_f64() * sample_rate as f64) as usize;
    while gap.1 < elapsed_samples {
        let count = (elapsed_samples - gap.1).min(chunk_size);
        let silence = Arc::<[f32]>::from(vec![0.0; count]);
        let frame = CaptureFrame {
            raw_mic: silence.clone(),
            raw_speaker: silence,
            aec_mic: None,
        };
        tokio::select! {
            _ = cancel.cancelled() => return false,
            result = tx.send(Ok(CaptureEvent::Gap(frame))) => {
                if result.is_err() { return false; }
            }
        }
        gap.1 += count;
    }
    true
}

struct AecReferenceAligner {
    probe: SyncProbe,
    mic_delay_line: SampleDelayLine,
    speaker_delay_line: SampleDelayLine,
    last_alignment: AecAlignment,
    last_logged_state: Option<SyncProbeState>,
    last_logged_stable_lag_samples: Option<isize>,
    log_probe_events: bool,
}

struct AecAlignedPair {
    mic: Arc<[f32]>,
    speaker: Arc<[f32]>,
    alignment_changed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AecAlignment {
    None,
    DelayMic(usize),
    DelaySpeaker(usize),
}

impl AecAlignment {
    fn from_lag_samples(lag_samples: Option<isize>) -> Self {
        match lag_samples {
            Some(lag) if lag > 0 => Self::DelaySpeaker(lag as usize),
            Some(lag) if lag < 0 => Self::DelayMic(lag.unsigned_abs()),
            _ => Self::None,
        }
    }

    fn mic_delay_samples(self) -> usize {
        match self {
            Self::DelayMic(samples) => samples,
            Self::None | Self::DelaySpeaker(_) => 0,
        }
    }

    fn speaker_delay_samples(self) -> usize {
        match self {
            Self::DelaySpeaker(samples) => samples,
            Self::None | Self::DelayMic(_) => 0,
        }
    }
}

impl AecReferenceAligner {
    fn new(sample_rate: u32) -> Self {
        let max_lag_samples = ((sample_rate as usize) * (AEC_MAX_REFERENCE_LAG_MS as usize)) / 1000;
        let min_overlap_samples =
            ((sample_rate as usize) * (AEC_MIN_REFERENCE_OVERLAP_MS as usize)) / 1000;
        let mut config = SyncProbeConfig::new(sample_rate);
        config.window_samples = config
            .window_samples
            .max((max_lag_samples + min_overlap_samples).next_power_of_two());
        config.max_lag_samples = max_lag_samples.max(config.max_lag_samples);
        let probe = SyncProbe::new(config);
        let max_delay_samples = probe.config().max_lag_samples;

        Self {
            probe,
            mic_delay_line: SampleDelayLine::new(max_delay_samples),
            speaker_delay_line: SampleDelayLine::new(max_delay_samples),
            last_alignment: AecAlignment::None,
            last_logged_state: None,
            last_logged_stable_lag_samples: None,
            log_probe_events: std::env::var(AUDIO_SYNC_PROBE_ENV).ok().as_deref() == Some("1"),
        }
    }

    fn align(&mut self, raw_speaker: &[f32], raw_mic: &[f32]) -> AecAlignedPair {
        let observed = catch_unwind(AssertUnwindSafe(|| {
            self.probe.observe(raw_speaker, raw_mic)
        }));
        let event = match observed {
            Ok(event) => event,
            Err(_) => {
                tracing::error!("audio_sync_probe_panicked");
                None
            }
        };

        let alignment_changed = if let Some(event) = event {
            let alignment_changed = self.update_alignment(&event);
            if self.log_probe_events {
                self.log_probe_event(event);
            }
            alignment_changed
        } else {
            false
        };

        let aligned_mic = self
            .mic_delay_line
            .process(raw_mic, self.last_alignment.mic_delay_samples());
        let aligned_speaker = self
            .speaker_delay_line
            .process(raw_speaker, self.last_alignment.speaker_delay_samples());

        AecAlignedPair {
            mic: Arc::from(aligned_mic),
            speaker: Arc::from(aligned_speaker),
            alignment_changed,
        }
    }

    fn update_alignment(&mut self, event: &SyncProbeEvent) -> bool {
        let snapshot = event.snapshot();
        let next_alignment = if matches!(
            snapshot.state,
            SyncProbeState::Locked | SyncProbeState::Holdover
        ) {
            AecAlignment::from_lag_samples(snapshot.stable_lag_samples)
        } else {
            AecAlignment::None
        };

        if next_alignment != self.last_alignment {
            tracing::info!(
                previous_alignment = ?self.last_alignment,
                alignment = ?next_alignment,
                mic_delay_samples = next_alignment.mic_delay_samples(),
                speaker_delay_samples = next_alignment.speaker_delay_samples(),
                mic_delay_ms = next_alignment.mic_delay_samples() as f32
                    / self.probe.config().sample_rate as f32
                    * 1000.0,
                speaker_delay_ms = next_alignment.speaker_delay_samples() as f32
                    / self.probe.config().sample_rate as f32
                    * 1000.0,
                stable_lag_samples = snapshot.stable_lag_samples,
                state = ?snapshot.state,
                "aec_reference_alignment_changed"
            );
            self.last_alignment = next_alignment;
            return true;
        }

        false
    }

    fn log_probe_event(&mut self, event: SyncProbeEvent) {
        let snapshot = event.snapshot();
        let should_log = self.last_logged_state != Some(snapshot.state)
            || self.last_logged_stable_lag_samples != snapshot.stable_lag_samples;

        if !should_log {
            return;
        }

        match event {
            SyncProbeEvent::Measured(measurement) => {
                tracing::info!(
                    capture_time_sec = measurement.capture_time_sec,
                    state = ?measurement.snapshot.state,
                    stable_lag_samples = measurement.snapshot.stable_lag_samples,
                    candidate_lag_samples = measurement.snapshot.candidate_lag_samples,
                    accepted_window_count = measurement.snapshot.accepted_window_count,
                    confidence = measurement.snapshot.confidence,
                    peak_ratio = measurement.estimate.peak_ratio,
                    distinctiveness = measurement.estimate.distinctiveness,
                    drift_ppm = measurement.trend.drift_ppm,
                    "audio_sync_probe"
                );
            }
            SyncProbeEvent::SkippedLowConfidence(skip) => {
                tracing::info!(
                    capture_time_sec = skip.capture_time_sec,
                    state = ?skip.snapshot.state,
                    stable_lag_samples = skip.snapshot.stable_lag_samples,
                    candidate_lag_samples = skip.snapshot.candidate_lag_samples,
                    accepted_window_count = skip.snapshot.accepted_window_count,
                    confidence = skip.snapshot.confidence,
                    reason = ?skip.reason,
                    peak_ratio = skip.estimate.peak_ratio,
                    distinctiveness = skip.estimate.distinctiveness,
                    "audio_sync_probe"
                );
            }
            SyncProbeEvent::SkippedLowEnergy(skip) => {
                tracing::info!(
                    capture_time_sec = skip.capture_time_sec,
                    state = ?skip.snapshot.state,
                    stable_lag_samples = skip.snapshot.stable_lag_samples,
                    accepted_window_count = skip.snapshot.accepted_window_count,
                    reference_rms = skip.reference_rms,
                    observed_rms = skip.observed_rms,
                    "audio_sync_probe"
                );
            }
        }

        self.last_logged_state = Some(snapshot.state);
        self.last_logged_stable_lag_samples = snapshot.stable_lag_samples;
    }
}

struct SampleDelayLine {
    history: VecDeque<f32>,
    max_delay_samples: usize,
}

impl SampleDelayLine {
    fn new(max_delay_samples: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_delay_samples + 1),
            max_delay_samples,
        }
    }

    fn process(&mut self, input: &[f32], delay_samples: usize) -> Vec<f32> {
        let delay_samples = delay_samples.min(self.max_delay_samples);
        let mut output = Vec::with_capacity(input.len());

        for &sample in input {
            self.history.push_back(sample);
            let delayed = self
                .history
                .len()
                .checked_sub(delay_samples + 1)
                .and_then(|idx| self.history.get(idx))
                .copied()
                .unwrap_or(0.0);
            output.push(delayed);

            while self.history.len() > self.max_delay_samples + 1 {
                self.history.pop_front();
            }
        }

        output
    }
}

async fn run_single_loop(
    tx: tokio::sync::mpsc::Sender<Result<CaptureEvent, Error>>,
    cancel_token: CancellationToken,
    mut chunk_stream: ChunkStream,
    side: CaptureSide,
) {
    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => return,
            item = chunk_stream.next() => {
                match item {
                    Some(Ok(data)) => {
                        let data = Arc::<[f32]>::from(data);
                        let silence = Arc::<[f32]>::from(vec![0.0f32; data.len()]);
                        let frame = match side {
                            CaptureSide::Mic => CaptureFrame {
                                raw_mic: data,
                                raw_speaker: silence,
                                aec_mic: None,
                            },
                            CaptureSide::Speaker => CaptureFrame {
                                raw_mic: silence,
                                raw_speaker: data,
                                aec_mic: None,
                            },
                        };
                        if tx.send(Ok(CaptureEvent::Frame(frame))).await.is_err() {
                            return;
                        }
                    }
                    Some(Err(_)) => {
                        let err = match side {
                            CaptureSide::Mic => Error::MicResampleFailed,
                            CaptureSide::Speaker => Error::SpeakerResampleFailed,
                        };
                        let _ = tx.send(Err(err)).await;
                        return;
                    }
                    None => {
                        let err = match side {
                            CaptureSide::Mic => Error::MicStreamEnded,
                            CaptureSide::Speaker => Error::SpeakerStreamEnded,
                        };
                        let _ = tx.send(Err(err)).await;
                        return;
                    }
                }
            }
        }
    }
}

fn build_aec() -> Option<AEC> {
    AEC::new()
        .map_err(|error| tracing::warn!(error.message = ?error, "aec_init_failed"))
        .ok()
}

fn process_aec(
    aec: &mut Option<AEC>,
    linear_echo_gain: &mut Option<f32>,
    mic: &[f32],
    speaker: &[f32],
) -> Option<Arc<[f32]>> {
    let processor = aec.as_mut()?;
    match processor.process_streaming(mic, speaker) {
        Ok(processed) => Some(Arc::<[f32]>::from(cancel_linear_echo(
            speaker,
            processed,
            linear_echo_gain,
        ))),
        Err(error) => {
            tracing::warn!(error.message = ?error, "aec_failed");
            None
        }
    }
}

fn cancel_linear_echo(
    speaker: &[f32],
    processed: Vec<f32>,
    linear_echo_gain: &mut Option<f32>,
) -> Vec<f32> {
    let len = speaker.len().min(processed.len());
    if len == 0 {
        return processed;
    }

    let mut processed_energy = 0.0;
    let mut speaker_energy = 0.0;
    let mut cross_energy = 0.0;
    for idx in 0..len {
        let processed_sample = processed[idx];
        let speaker_sample = speaker[idx];
        processed_energy += processed_sample * processed_sample;
        speaker_energy += speaker_sample * speaker_sample;
        cross_energy += processed_sample * speaker_sample;
    }

    let len_f32 = len as f32;
    let processed_rms = (processed_energy / len_f32).sqrt();
    let speaker_rms = (speaker_energy / len_f32).sqrt();
    if processed_rms < AEC_MIN_MIC_RMS || speaker_rms < AEC_MIN_REFERENCE_RMS {
        return processed;
    }

    let correlation = cross_energy.abs() / (processed_energy * speaker_energy).sqrt().max(1e-6);
    if correlation < AEC_MIN_REFERENCE_CORRELATION {
        return processed;
    }

    let instantaneous_gain =
        (cross_energy / speaker_energy.max(1e-6)).clamp(-AEC_MAX_LINEAR_GAIN, AEC_MAX_LINEAR_GAIN);
    let residual_energy =
        (processed_energy - (cross_energy * cross_energy / speaker_energy.max(1e-6))).max(0.0);
    let residual_ratio = (residual_energy / len_f32).sqrt() / processed_rms.max(1e-6);
    let measured_gain = if residual_ratio > AEC_DOUBLE_TALK_RESIDUAL_RATIO {
        segmented_trimmed_gain(&processed, speaker, len, instantaneous_gain)
    } else {
        instantaneous_gain
    };
    let smoothed_gain = linear_echo_gain
        .map(|gain| gain + (measured_gain - gain) * AEC_LINEAR_GAIN_SMOOTHING)
        .unwrap_or(measured_gain);
    *linear_echo_gain = Some(smoothed_gain);

    let gain = if residual_ratio > AEC_DOUBLE_TALK_RESIDUAL_RATIO {
        smoothed_gain
    } else {
        measured_gain
    };

    let mut output = Vec::with_capacity(processed.len());
    output.extend(processed.iter().zip(speaker).take(len).map(
        |(processed_sample, speaker_sample)| {
            (processed_sample - gain * speaker_sample).clamp(-1.0, 1.0)
        },
    ));
    output.extend_from_slice(&processed[len..]);
    output
}

fn segmented_trimmed_gain(
    processed: &[f32],
    speaker: &[f32],
    len: usize,
    initial_gain: f32,
) -> f32 {
    let segment_len = (len / AEC_ROBUST_GAIN_SEGMENTS).max(1);
    let mut gains = Vec::with_capacity(AEC_ROBUST_GAIN_SEGMENTS);
    let mut start = 0;
    while start < len {
        let end = (start + segment_len).min(len);
        let mut processed_energy = 0.0;
        let mut speaker_energy = 0.0;
        let mut cross_energy = 0.0;
        for idx in start..end {
            let processed_sample = processed[idx];
            let speaker_sample = speaker[idx];
            processed_energy += processed_sample * processed_sample;
            speaker_energy += speaker_sample * speaker_sample;
            cross_energy += processed_sample * speaker_sample;
        }

        if processed_energy > 1e-8 && speaker_energy > 1e-8 {
            let correlation =
                cross_energy.abs() / (processed_energy * speaker_energy).sqrt().max(1e-6);
            if correlation >= AEC_MIN_REFERENCE_CORRELATION {
                let gain = (cross_energy / speaker_energy.max(1e-6))
                    .clamp(-AEC_MAX_LINEAR_GAIN, AEC_MAX_LINEAR_GAIN);
                gains.push((gain, speaker_energy));
            }
        }

        start = end;
    }

    if gains.len() < AEC_ROBUST_GAIN_MIN_SEGMENTS {
        return initial_gain;
    }

    gains.sort_by(|left, right| left.0.total_cmp(&right.0));
    let trim = if gains.len() >= 6 { gains.len() / 6 } else { 0 };
    let kept = &gains[trim..gains.len() - trim];
    let weighted_gain = kept.iter().map(|(gain, energy)| gain * energy).sum::<f32>();
    let kept_energy = kept.iter().map(|(_, energy)| energy).sum::<f32>();
    if kept_energy <= 1e-8 {
        return initial_gain;
    }

    (weighted_gain / kept_energy).clamp(-AEC_MAX_LINEAR_GAIN, AEC_MAX_LINEAR_GAIN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_retry_without_a_device_keeps_the_last_requested_one() {
        let requested = Mutex::new(Some("Wired Receiver".to_string()));
        assert_eq!(
            split_retry_device(&requested, None).as_deref(),
            Some("Wired Receiver")
        );
        assert_eq!(
            split_retry_device(&requested, Some("USB Mic".to_string())).as_deref(),
            Some("USB Mic")
        );
        assert_eq!(
            split_retry_device(&requested, None).as_deref(),
            Some("USB Mic")
        );

        let started_on_default = Mutex::new(None);
        assert_eq!(split_retry_device(&started_on_default, None), None);
    }

    #[test]
    fn split_mic_stays_on_its_device_and_reopens_it_when_it_returns() {
        let pinned = Mutex::new(None);
        let present = std::cell::RefCell::new(vec!["Receiver", "MacBook Mic"]);
        // Like `MicInput::new`: the requested device when present, otherwise the first one listed.
        let open = |target: Option<String>| -> Result<(String, ()), Error> {
            let present = present.borrow();
            let name = target
                .filter(|target| present.contains(&target.as_str()))
                .unwrap_or_else(|| present[0].to_string());
            Ok((name, ()))
        };
        let opened = |device: Option<&str>| {
            open_pinned(device.map(str::to_string), &pinned, open, |_| Ok(())).map(|(_, name)| name)
        };

        assert_eq!(opened(None).unwrap(), "Receiver");

        present.borrow_mut().remove(0);
        assert!(opened(None).is_err());
        assert!(opened(None).is_err());

        present.borrow_mut().insert(0, "Receiver");
        assert_eq!(opened(None).unwrap(), "Receiver");

        assert_eq!(opened(Some("MacBook Mic")).unwrap(), "MacBook Mic");
        assert_eq!(opened(None).unwrap(), "MacBook Mic");
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn split_mic_failure_recovers_both_tracks_together_and_aligned() {
        use super::super::recovery::OpenChannel;
        use futures_util::stream;
        use std::sync::atomic::AtomicUsize;

        let left = |i: usize| vec![0.1 * i as f32; 10];
        let right = |i: usize| vec![-0.1 * i as f32; 10];
        let attempts = Arc::new(AtomicUsize::new(0));
        let counter = attempts.clone();
        let open: OpenChannel<AudioPair> = Arc::new(move |_, _| {
            let (l, r) = if counter.fetch_add(1, Ordering::Relaxed) == 0 {
                // The left side runs ahead past the joiner's lag, then the device dies.
                (
                    Box::pin(stream::iter((1..=8).map(move |i| Ok(left(i))))) as ChunkStream,
                    Box::pin(stream::iter([Ok(right(1)), Ok(right(2))]).chain(stream::pending()))
                        as ChunkStream,
                )
            } else {
                (
                    Box::pin(stream::iter([Ok(left(9))]).chain(stream::pending())) as ChunkStream,
                    Box::pin(stream::iter([Ok(right(9))]).chain(stream::pending())) as ChunkStream,
                )
            };
            Ok(OpenedChannel {
                stream: Box::pin(LockstepPairs::new(l, r)),
                device: Some("stereo mic".into()),
            })
        });
        let (_retry, retry_rx) = tokio::sync::mpsc::unbounded_channel();
        let (pairs_tx, pairs_rx) = tokio::sync::mpsc::channel(32);
        let (tx, mut rx) = tokio::sync::mpsc::channel(32);
        let cancel = CancellationToken::new();
        let device = tokio::spawn(run_channel(
            CaptureChannel::Mic,
            open,
            None,
            retry_rx,
            pairs_tx,
            cancel.clone(),
        ));
        let output = tokio::spawn(run_split_loop(tx, cancel.clone(), 100, 10, pairs_rx));

        let mut frames = Vec::new();
        let mut gap_samples = 0;
        let mut ready = 0;
        let mut failed = 0;
        while frames.len() < 3 {
            match tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
                .await
                .unwrap()
                .unwrap()
                .unwrap()
            {
                CaptureEvent::ChannelReady { channel, .. } => {
                    assert_eq!(channel, CaptureChannel::Mic);
                    ready += 1;
                }
                CaptureEvent::ChannelFailed { channel, .. } => {
                    assert_eq!(channel, CaptureChannel::Mic);
                    failed += 1;
                }
                CaptureEvent::Gap(frame) => {
                    assert_eq!(frame.raw_mic.len(), frame.raw_speaker.len());
                    assert!(
                        frame
                            .raw_mic
                            .iter()
                            .chain(frame.raw_speaker.iter())
                            .all(|sample| *sample == 0.0)
                    );
                    gap_samples += frame.raw_mic.len();
                }
                CaptureEvent::Frame(frame) => {
                    assert!(frame.aec_mic.is_none());
                    frames.push((frame.raw_mic.to_vec(), frame.raw_speaker.to_vec()));
                }
            }
        }
        cancel.cancel();
        device.await.unwrap();
        output.await.unwrap();

        // Left chunks that never got a right partner are dropped with the failed device, never
        // paired with silence, and both tracks resume together on the reopened device.
        assert_eq!(
            frames,
            vec![
                (left(1), right(1)),
                (left(2), right(2)),
                (left(9), right(9))
            ]
        );
        assert_eq!((ready, failed), (2, 1));
        assert!(
            gap_samples >= 50,
            "the outage must be padded on both tracks, got {gap_samples} samples"
        );
        assert_eq!(attempts.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn channel_failure_preserves_healthy_audio_and_recovery_restores_stereo_frames() {
        let (chunks_tx, chunks_rx) = tokio::sync::mpsc::channel(32);
        let (tx, mut rx) = tokio::sync::mpsc::channel(32);
        let task = tokio::spawn(run_dual_loop(
            tx,
            CancellationToken::new(),
            16_000,
            32,
            Arc::new(AtomicBool::new(false)),
            chunks_rx,
        ));
        let speaker = vec![0.25; 32];
        let mic = vec![0.5; 32];
        for item in [
            ChannelItem::Failed(CaptureChannel::Mic, Error::MicOpenFailed),
            ChannelItem::Ready(CaptureChannel::Speaker, None),
            ChannelItem::Chunk(CaptureChannel::Speaker, speaker.clone()),
            ChannelItem::Ready(CaptureChannel::Mic, Some("fallback".into())),
            ChannelItem::Chunk(CaptureChannel::Mic, mic.clone()),
            ChannelItem::Chunk(CaptureChannel::Speaker, speaker.clone()),
            ChannelItem::Failed(CaptureChannel::Speaker, Error::SpeakerStreamSetupFailed),
            ChannelItem::Chunk(CaptureChannel::Mic, mic.clone()),
        ] {
            chunks_tx.send(item).await.unwrap();
        }
        drop(chunks_tx);
        let mut frames = Vec::new();
        while let Some(event) = rx.recv().await {
            if let CaptureEvent::Frame(frame) = event.unwrap() {
                frames.push(frame);
            }
        }
        task.await.unwrap();
        assert_eq!(frames.len(), 3);
        assert_eq!(&*frames[0].raw_speaker, &speaker);
        assert!(frames[0].raw_mic.iter().all(|sample| *sample == 0.0));
        assert_eq!(&*frames[1].raw_mic, &mic);
        assert_eq!(&*frames[1].raw_speaker, &speaker);
        assert_eq!(&*frames[2].raw_mic, &mic);
        assert!(frames[2].raw_speaker.iter().all(|sample| *sample == 0.0));
    }

    #[tokio::test(start_paused = true)]
    async fn dual_outage_keeps_sixty_seconds_of_silence_before_recovered_audio() {
        let (chunks_tx, chunks_rx) = tokio::sync::mpsc::channel(32);
        let (tx, mut rx) = tokio::sync::mpsc::channel(32);
        let cancel = CancellationToken::new();
        let task = tokio::spawn(run_dual_loop(
            tx,
            cancel.clone(),
            100,
            10,
            Arc::new(AtomicBool::new(false)),
            chunks_rx,
        ));
        chunks_tx
            .send(ChannelItem::Ready(CaptureChannel::Mic, Some("mic".into())))
            .await
            .unwrap();
        chunks_tx
            .send(ChannelItem::Chunk(CaptureChannel::Mic, vec![0.5; 10]))
            .await
            .unwrap();
        while !matches!(
            tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
                .await
                .unwrap()
                .unwrap()
                .unwrap(),
            CaptureEvent::Frame(_)
        ) {}
        chunks_tx
            .send(ChannelItem::Failed(
                CaptureChannel::Mic,
                Error::MicOpenFailed,
            ))
            .await
            .unwrap();
        chunks_tx
            .send(ChannelItem::Failed(
                CaptureChannel::Speaker,
                Error::SpeakerStreamSetupFailed,
            ))
            .await
            .unwrap();
        for _ in 0..2 {
            assert!(matches!(
                tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap(),
                CaptureEvent::ChannelFailed { .. }
            ));
        }
        // Advance once; catch-up padding stays bounded by the output queue.
        tokio::time::advance(std::time::Duration::from_secs(60)).await;
        let mut gap_samples = 0;
        while gap_samples < 6000 {
            match tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
                .await
                .unwrap()
                .unwrap()
                .unwrap()
            {
                CaptureEvent::Gap(frame) => {
                    assert!(
                        frame
                            .raw_dual()
                            .0
                            .iter()
                            .chain(frame.raw_dual().1.iter())
                            .all(|sample| *sample == 0.0)
                    );
                    gap_samples += frame.raw_mic.len();
                }
                _ => panic!("silence cannot claim capture readiness"),
            }
        }
        chunks_tx
            .send(ChannelItem::Ready(CaptureChannel::Speaker, None))
            .await
            .unwrap();
        chunks_tx
            .send(ChannelItem::Chunk(CaptureChannel::Speaker, vec![0.25; 10]))
            .await
            .unwrap();
        loop {
            match tokio::time::timeout(std::time::Duration::from_secs(1), rx.recv())
                .await
                .unwrap()
                .unwrap()
                .unwrap()
            {
                CaptureEvent::Gap(frame) => gap_samples += frame.raw_mic.len(),
                CaptureEvent::Frame(frame) => {
                    assert_eq!(&*frame.raw_speaker, &[0.25; 10]);
                    assert!(
                        gap_samples >= 6000,
                        "recorder and transcription must retain the full outage offset"
                    );
                    break;
                }
                _ => {}
            }
        }
        cancel.cancel();
        task.await.unwrap();
    }

    #[test]
    fn process_aec_returns_output_when_enabled() {
        let mut aec = build_aec();
        let mut linear_echo_gain = None;
        let mic = Arc::<[f32]>::from(vec![0.1_f32; 160]);
        let speaker = Arc::<[f32]>::from(vec![0.2_f32; 160]);

        let processed = process_aec(&mut aec, &mut linear_echo_gain, &mic, &speaker);
        assert_eq!(processed.as_ref().map(|data| data.len()), Some(160));
    }

    #[test]
    fn sample_delay_line_delays_and_clamps() {
        let mut delay = SampleDelayLine::new(4);
        let output = delay.process(&[1.0, 2.0, 3.0], 0);
        assert_eq!(output, vec![1.0, 2.0, 3.0]);
        let mut delay = SampleDelayLine::new(4);
        let first = delay.process(&[1.0, 2.0, 3.0], 2);
        let second = delay.process(&[4.0, 5.0], 2);
        assert_eq!(first, vec![0.0, 0.0, 1.0]);
        assert_eq!(second, vec![2.0, 3.0]);
        let mut delay = SampleDelayLine::new(2);
        let first = delay.process(&[1.0, 2.0, 3.0], 10);
        let second = delay.process(&[4.0], 10);
        assert_eq!(first, vec![0.0, 0.0, 1.0]);
        assert_eq!(second, vec![2.0]);
    }

    #[test]
    fn aec_alignment_delays_the_leading_signal() {
        let speaker_alignment = AecAlignment::from_lag_samples(Some(320));
        assert_eq!(speaker_alignment, AecAlignment::DelaySpeaker(320));
        assert_eq!(speaker_alignment.mic_delay_samples(), 0);
        assert_eq!(speaker_alignment.speaker_delay_samples(), 320);

        let mic_alignment = AecAlignment::from_lag_samples(Some(-320));
        assert_eq!(mic_alignment, AecAlignment::DelayMic(320));
        assert_eq!(mic_alignment.mic_delay_samples(), 320);
        assert_eq!(mic_alignment.speaker_delay_samples(), 0);
    }

    fn test_signal(len: usize) -> Vec<f32> {
        let mut state = 0x1234_5678u32;
        (0..len)
            .map(|idx| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let noise = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
                let pulse = if idx % 257 == 0 { 0.75 } else { 0.0 };
                0.6 * noise + pulse
            })
            .collect()
    }

    fn delayed(input: &[f32], delay_samples: usize) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        output[delay_samples..].copy_from_slice(&input[..input.len() - delay_samples]);
        output
    }

    fn detected_alignment(speaker: &[f32], mic: &[f32]) -> AecAlignment {
        let mut aligner = AecReferenceAligner::new(16_000);
        for (speaker, mic) in speaker.chunks(1_600).zip(mic.chunks(1_600)) {
            aligner.align(speaker, mic);
        }
        aligner.last_alignment
    }

    #[test]
    fn aec_reference_aligner_locks_negative_macbook_scale_lag() {
        let mic = test_signal(16_000 * 5);
        let speaker = delayed(&mic, 7_200);

        let alignment = detected_alignment(&speaker, &mic);

        assert_eq!(alignment, AecAlignment::DelayMic(7_200));
    }

    #[test]
    fn aec_reference_aligner_locks_at_supported_lag_boundary() {
        let speaker = test_signal(16_000 * 5);
        let mic = delayed(&speaker, 9_600);

        let alignment = detected_alignment(&speaker, &mic);

        assert_eq!(alignment, AecAlignment::DelaySpeaker(9_600));
    }

    #[test]
    fn aec_reference_aligner_emits_detected_speaker_delay() {
        let speaker = test_signal(16_000 * 5);
        let mic = delayed(&speaker, 7_200);
        let mut aligner = AecReferenceAligner::new(16_000);
        let mut aligned = None;

        for (speaker, mic) in speaker.chunks(1_600).zip(mic.chunks(1_600)) {
            aligned = Some(aligner.align(speaker, mic));
        }

        let aligned = aligned.unwrap();
        assert_eq!(aligner.last_alignment, AecAlignment::DelaySpeaker(7_200));
        assert_eq!(&*aligned.speaker, &speaker[71_200..72_800]);
    }
}
