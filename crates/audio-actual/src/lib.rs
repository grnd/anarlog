mod async_ring;
mod capture;
mod mic;
pub mod native_meeting;
mod rt_ring;
mod speaker;

pub use mic::*;
pub use speaker::*;

pub use cpal;

pub use anlg_audio::{AudioProvider, CaptureConfig, CaptureFrame, CaptureStream, Error};
pub use anlg_audio_interface::AsyncSource;

pub const TAP_DEVICE_NAME: &str = "anarlog-audio-tap";

pub struct AudioOutput {}

impl AudioOutput {
    pub fn to_speaker(bytes: &'static [u8]) -> std::sync::mpsc::Sender<()> {
        use rodio::{Decoder, Player};
        let (tx, rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            if let Ok(stream) = anlg_audio_utils::open_default_playback_sink() {
                let file = std::io::Cursor::new(bytes);
                if let Ok(source) = Decoder::try_from(file) {
                    let player = Player::connect_new(stream.mixer());
                    player.append(source);

                    let _ = rx.recv_timeout(std::time::Duration::from_secs(3600));
                    player.stop();
                }
            }
        });

        tx
    }

    pub fn silence() -> std::sync::mpsc::Sender<()> {
        use rodio::{
            Player, nz,
            source::{Source, Zero},
        };

        let (tx, rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            if let Ok(stream) = anlg_audio_utils::open_default_playback_sink() {
                let silence = Zero::new(nz!(2u16), nz!(48_000u32))
                    .take_duration(std::time::Duration::from_secs(1))
                    .repeat_infinite();

                let player = Player::connect_new(stream.mixer());
                player.append(silence);

                let _ = rx.recv();
                player.stop();
            }
        });

        tx
    }
}

pub struct ActualAudio;

impl AudioProvider for ActualAudio {
    fn open_capture(&self, config: CaptureConfig) -> Result<CaptureStream, Error> {
        capture::open_capture(config)
    }

    fn open_speaker_capture(
        &self,
        sample_rate: u32,
        chunk_size: usize,
    ) -> Result<CaptureStream, Error> {
        capture::open_speaker_capture(sample_rate, chunk_size)
    }

    fn open_mic_capture(
        &self,
        device: Option<String>,
        sample_rate: u32,
        chunk_size: usize,
    ) -> Result<CaptureStream, Error> {
        capture::open_mic_capture(device, sample_rate, chunk_size)
    }

    fn open_split_mic_capture(
        &self,
        device: Option<String>,
        sample_rate: u32,
        chunk_size: usize,
    ) -> Result<CaptureStream, Error> {
        capture::open_split_mic_capture(device, sample_rate, chunk_size)
    }

    fn default_device_name(&self) -> String {
        MicInput::default_device_name()
    }

    fn list_mic_devices(&self) -> Vec<String> {
        MicInput::list_devices()
    }

    fn play_silence(&self) -> std::sync::mpsc::Sender<()> {
        AudioOutput::silence()
    }

    fn play_bytes(&self, bytes: &'static [u8]) -> std::sync::mpsc::Sender<()> {
        AudioOutput::to_speaker(bytes)
    }

    fn probe_mic(&self, device: Option<String>) -> Result<(), Error> {
        let input = MicInput::new(device)?;
        let _stream = input.stream()?;
        Ok(())
    }

    fn probe_mic_stereo(&self, device: Option<String>) -> Result<(), Error> {
        // A hung startup holds the host lock that opening a device needs; don't wait on it.
        if mic::startup_pending() {
            return Err(Error::MicStreamInitializationFailed(
                "previous microphone startup is still pending".into(),
            ));
        }
        let input = MicInput::probe(device.clone())?;
        // `MicInput::probe` falls back to another input when the requested one is absent.
        if !requested_device_opened(device.as_deref(), &input.device_name()) {
            return Err(Error::MicStreamInitializationFailed(
                "selected microphone not found".into(),
            ));
        }
        if input.channels() < 2 {
            return Err(Error::MicNotStereo);
        }
        Ok(())
    }

    fn probe_speaker(&self) -> Result<(), Error> {
        let speaker = SpeakerInput::new()
            .map_err(|error| Error::SpeakerStreamInitializationFailed(error.to_string()))?;
        let _stream = speaker
            .stream()
            .map_err(|error| Error::SpeakerStreamInitializationFailed(error.to_string()))?;
        Ok(())
    }

    /// Offered on macOS only; whether a capture may use it is the meeting runtime's decision.
    fn meeting_engine(
        &self,
    ) -> Option<std::sync::Arc<dyn anlg_audio::meeting::MeetingCaptureEngine>> {
        if cfg!(target_os = "macos") {
            Some(native_meeting::shared_engine())
        } else {
            None
        }
    }
}

/// True when no device was requested, or the opened input is the requested one.
pub(crate) fn requested_device_opened(requested: Option<&str>, opened: &str) -> bool {
    // An empty name means the default input, as in `MicInput::new`.
    requested.map(str::trim).is_none_or(|requested| {
        requested.is_empty() || requested.eq_ignore_ascii_case(opened.trim())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fallback_input_does_not_stand_in_for_the_requested_device() {
        assert!(requested_device_opened(None, "MacBook Mic"));
        assert!(requested_device_opened(Some("Receiver"), "Receiver"));
        assert!(!requested_device_opened(Some("Receiver"), "MacBook Mic"));
        assert!(requested_device_opened(Some(""), "MacBook Mic"));
        assert!(requested_device_opened(Some(" receiver "), "Receiver"));
    }
}
