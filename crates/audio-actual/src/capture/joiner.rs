use std::collections::VecDeque;

pub(crate) type AudioPair = (Vec<f32>, Vec<f32>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JoinMode {
    /// Independent devices: a side that falls behind is filled with silence.
    Independent,
    /// Two channels of one device: chunks are only ever paired or dropped together.
    Lockstep,
}

pub(crate) struct Joiner {
    mode: JoinMode,
    mic: VecDeque<Vec<f32>>,
    speaker: VecDeque<Vec<f32>>,
    // Lockstep only: chunks still to arrive whose partner was dropped on overflow.
    mic_drops_pending: usize,
    speaker_drops_pending: usize,
}

impl Joiner {
    pub(crate) const MAX_LAG: usize = 4;
    const MAX_QUEUE_SIZE: usize = 30;

    pub(crate) fn new() -> Self {
        Self::with_mode(JoinMode::Independent)
    }

    pub(crate) fn lockstep() -> Self {
        Self::with_mode(JoinMode::Lockstep)
    }

    fn with_mode(mode: JoinMode) -> Self {
        Self {
            mode,
            mic: VecDeque::new(),
            speaker: VecDeque::new(),
            mic_drops_pending: 0,
            speaker_drops_pending: 0,
        }
    }

    pub(crate) fn push_mic(&mut self, data: Vec<f32>) {
        if self.mode == JoinMode::Lockstep && self.mic_drops_pending > 0 {
            self.mic_drops_pending -= 1;
            return;
        }
        self.mic.push_back(data);
        if self.mic.len() > Self::MAX_QUEUE_SIZE {
            tracing::warn!("mic_queue_overflow");
            self.mic.pop_front();
            if self.mode == JoinMode::Lockstep && self.speaker.pop_front().is_none() {
                self.speaker_drops_pending += 1;
            }
        }
    }

    pub(crate) fn push_speaker(&mut self, data: Vec<f32>) {
        if self.mode == JoinMode::Lockstep && self.speaker_drops_pending > 0 {
            self.speaker_drops_pending -= 1;
            return;
        }
        self.speaker.push_back(data);
        if self.speaker.len() > Self::MAX_QUEUE_SIZE {
            tracing::warn!("speaker_queue_overflow");
            self.speaker.pop_front();
            if self.mode == JoinMode::Lockstep && self.mic.pop_front().is_none() {
                self.mic_drops_pending += 1;
            }
        }
    }

    pub(crate) fn clear_mic(&mut self) {
        self.mic.clear();
    }
    pub(crate) fn clear_speaker(&mut self) {
        self.speaker.clear();
    }

    pub(crate) fn pop_available_pair(
        &mut self,
        mic_ready: bool,
        speaker_ready: bool,
    ) -> Option<AudioPair> {
        if !mic_ready {
            self.clear_mic();
        }
        if !speaker_ready {
            self.clear_speaker();
        }
        if !mic_ready {
            let speaker = self.speaker.pop_front()?;
            return Some((vec![0.0; speaker.len()], speaker));
        }
        if !speaker_ready {
            let mic = self.mic.pop_front()?;
            let silence = vec![0.0; mic.len()];
            return Some((mic, silence));
        }
        self.pop_pair()
    }

    pub(crate) fn pop_pair(&mut self) -> Option<AudioPair> {
        if self.mic.front().is_some() && self.speaker.front().is_some() {
            return Some((self.mic.pop_front()?, self.speaker.pop_front()?));
        }

        if self.mode == JoinMode::Lockstep {
            return None;
        }

        if self.mic.front().is_some() && self.speaker.is_empty() && self.mic.len() > Self::MAX_LAG {
            let mic = self.mic.pop_front()?;
            let silence = vec![0.0; mic.len()];
            return Some((mic, silence));
        }
        if self.speaker.front().is_some()
            && self.mic.is_empty()
            && self.speaker.len() > Self::MAX_LAG
        {
            let speaker = self.speaker.pop_front()?;
            let silence = vec![0.0; speaker.len()];
            return Some((silence, speaker));
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joiner_emits_silence_filled_pair_after_max_lag() {
        let mut joiner = Joiner::new();
        for _ in 0..=Joiner::MAX_LAG {
            joiner.push_mic(vec![0.25, -0.25]);
        }

        let (mic, speaker) = joiner.pop_pair().unwrap();
        assert_eq!(mic, vec![0.25, -0.25]);
        assert_eq!(speaker, vec![0.0, 0.0]);
    }

    fn chunk(tag: f32) -> Vec<f32> {
        vec![tag, tag]
    }

    #[test]
    fn lockstep_joiner_waits_for_a_delayed_side_without_silence() {
        let mut joiner = Joiner::lockstep();
        for i in 1..=6 {
            joiner.push_mic(chunk(i as f32));
            assert!(joiner.pop_pair().is_none());
        }

        for i in 1..=6 {
            joiner.push_speaker(chunk(-(i as f32)));
            let (left, right) = joiner.pop_pair().unwrap();
            assert_eq!(left, chunk(i as f32));
            assert_eq!(right, chunk(-(i as f32)));
            assert!(joiner.pop_pair().is_none());
        }
    }

    #[test]
    fn lockstep_joiner_overflow_drops_both_sides_together() {
        let mut joiner = Joiner::lockstep();
        let total = Joiner::MAX_QUEUE_SIZE + 2;
        for i in 1..=total {
            joiner.push_mic(chunk(i as f32));
        }
        for i in 1..=total {
            joiner.push_speaker(chunk(-(i as f32)));
        }

        let mut pairs = Vec::new();
        while let Some(pair) = joiner.pop_pair() {
            pairs.push(pair);
        }
        assert_eq!(pairs.len(), Joiner::MAX_QUEUE_SIZE);
        assert_eq!(pairs[0].0, chunk(3.0));
        for (left, right) in pairs {
            assert_eq!(left[0], -right[0]);
        }
    }
}
