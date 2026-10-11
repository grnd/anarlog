use dasp::sample::ToSample;
use ringbuf::traits::Producer;

pub(crate) const DEFAULT_SCRATCH_LEN: usize = 8192;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PushStats {
    pub(crate) pushed: usize,
    pub(crate) dropped: usize,
}

pub(crate) fn push_interleaved_downmix_to_mono_ringbuf<S, P>(
    data: &[S],
    channels: usize,
    scratch: &mut [f32],
    producer: &mut P,
) -> PushStats
where
    S: Copy + ToSample<f32>,
    P: Producer<Item = f32>,
{
    if scratch.is_empty() {
        return PushStats::default();
    }

    if channels == 0 {
        return PushStats::default();
    }

    let frame_count = data.len() / channels;
    if frame_count == 0 {
        return PushStats::default();
    }

    let mut offset = 0usize;
    let mut pushed_total = 0usize;
    let mut dropped_total = 0usize;

    if channels == 1 {
        while offset < frame_count {
            let count = (frame_count - offset).min(scratch.len());

            let vacant = producer.vacant_len();
            if vacant == 0 {
                dropped_total += frame_count - offset;
                break;
            }

            let convert_count = count.min(vacant);

            for i in 0..convert_count {
                scratch[i] = data[offset + i].to_sample_();
            }

            let pushed = producer.push_slice(&scratch[..convert_count]);
            pushed_total += pushed;
            dropped_total += count - pushed;

            offset += count;
        }

        return PushStats {
            pushed: pushed_total,
            dropped: dropped_total,
        };
    }

    while offset < frame_count {
        let count = (frame_count - offset).min(scratch.len());

        let vacant = producer.vacant_len();
        if vacant == 0 {
            dropped_total += frame_count - offset;
            break;
        }

        let convert_count = count.min(vacant);

        for (i, sample) in scratch.iter_mut().enumerate().take(convert_count) {
            let frame_offset = (offset + i) * channels;
            let mut sum = 0.0f32;
            for channel in 0..channels {
                sum += data[frame_offset + channel].to_sample_();
            }
            *sample = sum / channels as f32;
        }

        let pushed = producer.push_slice(&scratch[..convert_count]);
        pushed_total += pushed;
        dropped_total += count - pushed;

        offset += count;
    }

    PushStats {
        pushed: pushed_total,
        dropped: dropped_total,
    }
}

/// Pushes the first two channels of interleaved frames into separate ring buffers so a stereo
/// input can carry two speakers. Channels past the second are ignored; frames are dropped on
/// both sides together when either buffer is full, so the two streams stay frame-aligned.
pub(crate) fn push_interleaved_first_two_channels_ringbuf<S, P>(
    data: &[S],
    channels: usize,
    scratch: &mut [f32],
    left: &mut P,
    right: &mut P,
) -> PushStats
where
    S: Copy + ToSample<f32>,
    P: Producer<Item = f32>,
{
    if scratch.is_empty() || channels < 2 {
        return PushStats::default();
    }

    let frame_count = data.len() / channels;
    let mut stats = PushStats::default();

    let mut offset = 0usize;
    while offset < frame_count {
        let count = (frame_count - offset).min(scratch.len());
        let writable = count.min(left.vacant_len()).min(right.vacant_len());
        if writable == 0 {
            stats.dropped += 2 * (frame_count - offset);
            break;
        }

        for (channel, producer) in [(0usize, &mut *left), (1usize, &mut *right)] {
            for (i, slot) in scratch[..writable].iter_mut().enumerate() {
                *slot = data[(offset + i) * channels + channel].to_sample_();
            }
            producer.push_slice(&scratch[..writable]);
        }

        stats.pushed += 2 * writable;
        stats.dropped += 2 * (count - writable);
        offset += count;
    }

    stats
}

pub(crate) fn convert_and_push_to_ringbuf<T, P>(
    samples: &[T],
    scratch: &mut [f32],
    producer: &mut P,
    mut convert: impl FnMut(T) -> f32,
) -> PushStats
where
    T: Copy,
    P: Producer<Item = f32>,
{
    if scratch.is_empty() {
        return PushStats::default();
    }

    if samples.is_empty() {
        return PushStats::default();
    }

    let mut offset = 0usize;
    let mut pushed_total = 0usize;
    let mut dropped_total = 0usize;

    while offset < samples.len() {
        let count = (samples.len() - offset).min(scratch.len());

        let vacant = producer.vacant_len();
        if vacant == 0 {
            dropped_total += samples.len() - offset;
            break;
        }

        let convert_count = count.min(vacant);

        for i in 0..convert_count {
            scratch[i] = convert(samples[offset + i]);
        }

        let pushed = producer.push_slice(&scratch[..convert_count]);
        pushed_total += pushed;
        dropped_total += count - pushed;

        offset += count;
    }

    PushStats {
        pushed: pushed_total,
        dropped: dropped_total,
    }
}

pub(crate) fn push_f32_to_ringbuf<P>(data: &[f32], producer: &mut P) -> PushStats
where
    P: Producer<Item = f32>,
{
    if data.is_empty() {
        return PushStats::default();
    }

    let pushed = producer.push_slice(data);
    PushStats {
        pushed,
        dropped: data.len() - pushed,
    }
}

#[cfg_attr(not(any(target_os = "linux", target_os = "windows")), allow(dead_code))]
pub(crate) fn push_f32le_bytes_first_channel_to_ringbuf<P>(
    data: &[u8],
    channels: usize,
    scratch: &mut [f32],
    producer: &mut P,
) -> PushStats
where
    P: Producer<Item = f32>,
{
    if scratch.is_empty() || channels == 0 {
        return PushStats::default();
    }

    let frame_size = channels.saturating_mul(std::mem::size_of::<f32>());
    if frame_size == 0 {
        return PushStats::default();
    }

    let frame_count = data.len() / frame_size;
    if frame_count == 0 {
        return PushStats::default();
    }

    let mut offset = 0usize;
    let mut pushed_total = 0usize;
    let mut dropped_total = 0usize;

    while offset < frame_count {
        let count = (frame_count - offset).min(scratch.len());

        let vacant = producer.vacant_len();
        if vacant == 0 {
            dropped_total += frame_count - offset;
            break;
        }

        let convert_count = count.min(vacant);

        for (i, slot) in scratch[..convert_count].iter_mut().enumerate() {
            let byte_offset = (offset + i) * frame_size;
            *slot = f32::from_le_bytes([
                data[byte_offset],
                data[byte_offset + 1],
                data[byte_offset + 2],
                data[byte_offset + 3],
            ]);
        }

        let pushed = producer.push_slice(&scratch[..convert_count]);
        pushed_total += pushed;
        dropped_total += count - pushed;

        offset += count;
    }

    PushStats {
        pushed: pushed_total,
        dropped: dropped_total,
    }
}

#[cfg(test)]
mod tests {
    use ringbuf::{
        HeapRb,
        traits::{Consumer, Observer, Split},
    };

    use super::*;

    #[test]
    fn downmix_to_mono_keeps_single_channel_samples() {
        let rb = HeapRb::<f32>::new(8);
        let (mut producer, mut consumer) = rb.split();
        let mut scratch = vec![0.0; 8];

        let stats = push_interleaved_downmix_to_mono_ringbuf(
            &[0.1_f32, 0.2, 0.3, 0.4],
            1,
            &mut scratch,
            &mut producer,
        );

        assert_eq!(stats.pushed, 4);
        let mut out = vec![0.0; consumer.occupied_len()];
        let read = consumer.pop_slice(&mut out);
        assert_eq!(read, 4);
        assert_eq!(out, vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn downmix_to_mono_averages_multichannel_frames() {
        let rb = HeapRb::<f32>::new(8);
        let (mut producer, mut consumer) = rb.split();
        let mut scratch = vec![0.0; 8];

        let stats = push_interleaved_downmix_to_mono_ringbuf(
            &[0.2_f32, 0.6, 0.1, 0.5],
            2,
            &mut scratch,
            &mut producer,
        );

        assert_eq!(stats.pushed, 2);
        let mut out = vec![0.0; consumer.occupied_len()];
        let read = consumer.pop_slice(&mut out);
        assert_eq!(read, 2);
        assert_eq!(out, vec![0.4, 0.3]);
    }

    #[test]
    fn split_first_two_channels_drops_frames_on_both_sides_together() {
        let (mut left_producer, mut left_consumer) = HeapRb::<f32>::new(8).split();
        let (mut right_producer, mut right_consumer) = HeapRb::<f32>::new(2).split();
        let mut scratch = vec![0.0; 8];

        // Three channels per frame; the third is ignored. Right buffer only fits two frames.
        let stats = push_interleaved_first_two_channels_ringbuf(
            &[0.1_f32, 0.9, 0.5, 0.2, 0.8, 0.5, 0.3, 0.7, 0.5],
            3,
            &mut scratch,
            &mut left_producer,
            &mut right_producer,
        );

        assert_eq!(stats.pushed, 4);
        assert_eq!(stats.dropped, 2);
        let mut left = vec![0.0; left_consumer.occupied_len()];
        left_consumer.pop_slice(&mut left);
        let mut right = vec![0.0; right_consumer.occupied_len()];
        right_consumer.pop_slice(&mut right);
        assert_eq!(left, vec![0.1, 0.2]);
        assert_eq!(right, vec![0.9, 0.8]);

        // Right drained: the next frame lands on both sides, so alignment continues.
        let stats = push_interleaved_first_two_channels_ringbuf(
            &[0.4_f32, 0.6, 0.5],
            3,
            &mut scratch,
            &mut left_producer,
            &mut right_producer,
        );
        assert_eq!(stats.pushed, 2);
        assert_eq!(stats.dropped, 0);
        assert_eq!(left_consumer.try_pop(), Some(0.4));
        assert_eq!(right_consumer.try_pop(), Some(0.6));
    }
}
