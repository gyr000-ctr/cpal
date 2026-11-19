//! Integration tests for the core format-negotiation and sample-conversion layer.
//!
//! Run with: `cargo test -p cpal --features custom --test format_negotiation`
//!
//! The tests pin down four guarantees:
//! 1. exact format matches are zero-copy and bit-identical;
//! 2. rounding and clipping rules are assertable per sample;
//! 3. conversion buffers have a fixed, queryable upper bound and never reallocate;
//! 4. negotiation failures are distinguishable by error variant.

use cpal::{
    ChannelCount, ConvertBuffer, ConvertError, FormatRequest, NegotiationError, SampleConverter,
    SampleFormat, SampleRate, SupportedBufferSize, SupportedStreamConfigRange,
    format_negotiation::negotiate,
};

fn offer(
    channels: ChannelCount,
    min_rate: SampleRate,
    max_rate: SampleRate,
    format: SampleFormat,
) -> SupportedStreamConfigRange {
    SupportedStreamConfigRange::new(
        channels,
        min_rate,
        max_rate,
        SupportedBufferSize::Range { min: 64, max: 4096 },
        format,
    )
}

/// Convert a single sample value through byte buffers, returning the raw source and target bytes.
fn convert_one<S: Copy>(converter: &SampleConverter, value: S) -> (Vec<u8>, Vec<u8>) {
    let input = as_bytes(&value);
    let mut output = vec![0u8; converter.output_len_for(input.len()).unwrap()];
    let written = converter.convert(input, &mut output).unwrap();
    output.truncate(written);
    (input.to_vec(), output)
}

fn as_bytes<T: Copy>(value: &T) -> &[u8] {
    // Safe: plain integer/float primitives without padding bytes.
    unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
}

fn read_bytes<T: Copy>(bytes: &[u8]) -> T {
    assert_eq!(bytes.len(), size_of::<T>());
    unsafe { (bytes.as_ptr() as *const T).read_unaligned() }
}

// ---------------------------------------------------------------------------
// 1. Exact match: zero-copy, bit-identical
// ---------------------------------------------------------------------------

#[test]
fn identity_converter_copies_bytes_verbatim() {
    for format in [
        SampleFormat::I8,
        SampleFormat::I16,
        SampleFormat::I24,
        SampleFormat::I32,
        SampleFormat::I64,
        SampleFormat::U8,
        SampleFormat::U16,
        SampleFormat::U24,
        SampleFormat::U32,
        SampleFormat::U64,
        SampleFormat::F32,
        SampleFormat::F64,
    ] {
        let converter = SampleConverter::new(format, format).unwrap();
        assert!(converter.is_identity());

        // Deliberately "impossible" bit patterns for narrow formats stay untouched.
        let input: Vec<u8> = (0..(format.sample_size() * 7))
            .map(|i| (i as u8).wrapping_mul(37).wrapping_add(9))
            .collect();
        let mut output = vec![0u8; input.len()];
        let written = converter.convert(&input, &mut output).unwrap();

        assert_eq!(written, input.len());
        assert_eq!(output, input, "format {format:?} must be bit-identical");
    }
}

#[test]
fn identity_converter_reports_same_formats() {
    let converter = SampleConverter::new(SampleFormat::F32, SampleFormat::F32).unwrap();
    assert_eq!(converter.source_format(), SampleFormat::F32);
    assert_eq!(converter.target_format(), SampleFormat::F32);
    assert!(converter.is_identity());
}

// ---------------------------------------------------------------------------
// 2. Rounding and clipping rules, asserted per sample
// ---------------------------------------------------------------------------

#[test]
fn f32_to_i16_half_away_from_zero_and_clipping() {
    let c = SampleConverter::new(SampleFormat::F32, SampleFormat::I16).unwrap();
    let check = |x: f32, expected: i16| {
        let (_, out) = convert_one(&c, x);
        assert_eq!(read_bytes::<i16>(&out), expected, "f32 {x} -> i16");
    };

    // Rounding: half away from zero.
    check(0.0, 0);
    check(-0.0, 0);
    check(0.5 / 32_768.0, 1); // smallest value rounding up
    check(-0.5 / 32_768.0, -1); // and down
    check(1.5 / 32_768.0, 2);
    check(-1.5 / 32_768.0, -2);

    // Exact representable points.
    check(1.0, i16::MAX); // full scale is reachable, not an overflow
    check(-1.0, i16::MIN);

    // Clipping: out-of-range and infinite saturate instead of wrapping.
    check(2.0, i16::MAX);
    check(-2.0, i16::MIN);
    check(f32::INFINITY, i16::MAX);
    check(f32::NEG_INFINITY, i16::MIN);

    // NaN maps to equilibrium.
    check(f32::NAN, 0);
}

#[test]
fn i16_to_f32_uses_full_scale_division() {
    let c = SampleConverter::new(SampleFormat::I16, SampleFormat::F32).unwrap();
    let check = |x: i16, expected: f32| {
        let (_, out) = convert_one(&c, x);
        assert_eq!(read_bytes::<f32>(&out), expected, "i16 {x} -> f32");
    };

    check(0, 0.0);
    check(1, 1.0 / 32_768.0);
    check(-1, -1.0 / 32_768.0);
    check(i16::MAX, 32_767.0 / 32_768.0);
    check(i16::MIN, -1.0);
}

#[test]
fn integer_widening_keeps_full_scale_in_msbs() {
    let c = SampleConverter::new(SampleFormat::I16, SampleFormat::I32).unwrap();
    let check = |x: i16, expected: i32| {
        let (_, out) = convert_one(&c, x);
        assert_eq!(read_bytes::<i32>(&out), expected, "i16 {x} -> i32");
    };

    check(0, 0);
    check(1, 0x0001_0000);
    check(-1, -0x0001_0000); // -1 stays -1 after a right shift back
    check(i16::MAX, 0x7FFF_0000);
    check(i16::MIN, i32::MIN);
}

#[test]
fn integer_narrowing_truncates_toward_negative_infinity() {
    let c = SampleConverter::new(SampleFormat::I32, SampleFormat::I16).unwrap();
    let check = |x: i32, expected: i16| {
        let (_, out) = convert_one(&c, x);
        assert_eq!(read_bytes::<i16>(&out), expected, "i32 {x} -> i16");
    };

    check(0, 0);
    check(0x0001_0000, 1);
    check(0x0001_FFFF, 1); // low bits discarded
    check(-1, -1); // arithmetic shift preserves -1
    check(-0x0001_0000, -1);
    check(i32::MAX, i16::MAX);
    check(i32::MIN, i16::MIN);
}

#[test]
fn signed_unsigned_reencode_through_midpoint_bias() {
    let i16_to_u16 = SampleConverter::new(SampleFormat::I16, SampleFormat::U16).unwrap();
    let u16_to_i16 = SampleConverter::new(SampleFormat::U16, SampleFormat::I16).unwrap();

    let check_pair = |signed: i16, unsigned: u16| {
        let (_, out) = convert_one(&i16_to_u16, signed);
        assert_eq!(read_bytes::<u16>(&out), unsigned, "i16 {signed} -> u16");
        let (_, out) = convert_one(&u16_to_i16, unsigned);
        assert_eq!(read_bytes::<i16>(&out), signed, "u16 {unsigned} -> i16");
    };

    check_pair(0, 32_768); // equilibrium maps to equilibrium
    check_pair(i16::MAX, u16::MAX);
    check_pair(i16::MIN, 0);
    check_pair(1, 32_769);
    check_pair(-1, 32_767);
}

#[test]
fn i24_uses_24_significant_bits_inside_i32_container() {
    let c = SampleConverter::new(SampleFormat::I16, SampleFormat::I24).unwrap();
    let check = |x: i16, expected_inner: i32| {
        let (_, out) = convert_one(&c, x);
        assert_eq!(read_bytes::<i32>(&out), expected_inner, "i16 {x} -> I24");
    };
    check(1, 1 << 8);
    check(i16::MAX, 0x7FFF00);
    check(i16::MIN, -0x800000);
}

#[test]
fn f32_to_f64_is_exact_and_f64_to_f32_rounds() {
    let up = SampleConverter::new(SampleFormat::F32, SampleFormat::F64).unwrap();
    let (_, out) = convert_one(&up, 0.1_f32);
    assert_eq!(read_bytes::<f64>(&out), f64::from(0.1_f32));

    let down = SampleConverter::new(SampleFormat::F64, SampleFormat::F32).unwrap();
    let (_, out) = convert_one(&down, 1.0_f64 / 3.0);
    assert_eq!(read_bytes::<f32>(&out), (1.0_f64 / 3.0) as f32);
}

#[test]
fn u24_round_trips_through_i32_container() {
    let to_u24 = SampleConverter::new(SampleFormat::U16, SampleFormat::U24).unwrap();
    let back = SampleConverter::new(SampleFormat::U24, SampleFormat::U16).unwrap();

    for value in [0u16, 1, 128, 32_768, u16::MAX - 1, u16::MAX] {
        let (_, wide) = convert_one(&to_u24, value);
        let mut narrow_buf = [0u8; 2];
        let written = back.convert(&wide, &mut narrow_buf).unwrap();
        assert_eq!(written, 2);
        assert_eq!(read_bytes::<u16>(&narrow_buf), value);
    }
}

// ---------------------------------------------------------------------------
// 3. Buffer bounds
// ---------------------------------------------------------------------------

#[test]
fn convert_buffer_capacity_is_bounded_and_never_grows() {
    const CHANNELS: u16 = 2;
    const MAX_FRAMES: usize = 512;

    let mut buffer =
        ConvertBuffer::new(SampleFormat::I16, SampleFormat::F32, CHANNELS, MAX_FRAMES).unwrap();

    let max_samples = MAX_FRAMES * CHANNELS as usize;
    let bound = max_samples * size_of::<f32>().max(size_of::<i16>());
    assert_eq!(buffer.capacity_bytes(), bound);
    assert_eq!(buffer.max_frames(), MAX_FRAMES);

    // Repeated conversions of varying legal sizes must not reallocate.
    for frames in [1usize, 7, 64, MAX_FRAMES] {
        let input = vec![0u8; frames * CHANNELS as usize * size_of::<i16>()];
        let out = buffer.convert(&input).unwrap();
        assert_eq!(out.len(), frames * CHANNELS as usize * size_of::<f32>());
        assert_eq!(buffer.capacity_bytes(), bound, "capacity must not grow");
    }

    // Over-capacity input is rejected rather than triggering a reallocation.
    let too_big = vec![0u8; (MAX_FRAMES + 1) * CHANNELS as usize * size_of::<i16>()];
    assert_eq!(
        buffer.convert(&too_big).unwrap_err(),
        ConvertError::OutputTooSmall
    );
}

#[test]
fn convert_buffer_rejects_partial_frames() {
    let mut buffer = ConvertBuffer::new(SampleFormat::F32, SampleFormat::I16, 2, 128).unwrap();
    // 3 bytes is neither frame-aligned (8 bytes) nor sample-aligned (4 bytes).
    assert_eq!(
        buffer.convert(&[0u8; 3]).unwrap_err(),
        ConvertError::FrameCountMismatch
    );
}

#[test]
fn direct_convert_reports_aligned_length_and_size_errors() {
    let c = SampleConverter::new(SampleFormat::F32, SampleFormat::I16).unwrap();
    assert_eq!(c.output_len_for(16), Ok(8));
    assert_eq!(c.output_len_for(5), Err(ConvertError::InputNotAligned));

    let mut small = [0u8; 2];
    assert_eq!(
        c.convert(&[0u8; 8], &mut small).unwrap_err(),
        ConvertError::OutputTooSmall
    );
}

// ---------------------------------------------------------------------------
// 4. Negotiation: distinguishable failures and explicit outcomes
// ---------------------------------------------------------------------------

#[test]
fn negotiation_exact_match_is_zero_copy() {
    let request = FormatRequest::new(2, 48_000, SampleFormat::F32);
    let offers = vec![offer(2, 44_100, 96_000, SampleFormat::F32)];

    let negotiated = negotiate(&request, offers).unwrap();
    assert!(negotiated.is_exact_match());
    assert!(negotiated.input_converter().is_identity());
    assert!(negotiated.output_converter().is_identity());
    assert_eq!(negotiated.device_config().sample_rate(), 48_000);
    assert_eq!(negotiated.requested_format(), SampleFormat::F32);
}

#[test]
fn negotiation_bridges_48k_f32_device_for_44_1k_i16_request() {
    // The motivating scenario: hardware only does 48 kHz f32, the app asks for 44.1 kHz i16.
    let request = FormatRequest::new(2, 44_100, SampleFormat::I16).with_alternative_rate(48_000);
    let offers = vec![offer(2, 48_000, 48_000, SampleFormat::F32)];

    let negotiated = negotiate(&request, offers).unwrap();
    assert!(!negotiated.is_exact_match());
    assert_eq!(negotiated.requested_format(), SampleFormat::I16);
    assert_eq!(
        negotiated.device_config().sample_format(),
        SampleFormat::F32
    );
    assert_eq!(negotiated.device_config().sample_rate(), 48_000);

    // Capture: device f32 -> app i16. Playback: app i16 -> device f32.
    assert_eq!(
        negotiated.input_converter(),
        SampleConverter::new(SampleFormat::F32, SampleFormat::I16).unwrap()
    );
    assert_eq!(
        negotiated.output_converter(),
        SampleConverter::new(SampleFormat::I16, SampleFormat::F32).unwrap()
    );

    // And the bridge actually converts with the documented rules end to end.
    let converter = negotiated.input_converter();
    let (_, out) = convert_one(&converter, 1.0_f32);
    assert_eq!(read_bytes::<i16>(&out), i16::MAX);
}

#[test]
fn negotiation_channel_mismatch_is_a_distinct_error() {
    let request = FormatRequest::new(2, 48_000, SampleFormat::F32);
    let offers = vec![offer(1, 48_000, 48_000, SampleFormat::F32)];
    assert_eq!(
        negotiate(&request, offers).unwrap_err(),
        NegotiationError::ChannelCountMismatch { requested: 2 }
    );
}

#[test]
fn negotiation_rate_mismatch_is_a_distinct_error() {
    let request = FormatRequest::new(2, 44_100, SampleFormat::I16);
    let offers = vec![offer(2, 48_000, 48_000, SampleFormat::F32)];
    let err = negotiate(&request, offers).unwrap_err();
    match err {
        NegotiationError::SampleRateMismatch {
            requested_rates,
            offered_rates,
        } => {
            assert_eq!(requested_rates, 1);
            assert_eq!(offered_rates, 1);
        }
        other => panic!("expected SampleRateMismatch, got {other:?}"),
    }
}

#[test]
fn negotiation_dsd_only_offer_is_unsupported_format() {
    let request = FormatRequest::new(2, 48_000, SampleFormat::I16);
    let offers = vec![offer(2, 48_000, 48_000, SampleFormat::DsdU8)];
    assert_eq!(
        negotiate(&request, offers).unwrap_err(),
        NegotiationError::UnsupportedFormat
    );
}

#[test]
fn negotiation_dsd_request_side_fails_with_conversion_error() {
    let request = FormatRequest::new(2, 48_000, SampleFormat::DsdU8);
    let offers = vec![offer(2, 48_000, 48_000, SampleFormat::F32)];
    assert!(matches!(
        negotiate(&request, offers).unwrap_err(),
        NegotiationError::ConversionFailed(ConvertError::DsdNotSupported)
    ));
}

#[test]
fn negotiation_invalid_request_is_rejected() {
    let request = FormatRequest::new(0, 48_000, SampleFormat::F32);
    assert_eq!(
        negotiate(&request, Vec::new()).unwrap_err(),
        NegotiationError::InvalidRequest
    );
}

#[test]
fn negotiation_prefers_exact_format_over_higher_ranked_fallback() {
    let request = FormatRequest::new(2, 48_000, SampleFormat::I16);
    let offers = vec![
        offer(2, 48_000, 48_000, SampleFormat::F32),
        offer(2, 48_000, 48_000, SampleFormat::I16),
    ];
    let negotiated = negotiate(&request, offers).unwrap();
    assert!(negotiated.is_exact_match());
    assert_eq!(
        negotiated.device_config().sample_format(),
        SampleFormat::I16
    );
}

#[test]
fn negotiation_honors_rate_preference_order() {
    let request = FormatRequest::new(2, 44_100, SampleFormat::F32).with_alternative_rate(48_000);
    let offers = vec![
        offer(2, 44_100, 44_100, SampleFormat::F32),
        offer(2, 48_000, 48_000, SampleFormat::F32),
    ];
    let negotiated = negotiate(&request, offers).unwrap();
    assert_eq!(negotiated.device_config().sample_rate(), 44_100);
}

#[test]
fn negotiation_error_converts_to_cpal_unsupported_config() {
    let request = FormatRequest::new(2, 44_100, SampleFormat::I16);
    let offers = vec![offer(2, 48_000, 48_000, SampleFormat::F32)];
    let err: cpal::Error = negotiate(&request, offers).unwrap_err().into();
    assert_eq!(err.kind(), cpal::ErrorKind::UnsupportedConfig);
}
