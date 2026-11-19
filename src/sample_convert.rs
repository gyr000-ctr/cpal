//! Core-level sample format conversion with explicit, testable rounding and clipping rules.
//!
//! This module converts raw sample buffers between the PCM [`SampleFormat`]s that CPAL supports
//! (`i8`/`i16`/`I24`/`i32`/`i64`, `u8`/`u16`/`U24`/`u32`/`u64`, `f32`/`f64`). DSD formats have no
//! PCM representation and are rejected by [`SampleConverter::new`].
//!
//! # Conversion rules
//!
//! The rules are intentionally deterministic so that every output sample can be asserted against a
//! known input sample:
//!
//! ## Integer to integer
//!
//! Every integer sample is first mapped to a *signed code* (`i128`). Signed samples keep their
//! value; unsigned samples have their midpoint bias removed (`u16::MIN` is encoded as
//! `-32_768`, the `u16` midpoint `32_768` as `0`). The code is then placed in the target format:
//!
//! - **Widening** shifts the code left into the most significant bits, preserving full scale
//!   (e.g. `i16::MAX` converts to `0x7FFF_0000` in `i32`, never to `i32::MAX`; `1i16` becomes
//!   `0x0001_0000` in `i32`).
//! - **Narrowing** arithmetically shifts the code right, discarding the low bits with truncation
//!   toward negative infinity (e.g. an `i32` of `-1` converts to an `i16` of `-1`).
//! - Signed/unsigned re-encoding re-applies the target midpoint bias; `0` always maps to the
//!   target equilibrium.
//!
//! ## Integer to float
//!
//! The signed code is divided by the 64-bit full-scale magnitude, so every representable integer
//! maps into `-1.0..=1.0`. The widest integers cannot be represented exactly in `f32`; the
//! platform's nearest representable value is produced.
//!
//! ## Float to integer
//!
//! 1. `NaN` maps to the target equilibrium (`0` / the unsigned midpoint).
//! 2. The value is scaled to the target's integer range and **clipped** to the target's signed
//!    code range, so out-of-range and infinite inputs saturate to the minimum/maximum code.
//! 3. Finite in-range values are rounded **half away from zero** (`0.5 -> +1`, `-0.5 -> -1`).
//!
//! ## Float to float
//!
//! `f32 -> f64` is exact; `f64 -> f32` uses the platform's round-to-nearest, ties-to-even cast.
//!
//! # Allocation and buffer bounds
//!
//! No conversion function allocates. [`SampleConverter::output_len_for`] gives the exact output
//! size, and [`ConvertBuffer`] owns a buffer whose capacity never exceeds
//! `max_frames * channels * max(target_size, source_size)` bytes and never grows after
//! construction.

use core::fmt;

use crate::{I24, SampleFormat, U24};

/// Errors returned by the conversion primitives in this module.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ConvertError {
    /// A DSD [`SampleFormat`] was requested; DSD streams have no PCM representation and cannot be
    /// converted with this module.
    DsdNotSupported,
    /// The input byte length is not a whole number of samples for its [`SampleFormat`].
    InputNotAligned,
    /// The output buffer is too small to hold the converted samples.
    OutputTooSmall,
    /// The number of samples is not a whole number of frames for the given channel count.
    FrameCountMismatch,
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DsdNotSupported => {
                f.write_str("DSD sample formats cannot be converted to or from PCM sample formats")
            }
            Self::InputNotAligned => {
                f.write_str("input byte length is not aligned to the sample size")
            }
            Self::OutputTooSmall => f.write_str("output buffer is too small for the conversion"),
            Self::FrameCountMismatch => {
                f.write_str("sample count is not a whole number of frames for the channel count")
            }
        }
    }
}

impl std::error::Error for ConvertError {}

/// Returns `true` if `format` is a PCM format the converter knows how to handle.
#[must_use]
pub const fn is_pcm(format: SampleFormat) -> bool {
    matches!(
        format,
        SampleFormat::I8
            | SampleFormat::I16
            | SampleFormat::I24
            | SampleFormat::I32
            | SampleFormat::I64
            | SampleFormat::U8
            | SampleFormat::U16
            | SampleFormat::U24
            | SampleFormat::U32
            | SampleFormat::U64
            | SampleFormat::F32
            | SampleFormat::F64
    )
}

/// Internal representation shared by the twelve PCM sample types.
///
/// Integer samples are carried by their *signed code* (unsigned samples have their midpoint bias
/// removed). Floating samples are carried as `f64` (an `f32` widens losslessly).
#[derive(Clone, Copy, Debug)]
enum SampleRepr {
    Int(i128),
    Float(f64),
}

/// Write cursor over a caller-provided byte slice; keeps conversion allocation-free.
struct ByteCursor<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn write<T: Copy>(&mut self, value: T) {
        debug_assert!(self.pos + size_of::<T>() <= self.buf.len());
        unsafe {
            (self.buf.as_mut_ptr().add(self.pos) as *mut T).write_unaligned(value);
        }
        self.pos += size_of::<T>();
    }
}

fn read_unaligned<T: Copy>(raw: &[u8]) -> T {
    debug_assert!(raw.len() >= size_of::<T>());
    unsafe { (raw.as_ptr() as *const T).read_unaligned() }
}

/// Per-format behaviour used by the generic conversion routine.
trait PcmSample: Copy {
    const FORMAT: SampleFormat;

    /// Decode one sample into its internal representation.
    fn decode(raw: &[u8]) -> SampleRepr;

    /// Encode one internal representation back into this sample type.
    ///
    /// `source` is the format the representation was decoded from, which determines integer
    /// widening/narrowing shifts.
    fn encode(repr: SampleRepr, source: SampleFormat, out: &mut ByteCursor<'_>);
}

/// Number of bits carried by an integer format (I24/U24 use 24 bits inside a 4-byte container).
const fn int_bits(format: SampleFormat) -> u32 {
    match format {
        SampleFormat::I8 | SampleFormat::U8 => 8,
        SampleFormat::I16 | SampleFormat::U16 => 16,
        SampleFormat::I24 | SampleFormat::U24 => 24,
        SampleFormat::I32 | SampleFormat::U32 => 32,
        SampleFormat::I64 | SampleFormat::U64 => 64,
        // Every caller only handles integer formats; floats and DSD are filtered out earlier.
        _ => panic!("int_bits called on a non-integer format"),
    }
}

/// Signed code range of an integer format: `-(1 << (bits - 1))..=(1 << (bits - 1)) - 1`.
fn int_code_range(format: SampleFormat) -> (i128, i128) {
    let bits = int_bits(format);
    let half = 1i128 << (bits - 1);
    (-half, half - 1)
}

/// Place a signed `code` (decoded from `source`) into integer format `target`, applying
/// widening/narrowing shifts and the unsigned midpoint bias.
fn encode_int(code: i128, source: SampleFormat, target: SampleFormat) -> i128 {
    let src_bits = int_bits(source);
    let dst_bits = int_bits(target);
    let shifted = if dst_bits >= src_bits {
        code << (dst_bits - src_bits)
    } else {
        code >> (src_bits - dst_bits)
    };
    if target.is_uint() {
        shifted + (1i128 << (dst_bits - 1))
    } else {
        shifted
    }
}

/// Round `x` half away from zero (`0.5 -> 1`, `-0.5 -> -1`). Negative zero stays `0`.
fn round_half_away(x: f64) -> f64 {
    let rounded = x.abs().round() * x.signum();
    if rounded == 0.0 { 0.0 } else { rounded }
}

/// Convert a floating sample to the signed integer code of `target`.
///
/// NaN maps to zero; everything else is clipped to the target range and rounded half away from
/// zero.
fn float_to_int_code(x: f64, target: SampleFormat) -> i128 {
    if x.is_nan() {
        return 0;
    }
    let (min_code, max_code) = int_code_range(target);
    let scale = (-(min_code) as f64) * x;
    if scale <= min_code as f64 {
        min_code
    } else if scale >= max_code as f64 {
        max_code
    } else {
        (round_half_away(scale) as i128).clamp(min_code, max_code)
    }
}

/// Divide a signed integer code (decoded from `source`) by that source format's full-scale
/// magnitude, yielding a value in `-1.0..=1.0`.
fn int_code_to_float(code: i128, source: SampleFormat) -> f64 {
    let full_scale = (1i128 << (int_bits(source) - 1)) as f64;
    let abs = (code.unsigned_abs() as f64) / full_scale;
    if code < 0 { -abs } else { abs }
}

macro_rules! impl_signed_int_sample {
    ($t:ty, $format:expr) => {
        impl PcmSample for $t {
            const FORMAT: SampleFormat = $format;

            fn decode(raw: &[u8]) -> SampleRepr {
                SampleRepr::Int(read_unaligned::<$t>(raw) as i128)
            }

            fn encode(repr: SampleRepr, source: SampleFormat, out: &mut ByteCursor<'_>) {
                let value = match repr {
                    SampleRepr::Int(code) => encode_int(code, source, Self::FORMAT),
                    SampleRepr::Float(x) => float_to_int_code(x, Self::FORMAT),
                };
                out.write(value as $t);
            }
        }
    };
}

macro_rules! impl_unsigned_int_sample {
    ($t:ty, $format:expr) => {
        impl PcmSample for $t {
            const FORMAT: SampleFormat = $format;

            fn decode(raw: &[u8]) -> SampleRepr {
                let bias = 1i128 << (int_bits(Self::FORMAT) - 1);
                SampleRepr::Int(read_unaligned::<$t>(raw) as i128 - bias)
            }

            fn encode(repr: SampleRepr, source: SampleFormat, out: &mut ByteCursor<'_>) {
                let value = match repr {
                    SampleRepr::Int(code) => encode_int(code, source, Self::FORMAT),
                    SampleRepr::Float(x) => {
                        float_to_int_code(x, Self::FORMAT) + (1i128 << (int_bits(Self::FORMAT) - 1))
                    }
                };
                out.write(value as $t);
            }
        }
    };
}

impl_signed_int_sample!(i8, SampleFormat::I8);
impl_signed_int_sample!(i16, SampleFormat::I16);
impl_signed_int_sample!(i32, SampleFormat::I32);
impl_signed_int_sample!(i64, SampleFormat::I64);
impl_unsigned_int_sample!(u8, SampleFormat::U8);
impl_unsigned_int_sample!(u16, SampleFormat::U16);
impl_unsigned_int_sample!(u32, SampleFormat::U32);
impl_unsigned_int_sample!(u64, SampleFormat::U64);

impl PcmSample for I24 {
    const FORMAT: SampleFormat = SampleFormat::I24;

    fn decode(raw: &[u8]) -> SampleRepr {
        // I24 stores its 24 significant bits in an i32 container.
        SampleRepr::Int(read_unaligned::<i32>(raw) as i128)
    }

    fn encode(repr: SampleRepr, source: SampleFormat, out: &mut ByteCursor<'_>) {
        let value = match repr {
            SampleRepr::Int(code) => encode_int(code, source, Self::FORMAT),
            SampleRepr::Float(x) => float_to_int_code(x, Self::FORMAT),
        };
        out.write(value as i32);
    }
}

impl PcmSample for U24 {
    const FORMAT: SampleFormat = SampleFormat::U24;

    fn decode(raw: &[u8]) -> SampleRepr {
        // U24 stores its 24 significant bits in an i32 container.
        SampleRepr::Int(read_unaligned::<i32>(raw) as i128 - (1i128 << 23))
    }

    fn encode(repr: SampleRepr, source: SampleFormat, out: &mut ByteCursor<'_>) {
        let value = match repr {
            SampleRepr::Int(code) => encode_int(code, source, Self::FORMAT),
            SampleRepr::Float(x) => float_to_int_code(x, Self::FORMAT) + (1i128 << 23),
        };
        out.write(value as i32);
    }
}

impl PcmSample for f32 {
    const FORMAT: SampleFormat = SampleFormat::F32;

    fn decode(raw: &[u8]) -> SampleRepr {
        SampleRepr::Float(read_unaligned::<f32>(raw) as f64)
    }

    fn encode(repr: SampleRepr, source: SampleFormat, out: &mut ByteCursor<'_>) {
        let value = match repr {
            SampleRepr::Float(x) => x as f32,
            SampleRepr::Int(code) => int_code_to_float(code, source) as f32,
        };
        out.write(value);
    }
}

impl PcmSample for f64 {
    const FORMAT: SampleFormat = SampleFormat::F64;

    fn decode(raw: &[u8]) -> SampleRepr {
        SampleRepr::Float(read_unaligned::<f64>(raw))
    }

    fn encode(repr: SampleRepr, source: SampleFormat, out: &mut ByteCursor<'_>) {
        let value = match repr {
            SampleRepr::Float(x) => x,
            SampleRepr::Int(code) => int_code_to_float(code, source),
        };
        out.write(value);
    }
}

/// Converts raw PCM sample buffers from one [`SampleFormat`] to another.
///
/// Construct once per stream and reuse it for every callback. The converter is stateless beyond
/// its format pair, so it can be shared across callbacks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SampleConverter {
    source: SampleFormat,
    target: SampleFormat,
}

impl SampleConverter {
    /// Build a converter between two PCM formats.
    ///
    /// # Errors
    ///
    /// Returns [`ConvertError::DsdNotSupported`] if either side is a DSD format.
    pub fn new(source: SampleFormat, target: SampleFormat) -> Result<Self, ConvertError> {
        if source.is_dsd() || target.is_dsd() {
            return Err(ConvertError::DsdNotSupported);
        }
        Ok(Self { source, target })
    }

    /// The format consumed by [`convert`](Self::convert).
    #[must_use]
    pub fn source_format(&self) -> SampleFormat {
        self.source
    }

    /// The format produced by [`convert`](Self::convert).
    #[must_use]
    pub fn target_format(&self) -> SampleFormat {
        self.target
    }

    /// Returns `true` when source and target are the same format, in which case
    /// [`convert`](Self::convert) copies bytes verbatim and output is bit-identical to the input.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.source == self.target
    }

    /// Build the inverse converter: its source is this converter's target and vice versa.
    ///
    /// This is useful for duplex/playback paths that need to convert application data back into a
    /// device format.
    #[must_use]
    pub fn reversed(&self) -> Self {
        Self {
            source: self.target,
            target: self.source,
        }
    }

    /// Output byte length for an input of `input_len` bytes, or [`ConvertError::InputNotAligned`]
    /// if the input is not a whole number of source samples.
    pub fn output_len_for(&self, input_len: usize) -> Result<usize, ConvertError> {
        if input_len % self.source.sample_size() != 0 {
            return Err(ConvertError::InputNotAligned);
        }
        let samples = input_len / self.source.sample_size();
        Ok(samples * self.target.sample_size())
    }

    /// Convert every sample in `input`, writing the result into `output`.
    ///
    /// On success returns the number of bytes written. An identity conversion is a plain
    /// `copy_from_slice`, so identical formats are bit-for-bit identical with no per-sample work.
    /// This function never allocates.
    ///
    /// # Errors
    ///
    /// - [`ConvertError::InputNotAligned`] if `input.len()` is not a multiple of the source sample
    ///   size.
    /// - [`ConvertError::OutputTooSmall`] if `output` cannot hold every converted sample.
    pub fn convert(&self, input: &[u8], output: &mut [u8]) -> Result<usize, ConvertError> {
        if input.len() % self.source.sample_size() != 0 {
            return Err(ConvertError::InputNotAligned);
        }
        let needed = self.output_len_for(input.len())?;
        if output.len() < needed {
            return Err(ConvertError::OutputTooSmall);
        }

        if self.is_identity() {
            output[..input.len()].copy_from_slice(input);
            return Ok(input.len());
        }

        let mut cursor = ByteCursor::new(&mut output[..needed]);
        convert_dispatch(self.source, self.target, input, &mut cursor);
        debug_assert_eq!(cursor.pos, needed);
        Ok(needed)
    }
}

/// Macro plumbing: decode via the source type, then encode into the target type.
macro_rules! convert_pair {
    ($src:ty, $dst:ty, $input:expr, $out:expr, $source:expr) => {{
        let mut rest = $input;
        while rest.len() >= size_of::<$src>() {
            let repr = <$src as PcmSample>::decode(rest);
            <$dst as PcmSample>::encode(repr, $source, $out);
            rest = &rest[size_of::<$src>()..];
        }
    }};
}

/// Macro plumbing: expand the target arm for one source type.
macro_rules! convert_targets {
    ($src:ty, $target:expr, $input:expr, $out:expr, $source:expr, $($dst:ty),*) => {
        match $target {
            $(<$dst as PcmSample>::FORMAT => convert_pair!($src, $dst, $input, $out, $source),)*
            _ => unreachable!("non-PCM formats are rejected by SampleConverter::new"),
        }
    };
}

/// Pick the (source, target) implementation pair at runtime.
fn convert_dispatch(
    source: SampleFormat,
    target: SampleFormat,
    input: &[u8],
    out: &mut ByteCursor<'_>,
) {
    macro_rules! arm {
        ($src:ty) => {
            convert_targets!(
                $src, target, input, out, source, i8, i16, I24, i32, i64, u8, u16, U24, u32, u64,
                f32, f64
            )
        };
    }
    match source {
        SampleFormat::I8 => arm!(i8),
        SampleFormat::I16 => arm!(i16),
        SampleFormat::I24 => arm!(I24),
        SampleFormat::I32 => arm!(i32),
        SampleFormat::I64 => arm!(i64),
        SampleFormat::U8 => arm!(u8),
        SampleFormat::U16 => arm!(u16),
        SampleFormat::U24 => arm!(U24),
        SampleFormat::U32 => arm!(u32),
        SampleFormat::U64 => arm!(u64),
        SampleFormat::F32 => arm!(f32),
        SampleFormat::F64 => arm!(f64),
        _ => unreachable!("non-PCM formats are rejected by SampleConverter::new"),
    }
}

/// A reusable, capacity-bounded buffer for conversions performed inside a stream callback.
///
/// The allocation happens once, at construction. [`convert`](Self::convert) only writes into the
/// existing allocation, so subsequent calls never allocate and the capacity never grows. It is
/// bounded by `max_frames * channels * max(source_size, target_size)` bytes.
#[derive(Debug)]
pub struct ConvertBuffer {
    converter: SampleConverter,
    channels: usize,
    max_frames: usize,
    buf: Vec<u8>,
}

impl ConvertBuffer {
    /// Allocate a buffer large enough to convert `max_frames` frames of `channels` samples between
    /// `source` and `target`.
    ///
    /// # Errors
    ///
    /// Propagates the [`ConvertError`]s from [`SampleConverter::new`], or
    /// [`ConvertError::FrameCountMismatch`] if `channels` is zero.
    pub fn new(
        source: SampleFormat,
        target: SampleFormat,
        channels: u16,
        max_frames: usize,
    ) -> Result<Self, ConvertError> {
        if channels == 0 {
            return Err(ConvertError::FrameCountMismatch);
        }
        let converter = SampleConverter::new(source, target)?;
        let max_bytes = max_frames
            .checked_mul(channels as usize)
            .and_then(|samples| {
                samples.checked_mul(
                    converter
                        .target
                        .sample_size()
                        .max(converter.source.sample_size()),
                )
            })
            .unwrap_or(0);
        Ok(Self {
            converter,
            channels: channels as usize,
            max_frames,
            buf: Vec::with_capacity(max_bytes),
        })
    }

    /// The underlying converter.
    #[must_use]
    pub fn converter(&self) -> &SampleConverter {
        &self.converter
    }

    /// Maximum number of frames this buffer was sized for.
    #[must_use]
    pub fn max_frames(&self) -> usize {
        self.max_frames
    }

    /// Allocated capacity in bytes; never grows after construction.
    #[must_use]
    pub fn capacity_bytes(&self) -> usize {
        self.buf.capacity()
    }

    /// Convert `input` and return the converted bytes.
    ///
    /// The returned slice is borrowed from this buffer and stays valid until the next call. This
    /// method never allocates.
    ///
    /// # Errors
    ///
    /// - [`ConvertError::FrameCountMismatch`] if the input does not contain a whole number of
    ///   frames for the configured channel count.
    /// - [`ConvertError::InputNotAligned`] if the byte length is not a multiple of the source
    ///   sample size.
    /// - [`ConvertError::OutputTooSmall`] if more than `max_frames` frames were supplied.
    pub fn convert<'a>(&'a mut self, input: &[u8]) -> Result<&'a [u8], ConvertError> {
        let frame_bytes = self.converter.source.sample_size() * self.channels;
        if input.len() % frame_bytes != 0 {
            return Err(ConvertError::FrameCountMismatch);
        }
        let frames = input.len() / frame_bytes;
        if frames > self.max_frames {
            return Err(ConvertError::OutputTooSmall);
        }
        let needed = self.converter.output_len_for(input.len())?;
        debug_assert!(self.buf.capacity() >= needed);
        self.buf.clear();
        self.buf.resize(needed, 0);
        let written = self.converter.convert(input, &mut self.buf)?;
        debug_assert_eq!(written, needed);
        Ok(&self.buf[..written])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsd_is_rejected() {
        assert_eq!(
            SampleConverter::new(SampleFormat::DsdU8, SampleFormat::F32).unwrap_err(),
            ConvertError::DsdNotSupported
        );
        assert_eq!(
            SampleConverter::new(SampleFormat::F32, SampleFormat::DsdU16).unwrap_err(),
            ConvertError::DsdNotSupported
        );
    }

    #[test]
    fn pcm_predicate_matches_the_twelve_formats() {
        let pcm = [
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
        ];
        for format in pcm {
            assert!(is_pcm(format));
        }
        assert!(!is_pcm(SampleFormat::DsdU8));
        assert!(!is_pcm(SampleFormat::DsdU16));
        assert!(!is_pcm(SampleFormat::DsdU32));
    }
}
