//! Negotiation between a requested stream configuration and the configurations a device offers.
//!
//! A device typically exposes a handful of [`SupportedStreamConfigRange`]s, while an application
//! wants one specific sample rate/channel count and a preferred sample format. This module picks
//! the best offer and packages the [`SampleConverter`] needed to bridge a format mismatch.
//!
//! # Selection policy
//!
//! Given a [`FormatRequest`] and the device's offers, [`negotiate`] applies, in order:
//!
//! 1. **Channel count**: only offers matching the requested channel count are considered.
//! 2. **Sample rate**: offers containing one of the requested rates (tried in the request's
//!    preference order) win. If no offered range contains any requested rate, negotiation fails
//!    with [`NegotiationError::SampleRateMismatch`] rather than silently resampling.
//! 3. **Sample format**:
//!    - an offer whose format is exactly one of the requested formats is an *exact match* and
//!      always wins;
//!    - otherwise the highest-ranked convertible PCM offer wins, and the result carries a
//!      non-identity converter between the request's first-choice format and the device format;
//!    - DSD-only offers fail with [`NegotiationError::UnsupportedFormat`] (DSD cannot be bridged
//!      to PCM).
//!
//! The returned [`NegotiatedConfig`] reports whether the match is exact. Exact matches need no
//! conversion at all; a raw stream can run zero-copy with bit-identical data.

use core::fmt;

use crate::{
    ChannelCount, SampleFormat, SampleRate, StreamConfig, SupportedBufferSize,
    SupportedStreamConfig, SupportedStreamConfigRange,
    sample_convert::{ConvertError, SampleConverter},
};

/// Reasons a negotiation can fail. Each variant is structurally distinct so callers can decide
/// whether to retry with different parameters or to surface a hard failure.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NegotiationError {
    /// No offered configuration uses the requested channel count.
    ChannelCountMismatch {
        /// The channel count requested by the application.
        requested: ChannelCount,
    },
    /// No offered configuration's sample-rate range covers any requested sample rate.
    SampleRateMismatch {
        /// The rates requested by the application, in preference order.
        requested_rates: u32,
        /// Rates the device offers, one per remaining candidate range.
        offered_rates: u32,
    },
    /// No offered configuration uses a PCM format the application can convert to or from (e.g.
    /// only DSD was offered, or no offers remained after channel/rate filtering).
    UnsupportedFormat,
    /// Building the bridge converter failed (currently only possible for DSD pairs).
    ConversionFailed(ConvertError),
    /// The request itself was invalid (zero channels, no sample rates, no formats).
    InvalidRequest,
}

impl fmt::Display for NegotiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChannelCountMismatch { requested } => write!(
                f,
                "no device configuration supports {requested} channel(s)"
            ),
            Self::SampleRateMismatch {
                requested_rates,
                offered_rates,
            } => write!(
                f,
                "no offered sample-rate range covers any of the {requested_rates} requested rate(s); \
                 device offers {offered_rates} range(s)"
            ),
            Self::UnsupportedFormat => {
                f.write_str("no offered sample format is convertible to a requested format")
            }
            Self::ConversionFailed(err) => write!(f, "cannot build sample converter: {err}"),
            Self::InvalidRequest => f.write_str(
                "format request must have at least one sample rate and one sample format and a non-zero channel count",
            ),
        }
    }
}

impl std::error::Error for NegotiationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ConversionFailed(err) => Some(err),
            _ => None,
        }
    }
}

impl From<NegotiationError> for crate::Error {
    fn from(err: NegotiationError) -> Self {
        crate::Error::with_message(crate::ErrorKind::UnsupportedConfig, err.to_string())
    }
}

/// What an application wants from a stream.
///
/// Construct with [`FormatRequest::new`] and optionally chain
/// [`with_alternative_format`](Self::with_alternative_format) /
/// [`with_alternative_rate`](Self::with_alternative_rate) to express fallback preferences. The
/// first format and first rate are the strongest preference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormatRequest {
    channels: ChannelCount,
    rates: Vec<SampleRate>,
    formats: Vec<SampleFormat>,
}

impl FormatRequest {
    /// Create a request for `channels` channels, preferring `sample_rate` and `sample_format`.
    #[must_use]
    pub fn new(
        channels: ChannelCount,
        sample_rate: SampleRate,
        sample_format: SampleFormat,
    ) -> Self {
        Self {
            channels,
            rates: vec![sample_rate],
            formats: vec![sample_format],
        }
    }

    /// Accept another sample format if the preferred one is unavailable. Earlier additions rank
    /// higher. An exact format match always beats conversion regardless of list position.
    #[must_use]
    pub fn with_alternative_format(mut self, format: SampleFormat) -> Self {
        if !self.formats.contains(&format) {
            self.formats.push(format);
        }
        self
    }

    /// Accept another sample rate if the preferred one is unavailable. Earlier additions rank
    /// higher.
    #[must_use]
    pub fn with_alternative_rate(mut self, rate: SampleRate) -> Self {
        if !self.rates.contains(&rate) {
            self.rates.push(rate);
        }
        self
    }

    /// Requested channel count.
    #[must_use]
    pub fn channels(&self) -> ChannelCount {
        self.channels
    }

    /// Requested sample rates in preference order.
    #[must_use]
    pub fn preferred_rates(&self) -> &[SampleRate] {
        &self.rates
    }

    /// Requested sample formats in preference order.
    #[must_use]
    pub fn preferred_formats(&self) -> &[SampleFormat] {
        &self.formats
    }
}

/// The outcome of a successful negotiation.
#[derive(Clone, Debug)]
pub struct NegotiatedConfig {
    config: SupportedStreamConfig,
    requested_format: SampleFormat,
    exact: bool,
    converter: SampleConverter,
}

impl NegotiatedConfig {
    /// The concrete device configuration the stream should be opened with.
    #[must_use]
    pub fn device_config(&self) -> SupportedStreamConfig {
        self.config
    }

    /// The sample format the application's callback will work in.
    #[must_use]
    pub fn requested_format(&self) -> SampleFormat {
        self.requested_format
    }

    /// `true` when the device format equals a requested format: no conversion is needed and data
    /// can flow zero-copy and bit-identically.
    #[must_use]
    pub fn is_exact_match(&self) -> bool {
        self.exact
    }

    /// Converter moving samples from the device's format into the application's requested format.
    /// For an input (capture) stream, apply this to the data delivered by the device before
    /// handing it to application code.
    ///
    /// For an exact match this converter is an identity converter.
    #[must_use]
    pub fn input_converter(&self) -> SampleConverter {
        self.converter
    }

    /// Converter moving samples from the application's requested format into the device's format.
    /// For an output (playback) stream, apply this to application-produced data before handing it
    /// to the device.
    ///
    /// For an exact match this converter is an identity converter.
    #[must_use]
    pub fn output_converter(&self) -> SampleConverter {
        self.converter.reversed()
    }

    /// The [`StreamConfig`] (without sample format) to pass to a raw stream builder.
    #[must_use]
    pub fn stream_config(&self) -> StreamConfig {
        self.config.into()
    }

    /// The device's buffer-size constraints.
    #[must_use]
    pub fn buffer_size(&self) -> SupportedBufferSize {
        *self.config.buffer_size()
    }
}

/// Preference rank for a device format when no exact match is available. Higher wins.
///
/// Floats are preferred (they are the universal realtime formats), then wider integer formats,
/// then signed over unsigned. DSD is not convertible and ranks below everything.
fn fallback_format_rank(format: SampleFormat) -> u8 {
    match format {
        SampleFormat::F64 => 14,
        SampleFormat::F32 => 13,
        SampleFormat::I64 => 12,
        SampleFormat::I32 => 11,
        SampleFormat::I24 => 10,
        SampleFormat::I16 => 9,
        SampleFormat::I8 => 8,
        SampleFormat::U64 => 7,
        SampleFormat::U32 => 6,
        SampleFormat::U24 => 5,
        SampleFormat::U16 => 4,
        SampleFormat::U8 => 3,
        SampleFormat::DsdU8 | SampleFormat::DsdU16 | SampleFormat::DsdU32 => 0,
    }
}

/// Pick the best configuration among `offers` for `request` and describe how to bridge any format
/// mismatch.
///
/// See the [module documentation](self) for the full selection policy.
///
/// # Errors
///
/// See [`NegotiationError`] for the distinguishable failure cases.
pub fn negotiate<I>(
    request: &FormatRequest,
    offers: I,
) -> Result<NegotiatedConfig, NegotiationError>
where
    I: IntoIterator<Item = SupportedStreamConfigRange>,
{
    if request.channels == 0 || request.rates.is_empty() || request.formats.is_empty() {
        return Err(NegotiationError::InvalidRequest);
    }

    // DSD cannot be bridged to PCM; reject it up front so the failure reason is explicit even
    // before any offer is inspected.
    if request.formats.iter().all(|format| format.is_dsd()) {
        return Err(NegotiationError::ConversionFailed(
            ConvertError::DsdNotSupported,
        ));
    }

    let offers: Vec<SupportedStreamConfigRange> = offers.into_iter().collect();

    // Step 1: channel count.
    let same_channels: Vec<SupportedStreamConfigRange> = offers
        .iter()
        .copied()
        .filter(|range| range.channels() == request.channels)
        .collect();
    if same_channels.is_empty() {
        return Err(NegotiationError::ChannelCountMismatch {
            requested: request.channels,
        });
    }

    // Step 2: sample rate, honoring the request's rate preference order. Among ranges covering a
    // requested rate, an exact requested-format match always beats a convertible fallback; ties
    // are broken by the fallback format rank.
    let mut chosen: Option<(SampleRate, SupportedStreamConfigRange)> = None;
    for &rate in &request.rates {
        let best = same_channels
            .iter()
            .filter(|range| range.contains_rate(rate))
            .max_by_key(|range| {
                (
                    request.formats.contains(&range.sample_format()),
                    fallback_format_rank(range.sample_format()),
                )
            });
        if let Some(range) = best {
            chosen = Some((rate, *range));
            break;
        }
    }

    let (rate, range) = chosen.ok_or(NegotiationError::SampleRateMismatch {
        requested_rates: request.rates.len() as u32,
        offered_rates: same_channels.len() as u32,
    })?;

    // Step 3: sample format.
    let device_format = range.sample_format();
    let exact = request.formats.contains(&device_format);

    let requested_format = if exact {
        device_format
    } else {
        // No exact format match anywhere: only PCM offers can be bridged. Pick the strongest
        // requested PCM format as the application-side format.
        if device_format.is_dsd() {
            return Err(NegotiationError::UnsupportedFormat);
        }
        *request
            .formats
            .iter()
            .find(|format| !format.is_dsd())
            .ok_or(NegotiationError::UnsupportedFormat)?
    };

    let converter = if exact {
        SampleConverter::new(requested_format, device_format)
    } else {
        // Capture direction: device format -> application format.
        SampleConverter::new(device_format, requested_format)
    }
    .map_err(NegotiationError::ConversionFailed)?;

    let config = range.with_sample_rate(rate);
    Ok(NegotiatedConfig {
        config,
        requested_format,
        exact,
        converter,
    })
}

/// Convenience wrapper: negotiate for an input (capture) stream.
///
/// Functionally identical to [`negotiate`]; the name documents intent at the call site.
pub fn negotiate_input<I>(
    request: &FormatRequest,
    offers: I,
) -> Result<NegotiatedConfig, NegotiationError>
where
    I: IntoIterator<Item = SupportedStreamConfigRange>,
{
    negotiate(request, offers)
}

/// Convenience wrapper: negotiate for an output (playback) stream.
///
/// Functionally identical to [`negotiate`]; the name documents intent at the call site. Use
/// [`NegotiatedConfig::output_converter`] to obtain the application-format -> device-format
/// converter.
pub fn negotiate_output<I>(
    request: &FormatRequest,
    offers: I,
) -> Result<NegotiatedConfig, NegotiationError>
where
    I: IntoIterator<Item = SupportedStreamConfigRange>,
{
    negotiate(request, offers)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(
        channels: ChannelCount,
        min_rate: SampleRate,
        max_rate: SampleRate,
        format: SampleFormat,
    ) -> SupportedStreamConfigRange {
        SupportedStreamConfigRange::new(
            channels,
            min_rate,
            max_rate,
            SupportedBufferSize::Range {
                min: 256,
                max: 1024,
            },
            format,
        )
    }

    #[test]
    fn invalid_request_is_rejected() {
        let request = FormatRequest::new(0, 48_000, SampleFormat::F32);
        assert_eq!(
            negotiate(&request, Vec::new()).unwrap_err(),
            NegotiationError::InvalidRequest
        );
    }

    #[test]
    fn channel_mismatch_is_distinguishable() {
        let request = FormatRequest::new(2, 48_000, SampleFormat::F32);
        let offers = vec![range(1, 48_000, 48_000, SampleFormat::F32)];
        assert_eq!(
            negotiate(&request, offers).unwrap_err(),
            NegotiationError::ChannelCountMismatch { requested: 2 }
        );
    }

    #[test]
    fn rate_mismatch_is_distinguishable() {
        let request = FormatRequest::new(2, 44_100, SampleFormat::I16);
        let offers = vec![range(2, 48_000, 48_000, SampleFormat::F32)];
        assert!(matches!(
            negotiate(&request, offers).unwrap_err(),
            NegotiationError::SampleRateMismatch { .. }
        ));
    }

    #[test]
    fn dsd_only_offer_is_unsupported_format() {
        let request = FormatRequest::new(2, 48_000, SampleFormat::I16);
        let offers = vec![range(2, 48_000, 48_000, SampleFormat::DsdU8)];
        assert_eq!(
            negotiate(&request, offers).unwrap_err(),
            NegotiationError::UnsupportedFormat
        );
    }

    #[test]
    fn exact_match_is_zero_copy() {
        let request = FormatRequest::new(2, 48_000, SampleFormat::F32);
        let offers = vec![range(2, 44_100, 96_000, SampleFormat::F32)];
        let negotiated = negotiate(&request, offers).unwrap();
        assert!(negotiated.is_exact_match());
        assert!(negotiated.input_converter().is_identity());
        assert!(negotiated.output_converter().is_identity());
        assert_eq!(negotiated.device_config().sample_rate(), 48_000);
    }

    #[test]
    fn format_mismatch_yields_bridge_converter() {
        // The motivating scenario: device only does 48 kHz f32, app asks for 44.1 kHz i16.
        let request =
            FormatRequest::new(2, 44_100, SampleFormat::I16).with_alternative_rate(48_000);
        let offers = vec![range(2, 48_000, 48_000, SampleFormat::F32)];
        let negotiated = negotiate(&request, offers).unwrap();

        assert!(!negotiated.is_exact_match());
        assert_eq!(negotiated.requested_format(), SampleFormat::I16);
        assert_eq!(
            negotiated.device_config().sample_format(),
            SampleFormat::F32
        );
        assert_eq!(
            negotiated.input_converter(),
            SampleConverter::new(SampleFormat::F32, SampleFormat::I16).unwrap()
        );
        assert_eq!(
            negotiated.output_converter(),
            SampleConverter::new(SampleFormat::I16, SampleFormat::F32).unwrap()
        );
    }

    #[test]
    fn rate_preference_order_is_honored() {
        let request =
            FormatRequest::new(2, 44_100, SampleFormat::F32).with_alternative_rate(48_000);
        let offers = vec![
            range(2, 44_100, 44_100, SampleFormat::F32),
            range(2, 48_000, 48_000, SampleFormat::F32),
        ];
        let negotiated = negotiate(&request, offers).unwrap();
        assert_eq!(negotiated.device_config().sample_rate(), 44_100);
    }

    #[test]
    fn exact_format_beats_higher_ranked_fallback() {
        // I16 is requested and offered alongside a "better" F32 offer; the exact I16 match wins.
        let request = FormatRequest::new(2, 48_000, SampleFormat::I16);
        let offers = vec![
            range(2, 48_000, 48_000, SampleFormat::F32),
            range(2, 48_000, 48_000, SampleFormat::I16),
        ];
        let negotiated = negotiate(&request, offers).unwrap();
        assert!(negotiated.is_exact_match());
        assert_eq!(
            negotiated.device_config().sample_format(),
            SampleFormat::I16
        );
    }
}
