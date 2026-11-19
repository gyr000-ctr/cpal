//! Reporting of device hotplug and default-route changes.
//!
//! A [`Device`](crate::Device) handle is only a handle. When the hardware is unplugged, or when
//! the system default device changes, an existing handle keeps referring to the device it was
//! created for. Applications that never learn about that keep using a stale handle, or keep
//! running a stream on a device that is gone, until they are restarted. This module adds a
//! subscription that reports those changes.
//!
//! # Layers
//!
//! - [`HostTrait::device_events`] watches the whole host: devices appearing and disappearing,
//!   and the system default input/output changing.
//! - [`DeviceTrait::device_events`] watches one device within a host: whether that device is
//!   still present, and default-route changes in the directions it supports.
//!
//! Both report the same [`DeviceEvent`], and neither changes what
//! [`devices`](crate::traits::HostTrait::devices),
//! [`default_input_device`](crate::traits::HostTrait::default_input_device) or
//! [`default_output_device`](crate::traits::HostTrait::default_output_device) return. A
//! watcher only reads them.
//!
//! # Ordering and deduplication
//!
//! A single poll reports its changes in a fixed order:
//!
//! 1. devices that disappeared, in ascending [`DeviceId`] order;
//! 2. devices that appeared, in ascending [`DeviceId`] order;
//! 3. [`DeviceEventKind::DefaultInputChanged`];
//! 4. [`DeviceEventKind::DefaultOutputChanged`].
//!
//! Ordering compares the host name first, then the backend identifier, so it does not depend on
//! enumeration order and is the same on every run. Device-level watchers order the same way,
//! with their own presence change first.
//!
//! Events are only emitted when the observed state changed, so polling repeatedly reports
//! nothing. A device reported twice by the backend is still one device. A default that changes
//! to a device that was just added is reported after that device's [`DeviceEventKind::Added`].
//! An event identical to the one already at the end of the queue is coalesced, so a backend
//! that signals the same change more than once does not multiply it.
//!
//! A default with no device behind it is reported with [`DeviceEvent::device`] returning
//! `None`, for example when the last output device is removed.
//!
//! # Allocation
//!
//! A watcher allocates its event queue and its snapshot buffers once, when it is created, and
//! reuses them. Delivering an event, deduplicating it, and ordering a batch do not allocate.
//! The queue never grows: when it is full the oldest event is dropped and counted in
//! [`DeviceEvents::dropped_events`].
//!
//! Comparing snapshots means calling [`HostTrait::devices`] and the default-device queries,
//! which belong to the backend and may allocate there depending on the host. The watcher adds
//! no allocation of its own on top.
//!
//! # Platform support
//!
//! The subscription is implemented on top of device enumeration and the default-device
//! queries, so it works on every host that can enumerate devices, and it never panics on a host
//! that cannot:
//!
//! - a host whose enumeration fails reports that error, for example
//!   [`ErrorKind::HostUnavailable`];
//! - a host that reports devices but cannot identify any of them (every [`DeviceTrait::id`]
//!   call fails) cannot be watched, and reports [`ErrorKind::UnsupportedOperation`].
//!
//! The two are distinguishable: the first is a transient failure that may clear on the next
//! poll, the second is a property of the host that will not change by waiting.
//!
//! Because the watcher compares identities, a default that is a routing alias rather than a
//! device (ALSA's `default` PCM, for example) only changes identity when the alias itself
//! changes; a route change behind the alias is not visible to polling. A backend with a native
//! notification API can override the two `device_events` methods and report it directly.
//!
//! A change that happens and is undone between two polls is not reported; polling observes
//! state, it does not log transitions.

use std::{cmp::Ordering, fmt};

use crate::{
    DeviceId, Error, ErrorKind,
    traits::{DeviceTrait, HostTrait},
};

/// Default number of undelivered events a watcher holds before dropping the oldest.
pub const DEFAULT_EVENT_CAPACITY: usize = 64;

/// What happened to a device or to a default route.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceEventKind {
    /// A device became available.
    Added,
    /// A device is no longer available.
    Removed,
    /// The system default input device changed.
    DefaultInputChanged,
    /// The system default output device changed.
    DefaultOutputChanged,
}

impl DeviceEventKind {
    /// Whether this event reports a device appearing or disappearing.
    pub const fn is_hotplug(self) -> bool {
        matches!(self, Self::Added | Self::Removed)
    }

    /// Whether this event reports a default-route change.
    pub const fn is_default_change(self) -> bool {
        matches!(self, Self::DefaultInputChanged | Self::DefaultOutputChanged)
    }
}

impl fmt::Display for DeviceEventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::Added => "device added",
            Self::Removed => "device removed",
            Self::DefaultInputChanged => "default input device changed",
            Self::DefaultOutputChanged => "default output device changed",
        };
        f.write_str(text)
    }
}

/// A single device hotplug or default-route change.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct DeviceEvent {
    kind: DeviceEventKind,
    device: Option<DeviceId>,
}

impl DeviceEvent {
    /// Creates an event for `kind` referring to `device`.
    ///
    /// `device` is `None` when the event leaves the system without a device, for example when
    /// the last output device is removed.
    pub fn new(kind: DeviceEventKind, device: Option<DeviceId>) -> Self {
        Self { kind, device }
    }

    /// What happened.
    pub fn kind(&self) -> DeviceEventKind {
        self.kind
    }

    /// The device the event refers to, or `None` if no device is left.
    pub fn device(&self) -> Option<&DeviceId> {
        self.device.as_ref()
    }

    /// The device the event refers to, consuming the event.
    pub fn into_device(self) -> Option<DeviceId> {
        self.device
    }

    /// The system default changed to this event's device.
    pub fn is_default_change(&self) -> bool {
        self.kind.is_default_change()
    }

    /// A device appeared or disappeared.
    pub fn is_hotplug(&self) -> bool {
        self.kind.is_hotplug()
    }
}

impl fmt::Debug for DeviceEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceEvent")
            .field("kind", &self.kind)
            .field("device", &self.device)
            .finish()
    }
}

impl fmt::Display for DeviceEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.device {
            Some(device) => write!(f, "{}: {device}", self.kind),
            None => write!(f, "{}: no device", self.kind),
        }
    }
}

/// A subscription to device hotplug and default-route changes.
///
/// Create one with [`HostTrait::device_events`] or
/// [`DeviceTrait::device_events`], then call
/// [`poll`](Self::poll) or [`drain`](Self::drain) with the host it belongs to.
///
/// The watcher is seeded with the state at construction, so it reports changes made after it
/// was created, not the devices that already existed.
///
/// ```no_run
/// use cpal::traits::HostTrait;
/// use cpal::DeviceEventKind;
///
/// let host = cpal::default_host();
/// let mut events = host.device_events()?;
/// loop {
///     std::thread::sleep(std::time::Duration::from_millis(500));
///     for event in events.drain(&host)? {
///         if event.is_default_change() || event.kind() == DeviceEventKind::Removed {
///             // The device handles in use may be stale; rebuild them.
///         }
///     }
/// }
/// # Ok::<(), cpal::Error>(())
/// ```
#[derive(Debug)]
pub struct DeviceEvents {
    scope: Scope,
    queue: Vec<DeviceEvent>,
    capacity: usize,
    dropped: u64,
    /// Device ids seen by the previous poll, sorted by [`cmp_device_ids`].
    known: Vec<DeviceId>,
    /// Reused enumeration buffer; swapped with `known` on every successful poll.
    scratch: Vec<DeviceId>,
    known_input_default: Option<DeviceId>,
    known_output_default: Option<DeviceId>,
}

#[derive(Debug)]
enum Scope {
    /// Watch every device and both default routes.
    Host,
    /// Watch one device and the default routes in the directions it supports.
    Device {
        id: DeviceId,
        watch_input: bool,
        watch_output: bool,
        present: bool,
    },
}

impl DeviceEvents {
    /// Watches every device and both default routes of `host`.
    ///
    /// The watcher is seeded with the host's current state and queues no events. The queue
    /// holds up to [`DEFAULT_EVENT_CAPACITY`] undelivered events.
    ///
    /// # Errors
    ///
    /// - [`ErrorKind::HostUnavailable`] (or any other enumeration error) if the host cannot be
    ///   enumerated; see [`HostTrait::devices`].
    /// - [`ErrorKind::UnsupportedOperation`] if the host reports devices but cannot identify
    ///   any of them, so that changes cannot be tracked.
    pub fn for_host<H>(host: &H) -> Result<Self, Error>
    where
        H: HostTrait + ?Sized,
    {
        Self::for_host_with_capacity(host, DEFAULT_EVENT_CAPACITY)
    }

    /// [`for_host`](Self::for_host) with an explicit queue capacity.
    ///
    /// A capacity of zero is treated as one, since a watcher that can hold no events could
    /// never report anything.
    ///
    /// # Errors
    ///
    /// See [`for_host`](Self::for_host).
    pub fn for_host_with_capacity<H>(host: &H, capacity: usize) -> Result<Self, Error>
    where
        H: HostTrait + ?Sized,
    {
        let mut known = Vec::new();
        collect_device_ids(host, &mut known)?;
        sort_and_dedup_ids(&mut known);
        let capacity = capacity.max(1);
        Ok(Self {
            scope: Scope::Host,
            queue: Vec::with_capacity(capacity),
            capacity,
            dropped: 0,
            known,
            scratch: Vec::new(),
            known_input_default: default_device_id(host.default_input_device()),
            known_output_default: default_device_id(host.default_output_device()),
        })
    }

    /// Watches `device` within `host`: whether it is still present, and default-route changes in
    /// the directions it supports.
    ///
    /// `device` is only read here (its id and supported directions); every later check goes
    /// through `host`. That keeps the watcher correct for a device handle that is a wrapper over
    /// the backend's device, as the platform-dispatched `Device` is.
    ///
    /// # Errors
    ///
    /// - [`ErrorKind::DeviceNotAvailable`] (or another error from [`DeviceTrait::id`]) if the
    ///   device cannot be identified, for example because it is already gone.
    /// - Any error from [`HostTrait::devices`] while checking whether the device is currently
    ///   present.
    pub fn for_device<H, D>(host: &H, device: &D) -> Result<Self, Error>
    where
        H: HostTrait + ?Sized,
        D: DeviceTrait + ?Sized,
    {
        Self::for_device_with_capacity(host, device, DEFAULT_EVENT_CAPACITY)
    }

    /// [`for_device`](Self::for_device) with an explicit queue capacity.
    ///
    /// A capacity of zero is treated as one.
    ///
    /// # Errors
    ///
    /// See [`for_device`](Self::for_device).
    pub fn for_device_with_capacity<H, D>(
        host: &H,
        device: &D,
        capacity: usize,
    ) -> Result<Self, Error>
    where
        H: HostTrait + ?Sized,
        D: DeviceTrait + ?Sized,
    {
        let id = device.id()?;
        Self::for_identified_device(
            host,
            id,
            device.supports_input(),
            device.supports_output(),
            capacity,
        )
    }

    /// Watches a device that is already identified by `id`, without needing a handle to it.
    ///
    /// The id is supplied here, so only the presence check can fail.
    fn for_identified_device<H>(
        host: &H,
        id: DeviceId,
        watch_input: bool,
        watch_output: bool,
        capacity: usize,
    ) -> Result<Self, Error>
    where
        H: HostTrait + ?Sized,
    {
        // Presence is checked through the host, by id, so the device argument is not needed
        // again.
        let present = host
            .devices()?
            .any(|other| other.id().ok().as_ref() == Some(&id));
        let capacity = capacity.max(1);

        Ok(Self {
            scope: Scope::Device {
                id,
                watch_input,
                watch_output,
                present,
            },
            queue: Vec::with_capacity(capacity),
            capacity,
            dropped: 0,
            known: Vec::new(),
            scratch: Vec::new(),
            known_input_default: if watch_input {
                default_device_id(host.default_input_device())
            } else {
                None
            },
            known_output_default: if watch_output {
                default_device_id(host.default_output_device())
            } else {
                None
            },
        })
    }

    /// The maximum number of undelivered events held before the oldest is dropped.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The number of undelivered events currently held.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether no events are waiting to be delivered.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// The total number of events dropped because the queue was full.
    ///
    /// Events are dropped oldest-first, so the queue keeps the most recent changes.
    pub fn dropped_events(&self) -> u64 {
        self.dropped
    }

    /// Discards undelivered events.
    ///
    /// The remembered device set and default routes are kept, so they are not reported again.
    pub fn clear(&mut self) {
        self.queue.clear();
    }

    /// Checks the host for changes and returns every undelivered event, in order.
    ///
    /// Events stay queued until [`drain`](Self::drain) or [`clear`](Self::clear), so polling
    /// twice without draining returns the same events again.
    ///
    /// # Errors
    ///
    /// Propagates [`HostTrait::devices`] errors and the
    /// unidentifiable-host error described on [`for_host`](Self::for_host). Undelivered events
    /// are kept and the remembered state is not updated, so polling again after the host
    /// recovers reports the changes that happened meanwhile.
    pub fn poll<H>(&mut self, host: &H) -> Result<&[DeviceEvent], Error>
    where
        H: HostTrait + ?Sized,
    {
        self.refresh(host)?;
        Ok(&self.queue)
    }

    /// Checks the host for changes and drains them in order.
    ///
    /// Events are moved out without copying.
    ///
    /// # Errors
    ///
    /// See [`poll`](Self::poll).
    pub fn drain<H>(&mut self, host: &H) -> Result<std::vec::Drain<'_, DeviceEvent>, Error>
    where
        H: HostTrait + ?Sized,
    {
        self.refresh(host)?;
        Ok(self.queue.drain(..))
    }

    fn refresh<H>(&mut self, host: &H) -> Result<(), Error>
    where
        H: HostTrait + ?Sized,
    {
        match self.scope {
            Scope::Host => self.refresh_host(host),
            Scope::Device { .. } => self.refresh_device(host),
        }
    }

    fn refresh_host<H>(&mut self, host: &H) -> Result<(), Error>
    where
        H: HostTrait + ?Sized,
    {
        // Take both buffers so the diff below can call `push` without borrowing `self`.
        let old = std::mem::take(&mut self.known);
        let mut new = std::mem::take(&mut self.scratch);
        new.clear();

        if let Err(err) = collect_device_ids(host, &mut new) {
            self.known = old;
            self.scratch = new;
            return Err(err);
        }
        sort_and_dedup_ids(&mut new);

        // Removals first, in ascending order. Both lists are sorted, so a single merge pass
        // yields the order without allocating sets.
        let mut old_i = 0;
        let mut new_i = 0;
        while old_i < old.len() {
            match new.get(new_i) {
                Some(new_id) => match cmp_device_ids(&old[old_i], new_id) {
                    Ordering::Less => {
                        let id = old[old_i].clone();
                        old_i += 1;
                        self.push(DeviceEventKind::Removed, Some(id));
                    }
                    Ordering::Equal => {
                        old_i += 1;
                        new_i += 1;
                    }
                    Ordering::Greater => new_i += 1,
                },
                None => {
                    let id = old[old_i].clone();
                    old_i += 1;
                    self.push(DeviceEventKind::Removed, Some(id));
                }
            }
        }

        // Additions, in ascending order.
        let mut old_i = 0;
        let mut new_i = 0;
        while new_i < new.len() {
            match old.get(old_i) {
                Some(old_id) => match cmp_device_ids(old_id, &new[new_i]) {
                    Ordering::Less => old_i += 1,
                    Ordering::Equal => {
                        old_i += 1;
                        new_i += 1;
                    }
                    Ordering::Greater => {
                        let id = new[new_i].clone();
                        new_i += 1;
                        self.push(DeviceEventKind::Added, Some(id));
                    }
                },
                None => {
                    let id = new[new_i].clone();
                    new_i += 1;
                    self.push(DeviceEventKind::Added, Some(id));
                }
            }
        }

        // Default routes, last. Query them after the device set: a default that changed because
        // a device was removed points at the replacement, which the events above already name.
        let new_input_default = default_device_id(host.default_input_device());
        let new_output_default = default_device_id(host.default_output_device());
        if new_input_default != self.known_input_default {
            self.push(
                DeviceEventKind::DefaultInputChanged,
                new_input_default.clone(),
            );
        }
        if new_output_default != self.known_output_default {
            self.push(
                DeviceEventKind::DefaultOutputChanged,
                new_output_default.clone(),
            );
        }
        self.known_input_default = new_input_default;
        self.known_output_default = new_output_default;

        // Commit, and keep the old buffer for reuse.
        self.known = new;
        self.scratch = old;
        Ok(())
    }

    fn refresh_device<H>(&mut self, host: &H) -> Result<(), Error>
    where
        H: HostTrait + ?Sized,
    {
        let Scope::Device {
            ref id,
            watch_input,
            watch_output,
            ref mut present,
        } = self.scope
        else {
            return Ok(());
        };
        let id = id.clone();

        let present_now = host
            .devices()?
            .any(|other| other.id().ok().as_ref() == Some(&id));
        if present_now != *present {
            let kind = if present_now {
                DeviceEventKind::Added
            } else {
                DeviceEventKind::Removed
            };
            *present = present_now;
            self.push(kind, Some(id.clone()));
        }

        if watch_input {
            let new_default = default_device_id(host.default_input_device());
            if new_default != self.known_input_default {
                self.push(DeviceEventKind::DefaultInputChanged, new_default.clone());
            }
            self.known_input_default = new_default;
        }
        if watch_output {
            let new_default = default_device_id(host.default_output_device());
            if new_default != self.known_output_default {
                self.push(DeviceEventKind::DefaultOutputChanged, new_default.clone());
            }
            self.known_output_default = new_default;
        }
        Ok(())
    }

    fn push(&mut self, kind: DeviceEventKind, device: Option<DeviceId>) {
        // Coalesce a repeated signal for the same change. Backends may report one change more
        // than once; identical consecutive events carry no additional information.
        if let Some(last) = self.queue.last()
            && last.kind == kind
            && last.device == device
        {
            return;
        }
        if self.queue.len() >= self.capacity {
            self.queue.remove(0);
            self.dropped = self.dropped.saturating_add(1);
        }
        self.queue.push(DeviceEvent { kind, device });
    }
}

/// Enumerates `host` into `out`, returning an error if the host reports devices but none of them
/// can be identified.
fn collect_device_ids<H>(host: &H, out: &mut Vec<DeviceId>) -> Result<(), Error>
where
    H: HostTrait + ?Sized,
{
    let mut enumerated = 0usize;
    for device in host.devices()? {
        enumerated += 1;
        if let Ok(id) = device.id() {
            out.push(id);
        }
    }
    if enumerated > 0 && out.is_empty() {
        return Err(Error::with_message(
            ErrorKind::UnsupportedOperation,
            "the host reports devices but cannot identify any of them, \
             so device events cannot be tracked",
        ));
    }
    Ok(())
}

/// Compares ids by host name, then backend identifier.
///
/// This is deliberately independent of the backend's enumeration order, and compares string
/// contents rather than hashes, so the order is the same on every run.
fn cmp_device_ids(a: &DeviceId, b: &DeviceId) -> Ordering {
    a.host()
        .name()
        .cmp(b.host().name())
        .then_with(|| a.id().cmp(b.id()))
}

fn sort_and_dedup_ids(ids: &mut Vec<DeviceId>) {
    // Unstable sort: the comparison is a total order over the full id, so equal ids are
    // interchangeable, and this avoids the sort's merge buffer allocation.
    ids.sort_unstable_by(cmp_device_ids);
    ids.dedup_by(|a, b| a == b);
}

/// The id of a default device, if there is one and it can be identified.
///
/// A default that cannot be identified is treated as no default: reporting a change back and
/// forth between a known and an unknown id would be noise, and the device set already reports
/// the hotplug that caused it.
fn default_device_id<D>(device: Option<D>) -> Option<DeviceId>
where
    D: DeviceTrait,
{
    device.and_then(|device| device.id().ok())
}
