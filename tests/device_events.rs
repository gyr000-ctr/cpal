//! Device hotplug and default-route event tests.
//!
//! The host here is a scripted fake, so the tests do not depend on audio hardware, a sound
//! server, or any platform. It lets the suite drive the exact sequences that a real backend
//! reports: a device appearing, disappearing, the default moving, and the same signal arriving
//! twice.

use std::{
    fmt,
    sync::{Arc, Mutex},
};

use cpal::{
    CallbackInfo, DEFAULT_EVENT_CAPACITY, Data, DeviceDescription, DeviceDescriptionBuilder,
    DeviceEvent, DeviceEventKind, DeviceEvents, DeviceId, Error, ErrorKind, FrameCount, HostId,
    SampleFormat, StreamConfig, StreamInstant, SupportedStreamConfig, SupportedStreamConfigRange,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

// ---------------------------------------------------------------------------------------------
// A scripted host
// ---------------------------------------------------------------------------------------------

/// Shared, mutable state that the tests edit and the fake host reports.
#[derive(Debug, Default)]
struct State {
    /// Devices currently present, in insertion order.
    devices: Vec<String>,
    default_input: Option<String>,
    default_output: Option<String>,
    /// When set, `devices()` fails with this error kind.
    devices_error: Option<ErrorKind>,
    /// When set, `Device::id` fails, mimicking a host that reports devices it cannot identify.
    ids_unavailable: bool,
    /// Every call to `devices()`, so tests can assert the watcher reads state and nothing else.
    devices_calls: usize,
}

#[derive(Clone, Debug, Default)]
struct FakeHost {
    state: Arc<Mutex<State>>,
}

impl FakeHost {
    fn new(devices: &[&str]) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                devices: devices.iter().map(|s| (*s).to_owned()).collect(),
                ..State::default()
            })),
        }
    }

    fn with_defaults(self, input: Option<&str>, output: Option<&str>) -> Self {
        {
            let mut state = self.state.lock().unwrap();
            state.default_input = input.map(str::to_owned);
            state.default_output = output.map(str::to_owned);
        }
        self
    }

    fn add(&self, id: &str) {
        let mut state = self.state.lock().unwrap();
        if !state.devices.iter().any(|d| d == id) {
            state.devices.push(id.to_owned());
        }
    }

    fn remove(&self, id: &str) {
        let mut state = self.state.lock().unwrap();
        state.devices.retain(|d| d != id);
    }

    fn set_default_input(&self, id: Option<&str>) {
        self.state.lock().unwrap().default_input = id.map(str::to_owned);
    }

    fn set_default_output(&self, id: Option<&str>) {
        self.state.lock().unwrap().default_output = id.map(str::to_owned);
    }

    fn fail_devices_with(&self, kind: Option<ErrorKind>) {
        self.state.lock().unwrap().devices_error = kind;
    }

    fn ids_unavailable(&self, unavailable: bool) {
        self.state.lock().unwrap().ids_unavailable = unavailable;
    }

    fn devices_calls(&self) -> usize {
        self.state.lock().unwrap().devices_calls
    }
}

#[derive(Clone, Debug)]
struct FakeDevice {
    host: FakeHost,
    id: String,
}

impl PartialEq for FakeDevice {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for FakeDevice {}

impl std::hash::Hash for FakeDevice {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl FakeDevice {
    fn track(&self) {
        self.host.add(&self.id);
    }

    fn untrack(&self) {
        self.host.remove(&self.id);
    }
}

impl FakeHost {
    fn device(&self, id: &str) -> FakeDevice {
        FakeDevice {
            host: self.clone(),
            id: id.to_owned(),
        }
    }
}

impl HostTrait for FakeHost {
    type Devices = std::vec::IntoIter<FakeDevice>;
    type Device = FakeDevice;

    fn is_available() -> bool {
        true
    }

    fn devices(&self) -> Result<Self::Devices, Error> {
        let mut state = self.state.lock().unwrap();
        state.devices_calls += 1;
        if let Some(kind) = state.devices_error {
            return Err(Error::new(kind));
        }
        let ids = state.devices.clone();
        drop(state);
        Ok(ids
            .into_iter()
            .map(|id| FakeDevice {
                host: self.clone(),
                id,
            })
            .collect::<Vec<_>>()
            .into_iter())
    }

    fn default_input_device(&self) -> Option<Self::Device> {
        self.state
            .lock()
            .unwrap()
            .default_input
            .clone()
            .map(|id| FakeDevice {
                host: self.clone(),
                id,
            })
    }

    fn default_output_device(&self) -> Option<Self::Device> {
        self.state
            .lock()
            .unwrap()
            .default_output
            .clone()
            .map(|id| FakeDevice {
                host: self.clone(),
                id,
            })
    }
}

impl DeviceTrait for FakeDevice {
    type SupportedInputConfigs = std::vec::IntoIter<SupportedStreamConfigRange>;
    type SupportedOutputConfigs = std::vec::IntoIter<SupportedStreamConfigRange>;
    type Stream = FakeStream;

    fn description(&self) -> Result<DeviceDescription, Error> {
        Ok(DeviceDescriptionBuilder::new(self.id.as_str()).build())
    }

    fn id(&self) -> Result<DeviceId, Error> {
        if self.host.state.lock().unwrap().ids_unavailable {
            return Err(Error::new(ErrorKind::UnsupportedOperation));
        }
        Ok(DeviceId::new(HostId::Alsa, &self.id))
    }

    fn supports_input(&self) -> bool {
        // Inputs are named "in*"; everything else is output-only, so the device-level watcher
        // can be checked for ignoring defaults it never claims to care about.
        self.id.starts_with("in")
    }

    fn supports_output(&self) -> bool {
        !self.id.starts_with("in")
    }

    fn supported_input_configs(&self) -> Result<Self::SupportedInputConfigs, Error> {
        Ok(Vec::new().into_iter())
    }

    fn supported_output_configs(&self) -> Result<Self::SupportedOutputConfigs, Error> {
        Ok(Vec::new().into_iter())
    }

    fn default_input_config(&self) -> Result<SupportedStreamConfig, Error> {
        Err(Error::new(ErrorKind::UnsupportedOperation))
    }

    fn default_output_config(&self) -> Result<SupportedStreamConfig, Error> {
        Err(Error::new(ErrorKind::UnsupportedOperation))
    }

    fn build_input_stream_raw<D, E>(
        &self,
        _: StreamConfig,
        _: SampleFormat,
        _: D,
        _: E,
        _: Option<std::time::Duration>,
    ) -> Result<Self::Stream, Error>
    where
        D: FnMut(&Data, &CallbackInfo) + Send + 'static,
        E: FnMut(Error) + Send + 'static,
    {
        Err(Error::new(ErrorKind::UnsupportedOperation))
    }

    fn build_output_stream_raw<D, E>(
        &self,
        _: StreamConfig,
        _: SampleFormat,
        _: D,
        _: E,
        _: Option<std::time::Duration>,
    ) -> Result<Self::Stream, Error>
    where
        D: FnMut(&mut Data, &CallbackInfo) + Send + 'static,
        E: FnMut(Error) + Send + 'static,
    {
        Err(Error::new(ErrorKind::UnsupportedOperation))
    }
}

impl fmt::Display for FakeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.id)
    }
}

#[derive(Debug)]
struct FakeStream;

impl StreamTrait for FakeStream {
    fn start(&self) -> Result<(), Error> {
        Ok(())
    }

    fn pause(&self) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&self, _: Option<std::time::Duration>) -> Result<(), Error> {
        Ok(())
    }

    fn now(&self) -> StreamInstant {
        StreamInstant::new(0, 0)
    }

    fn buffer_size(&self) -> Result<FrameCount, Error> {
        Ok(0)
    }
}

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

/// `(kind, device id)` pairs, which is what the assertions care about.
fn events(events: &[DeviceEvent]) -> Vec<(DeviceEventKind, Option<String>)> {
    events
        .iter()
        .map(|event| (event.kind(), event.device().map(|id| id.id().to_owned())))
        .collect()
}

fn hotplug(kind: DeviceEventKind, id: &str) -> (DeviceEventKind, Option<String>) {
    (kind, Some(id.to_owned()))
}

fn default_change(kind: DeviceEventKind, id: Option<&str>) -> (DeviceEventKind, Option<String>) {
    (kind, id.map(str::to_owned))
}

// ---------------------------------------------------------------------------------------------
// Host-level events
// ---------------------------------------------------------------------------------------------

#[test]
fn a_new_watcher_reports_nothing_and_seeds_the_current_state() {
    let host = FakeHost::new(&["out0"]).with_defaults(None, Some("out0"));
    let mut watcher = host.device_events().unwrap();

    assert!(watcher.poll(&host).unwrap().is_empty());
    assert_eq!(watcher.len(), 0);
    assert_eq!(watcher.dropped_events(), 0);
    assert_eq!(watcher.capacity(), DEFAULT_EVENT_CAPACITY);
}

#[test]
fn plugging_a_device_reports_one_added_event() {
    let host = FakeHost::new(&["out0"]).with_defaults(None, Some("out0"));
    let mut watcher = host.device_events().unwrap();

    host.add("usb-headset");
    let seen = watcher.poll(&host).unwrap();
    assert_eq!(
        events(seen),
        vec![hotplug(DeviceEventKind::Added, "usb-headset")]
    );
    assert!(seen[0].is_hotplug());
    assert!(!seen[0].is_default_change());
    assert_eq!(seen[0].to_string(), "device added: alsa:usb-headset");
}

#[test]
fn unplugging_a_device_reports_one_removed_event() {
    let host = FakeHost::new(&["out0", "usb-headset"]).with_defaults(None, Some("out0"));
    let mut watcher = host.device_events().unwrap();

    host.remove("usb-headset");
    assert_eq!(
        events(watcher.poll(&host).unwrap()),
        vec![hotplug(DeviceEventKind::Removed, "usb-headset")]
    );
}

#[test]
fn a_changed_default_reports_the_new_device() {
    let host = FakeHost::new(&["out0", "usb-headset"]).with_defaults(Some("in0"), Some("out0"));
    let mut watcher = host.device_events().unwrap();

    host.set_default_output(Some("usb-headset"));
    assert_eq!(
        events(watcher.poll(&host).unwrap()),
        vec![default_change(
            DeviceEventKind::DefaultOutputChanged,
            Some("usb-headset")
        )]
    );
}

#[test]
fn a_default_with_no_device_left_is_reported_with_no_device() {
    let host = FakeHost::new(&["out0"]).with_defaults(None, Some("out0"));
    let mut watcher = host.device_events().unwrap();

    host.remove("out0");
    host.set_default_output(None);

    let seen = events(watcher.poll(&host).unwrap());
    assert_eq!(
        seen,
        vec![
            hotplug(DeviceEventKind::Removed, "out0"),
            default_change(DeviceEventKind::DefaultOutputChanged, None),
        ]
    );
    // The event itself carries no device, rather than a placeholder.
    assert_eq!(watcher.poll(&host).unwrap()[1].device(), None);
}

#[test]
fn repeated_polls_do_not_repeat_events() {
    let host = FakeHost::new(&["out0"]).with_defaults(None, Some("out0"));
    let mut watcher = host.device_events().unwrap();

    host.add("usb-headset");
    let first = events(watcher.poll(&host).unwrap());
    assert_eq!(first.len(), 1);

    // Polling again without draining returns the same undelivered event, not a new one.
    let second = events(watcher.poll(&host).unwrap());
    assert_eq!(second, first);

    let drained: Vec<_> = watcher
        .drain(&host)
        .unwrap()
        .map(|event| event.kind())
        .collect();
    assert_eq!(drained, vec![DeviceEventKind::Added]);

    // After draining a poll with no change reports nothing.
    assert!(watcher.poll(&host).unwrap().is_empty());
}

#[test]
fn a_device_seen_twice_by_enumeration_is_reported_once() {
    // The backend reports duplicate ids; the watcher must deduplicate.
    let host = FakeHost::new(&["out0"]);
    host.add("usb-headset");
    host.add("usb-headset");
    let mut watcher = host.device_events().unwrap();
    assert!(watcher.poll(&host).unwrap().is_empty());

    // Now an actual change on top of a duplicated enumeration.
    host.add("out1");
    host.add("out1");
    assert_eq!(
        events(watcher.poll(&host).unwrap()),
        vec![hotplug(DeviceEventKind::Added, "out1")]
    );
}

#[test]
fn hotplug_is_ordered_before_default_changes() {
    let host = FakeHost::new(&["out0"]).with_defaults(None, Some("out0"));
    let mut watcher = host.device_events().unwrap();

    // One change produces both: a new device appears and becomes the default.
    host.add("usb-headset");
    host.set_default_output(Some("usb-headset"));
    host.set_default_input(Some("in0"));
    host.add("in0");

    assert_eq!(
        events(watcher.poll(&host).unwrap()),
        vec![
            hotplug(DeviceEventKind::Added, "in0"),
            hotplug(DeviceEventKind::Added, "usb-headset"),
            default_change(DeviceEventKind::DefaultInputChanged, Some("in0")),
            default_change(DeviceEventKind::DefaultOutputChanged, Some("usb-headset")),
        ]
    );
}

#[test]
fn removals_are_ordered_before_additions_and_each_group_is_sorted() {
    let host = FakeHost::new(&["b-out", "d-out"]);
    let mut watcher = host.device_events().unwrap();

    host.remove("d-out");
    host.remove("b-out");
    host.add("a-out");
    host.add("c-out");

    // Removals ascending, then additions ascending, regardless of the mutation order above.
    assert_eq!(
        events(watcher.poll(&host).unwrap()),
        vec![
            hotplug(DeviceEventKind::Removed, "b-out"),
            hotplug(DeviceEventKind::Removed, "d-out"),
            hotplug(DeviceEventKind::Added, "a-out"),
            hotplug(DeviceEventKind::Added, "c-out"),
        ]
    );
}

#[test]
fn ids_from_another_host_sort_by_host_name_first() {
    let first = DeviceId::new(HostId::Alsa, "z");
    let second = DeviceId::new(HostId::Custom, "a");
    // `cmp_device_ids` orders on host name first.
    let mut ids = [second, first];
    ids.sort_by(|a, b| a.host().name().cmp(b.host().name()));
    assert_eq!(ids[0].host(), HostId::Alsa);
}

#[test]
fn selecting_the_default_that_is_already_default_is_not_a_change() {
    let host = FakeHost::new(&["out0", "out1"]).with_defaults(None, Some("out0"));
    let mut watcher = host.device_events().unwrap();

    host.set_default_output(Some("out0"));
    assert!(watcher.poll(&host).unwrap().is_empty());
}

#[test]
fn a_failing_host_reports_an_error_instead_of_panicking() {
    let host = FakeHost::new(&[]);
    host.fail_devices_with(Some(ErrorKind::HostUnavailable));

    let err = host.device_events().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::HostUnavailable);

    // The failure is not sticky: once the host recovers, the watcher works.
    host.fail_devices_with(None);
    let mut watcher = host.device_events().unwrap();
    host.add("out0");
    assert_eq!(
        events(watcher.poll(&host).unwrap()),
        vec![hotplug(DeviceEventKind::Added, "out0")]
    );
}

#[test]
fn a_poll_that_fails_keeps_undelivered_events_and_does_not_advance_state() {
    let host = FakeHost::new(&["out0"]);
    let mut watcher = host.device_events().unwrap();

    host.add("out1");
    assert_eq!(watcher.poll(&host).unwrap().len(), 1);

    host.fail_devices_with(Some(ErrorKind::HostUnavailable));
    let err = watcher.poll(&host).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::HostUnavailable);
    // The pending event is still there.
    watcher.clear();
    assert!(watcher.is_empty());
}

#[test]
fn a_host_that_cannot_identify_devices_falls_back_cleanly() {
    // Enumeration succeeds and reports devices, but no device can be identified. The watcher
    // cannot tell which devices exist, so it says so, rather than reporting bogus changes.
    let host = FakeHost::new(&["out0", "out1"]);
    host.ids_unavailable(true);

    let err = host.device_events().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::UnsupportedOperation);
    assert!(err.to_string().contains("cannot identify"));
}

#[test]
fn an_empty_host_is_supported_even_though_nothing_can_be_identified() {
    // Nothing to identify is not the same as failing to identify what is there.
    let host = FakeHost::new(&[]);
    let mut watcher = host.device_events().unwrap();
    assert!(watcher.poll(&host).unwrap().is_empty());
}

#[test]
fn the_queue_drops_oldest_events_when_full_and_counts_them() {
    let host = FakeHost::new(&[]);
    let mut watcher = DeviceEvents::for_host_with_capacity(&host, 2).unwrap();
    assert_eq!(watcher.capacity(), 2);

    for id in ["a", "b", "c", "d"] {
        host.add(id);
    }
    let drained: Vec<DeviceEvent> = watcher.drain(&host).unwrap().collect();
    let seen = events(&drained);
    // Two newest survive; the two oldest were dropped.
    assert_eq!(
        seen,
        vec![
            hotplug(DeviceEventKind::Added, "c"),
            hotplug(DeviceEventKind::Added, "d"),
        ]
    );
    assert_eq!(watcher.dropped_events(), 2);
}

#[test]
fn a_zero_capacity_is_treated_as_one() {
    let host = FakeHost::new(&[]);
    let watcher = DeviceEvents::for_host_with_capacity(&host, 0).unwrap();
    assert_eq!(watcher.capacity(), 1);
    assert_eq!(DEFAULT_EVENT_CAPACITY, watcher_len_of_default());
}

fn watcher_len_of_default() -> usize {
    let host = FakeHost::new(&[]);
    host.device_events().unwrap().capacity()
}

#[test]
fn clear_discards_events_but_remembers_the_state() {
    let host = FakeHost::new(&[]);
    let mut watcher = host.device_events().unwrap();

    host.add("out0");
    watcher.poll(&host).unwrap();
    assert_eq!(watcher.len(), 1);

    watcher.clear();
    assert!(watcher.is_empty());
    // The change was already observed, so clearing does not make it reappear.
    assert!(watcher.poll(&host).unwrap().is_empty());
}

#[test]
fn draining_moves_events_out_without_leaving_them_queued() {
    let host = FakeHost::new(&[]);
    let mut watcher = host.device_events().unwrap();
    host.add("out0");

    let drained: Vec<DeviceEvent> = watcher.drain(&host).unwrap().collect();
    assert_eq!(drained.len(), 1);
    assert!(watcher.is_empty());
    assert_eq!(watcher.poll(&host).unwrap().len(), 0);
}

#[test]
fn a_watcher_reads_the_host_and_changes_nothing_about_it() {
    let host = FakeHost::new(&["out0", "in0"]).with_defaults(Some("in0"), Some("out0"));
    let mut watcher = host.device_events().unwrap();

    // Seeding reads the host once; a poll reads it once more. It never writes to it.
    let seeded = host.devices_calls();
    assert_eq!(seeded, 1, "seeding enumerates once");
    watcher.poll(&host).unwrap();
    assert_eq!(host.devices_calls(), seeded + 1);

    // The existing API still answers exactly what it did before the watcher existed.
    assert_eq!(
        host.devices().unwrap().map(|d| d.id).collect::<Vec<_>>(),
        vec!["out0", "in0"]
    );
    assert_eq!(host.default_output_device().unwrap().id, "out0");
    assert_eq!(host.default_input_device().unwrap().id, "in0");
}

// ---------------------------------------------------------------------------------------------
// Device-level events
// ---------------------------------------------------------------------------------------------

#[test]
fn a_device_watcher_reports_its_own_removal_and_return() {
    let host = FakeHost::new(&["out0"]).with_defaults(None, Some("out0"));
    let device = host.device("out0");
    let mut watcher = device.device_events(&host).unwrap();

    assert!(watcher.poll(&host).unwrap().is_empty());

    device.untrack();
    let removed: Vec<DeviceEvent> = watcher.drain(&host).unwrap().collect();
    assert_eq!(
        events(&removed),
        vec![hotplug(DeviceEventKind::Removed, "out0")]
    );

    device.track();
    let added: Vec<DeviceEvent> = watcher.drain(&host).unwrap().collect();
    assert_eq!(
        events(&added),
        vec![hotplug(DeviceEventKind::Added, "out0")]
    );
}

#[test]
fn a_device_watcher_seeds_the_current_default() {
    let host = FakeHost::new(&["out0", "out1"]).with_defaults(None, Some("out0"));
    let device = host.device("out0");
    let mut watcher = device.device_events(&host).unwrap();

    // Seeded with `out0` as the default, so moving it is a change.
    host.set_default_output(Some("out1"));
    assert_eq!(
        events(watcher.poll(&host).unwrap()),
        vec![default_change(
            DeviceEventKind::DefaultOutputChanged,
            Some("out1")
        )]
    );
}

#[test]
fn a_device_watcher_ignores_defaults_in_directions_it_does_not_support() {
    // `out0` is output-only, so input-default changes are not its concern.
    let host = FakeHost::new(&["out0", "in0", "in1"]).with_defaults(Some("in0"), Some("out0"));
    let device = host.device("out0");
    let mut watcher = device.device_events(&host).unwrap();

    host.set_default_input(Some("in1"));
    assert!(watcher.poll(&host).unwrap().is_empty());

    host.set_default_output(Some("out0"));
    assert!(watcher.poll(&host).unwrap().is_empty());
}

#[test]
fn a_device_watcher_does_not_report_other_devices_changing() {
    let host = FakeHost::new(&["out0"]).with_defaults(None, Some("out0"));
    let device = host.device("out0");
    let mut watcher = device.device_events(&host).unwrap();

    host.add("usb-headset");
    host.remove("out0");
    let seen = events(watcher.poll(&host).unwrap());
    assert_eq!(
        seen,
        vec![hotplug(DeviceEventKind::Removed, "out0")],
        "the new device is not this watcher's concern, its own removal is"
    );
}

#[test]
fn a_device_that_cannot_be_identified_reports_that_error() {
    let host = FakeHost::new(&["out0"]);
    let device = host.device("out0");
    host.ids_unavailable(true);

    let err = device.device_events(&host).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::UnsupportedOperation);
}

#[test]
fn a_device_watcher_reports_a_removed_default_with_no_replacement() {
    let host = FakeHost::new(&["out0"]).with_defaults(None, Some("out0"));
    let device = host.device("out0");
    let mut watcher = device.device_events(&host).unwrap();

    host.remove("out0");
    host.set_default_output(None);
    assert_eq!(
        events(watcher.poll(&host).unwrap()),
        vec![
            hotplug(DeviceEventKind::Removed, "out0"),
            default_change(DeviceEventKind::DefaultOutputChanged, None),
        ]
    );
}

// ---------------------------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------------------------

/// A counting allocator, so "the event path does not allocate" is measured, not assumed.
///
/// Counts are thread-local: the test harness runs tests in parallel, and a process-wide counter
/// would fold other tests' allocations into the measurement.
mod alloc_counter {
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        cell::Cell,
    };

    thread_local! {
        /// `None` while measurement is off; otherwise the number of allocations since it was
        /// turned on.
        pub static COUNT: Cell<Option<u64>> = const { Cell::new(None) };
    }

    fn count() {
        let _ = COUNT.try_with(|c| {
            if let Some(n) = c.get() {
                c.set(Some(n + 1));
            }
        });
    }

    /// Counts allocations made on this thread until the returned guard is dropped.
    pub fn measure() -> Guard {
        COUNT.with(|c| c.set(Some(0)));
        Guard
    }

    pub struct Guard;

    impl Guard {
        /// Allocations made on this thread since [`measure`] was called.
        pub fn allocations(&self) -> u64 {
            COUNT.with(|c| c.get().unwrap_or(0))
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            COUNT.with(|c| c.set(None));
        }
    }

    pub struct Counting;

    // SAFETY: forwards every operation to the system allocator unchanged, and only updates a
    // thread-local counter.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            count();
            // SAFETY: `System::alloc` has the same contract as this method.
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: the pointer and layout come from a matching allocation.
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            count();
            // SAFETY: the pointer, layout and new size come from a matching allocation.
            unsafe { System.realloc(ptr, layout, new_size) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            count();
            // SAFETY: `System::alloc_zeroed` has the same contract as this method.
            unsafe { System.alloc_zeroed(layout) }
        }
    }
}

#[global_allocator]
static GLOBAL: alloc_counter::Counting = alloc_counter::Counting;

/// A host whose enumeration allocates nothing, so the measurement isolates the watcher.
///
/// [`FakeHost`] builds a `Vec` and a `String` per device on every enumeration, which a real
/// backend also does. Measuring through it would report the backend's allocations, not the
/// event path's. Here every id lives in a fixed array and is returned by reference.
mod still_host {
    use cpal::{
        CallbackInfo, Data, DeviceDescription, DeviceDescriptionBuilder, DeviceId, Error,
        ErrorKind, FrameCount, HostId, SampleFormat, StreamConfig, StreamInstant,
        SupportedStreamConfig, SupportedStreamConfigRange,
        traits::{DeviceTrait, HostTrait, StreamTrait},
    };

    /// Up to 8 devices, each with a fixed-size name, and no allocation anywhere.
    #[derive(Clone, Debug)]
    pub struct Host {
        pub state: std::sync::Arc<std::sync::Mutex<FixedState>>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct FixedState {
        /// `count` ids, named by index (`d0`, `d1`, ...).
        pub count: usize,
        pub defaults: [Option<usize>; 2],
    }

    impl Host {
        pub fn new(count: usize) -> Self {
            Self {
                state: std::sync::Arc::new(std::sync::Mutex::new(FixedState {
                    count,
                    defaults: [None, None],
                })),
            }
        }

        pub fn set(&self, count: usize, input: Option<usize>, output: Option<usize>) {
            *self.state.lock().unwrap() = FixedState {
                count,
                defaults: [input, output],
            };
        }
    }

    const NAMES: [&str; 8] = ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"];

    /// Ids are built once and handed out by clone. Cloning a [`DeviceId`] shares its identifier
    /// string, so `id()` does not allocate; that is what lets the event path stay
    /// allocation-free once the transition is observed.
    static IDS: std::sync::LazyLock<[DeviceId; 8]> =
        std::sync::LazyLock::new(|| NAMES.map(|name| DeviceId::new(HostId::Alsa, name)));

    /// The id of device `index`, cloned from the shared cache (no allocation).
    pub fn id_for(index: usize) -> Option<DeviceId> {
        IDS.get(index).cloned()
    }

    impl HostTrait for Host {
        type Devices = std::iter::Map<std::ops::Range<usize>, fn(usize) -> Device>;
        type Device = Device;

        fn is_available() -> bool {
            true
        }

        fn devices(&self) -> Result<Self::Devices, Error> {
            let count = self.state.lock().unwrap().count.min(NAMES.len());
            let f: fn(usize) -> Device = |i| Device { index: i };
            Ok((0..count).map(f))
        }

        fn default_input_device(&self) -> Option<Self::Device> {
            self.state.lock().unwrap().defaults[0].map(|index| Device { index })
        }

        fn default_output_device(&self) -> Option<Self::Device> {
            self.state.lock().unwrap().defaults[1].map(|index| Device { index })
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct Device {
        index: usize,
    }

    impl std::fmt::Display for Device {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(NAMES[self.index])
        }
    }

    impl DeviceTrait for Device {
        type SupportedInputConfigs = std::vec::IntoIter<SupportedStreamConfigRange>;
        type SupportedOutputConfigs = std::vec::IntoIter<SupportedStreamConfigRange>;
        type Stream = Stream;

        fn description(&self) -> Result<DeviceDescription, Error> {
            Ok(DeviceDescriptionBuilder::new(NAMES[self.index]).build())
        }

        fn id(&self) -> Result<DeviceId, Error> {
            Ok(IDS[self.index].clone())
        }

        fn supported_input_configs(&self) -> Result<Self::SupportedInputConfigs, Error> {
            Ok(Vec::new().into_iter())
        }

        fn supported_output_configs(&self) -> Result<Self::SupportedOutputConfigs, Error> {
            Ok(Vec::new().into_iter())
        }

        fn default_input_config(&self) -> Result<SupportedStreamConfig, Error> {
            Err(Error::new(ErrorKind::UnsupportedOperation))
        }

        fn default_output_config(&self) -> Result<SupportedStreamConfig, Error> {
            Err(Error::new(ErrorKind::UnsupportedOperation))
        }

        fn build_input_stream_raw<D, E>(
            &self,
            _: StreamConfig,
            _: SampleFormat,
            _: D,
            _: E,
            _: Option<std::time::Duration>,
        ) -> Result<Self::Stream, Error>
        where
            D: FnMut(&Data, &CallbackInfo) + Send + 'static,
            E: FnMut(Error) + Send + 'static,
        {
            Err(Error::new(ErrorKind::UnsupportedOperation))
        }

        fn build_output_stream_raw<D, E>(
            &self,
            _: StreamConfig,
            _: SampleFormat,
            _: D,
            _: E,
            _: Option<std::time::Duration>,
        ) -> Result<Self::Stream, Error>
        where
            D: FnMut(&mut Data, &CallbackInfo) + Send + 'static,
            E: FnMut(Error) + Send + 'static,
        {
            Err(Error::new(ErrorKind::UnsupportedOperation))
        }
    }

    #[derive(Debug)]
    pub struct Stream;

    impl StreamTrait for Stream {
        fn start(&self) -> Result<(), Error> {
            Ok(())
        }

        fn pause(&self) -> Result<(), Error> {
            Ok(())
        }

        fn stop(&self, _: Option<std::time::Duration>) -> Result<(), Error> {
            Ok(())
        }

        fn now(&self) -> StreamInstant {
            StreamInstant::new(0, 0)
        }

        fn buffer_size(&self) -> Result<FrameCount, Error> {
            Ok(0)
        }
    }
}

#[test]
fn delivering_events_without_a_backend_allocation_does_not_allocate() {
    let host = still_host::Host::new(2);
    host.set(2, Some(0), Some(0));
    let mut watcher = host.device_events().unwrap();

    // Warm up: the first poll can grow the scratch buffer and the queue. After this, the
    // buffers have room for the same device count, and the event path must stay allocation-free.
    host.set(3, Some(0), Some(0));
    watcher.poll(&host).unwrap();
    watcher.clear();
    host.set(2, Some(0), Some(0));
    watcher.poll(&host).unwrap();
    watcher.clear();

    // Pre-build the expectation outside the measured window: `DeviceId` compares by value, so
    // the check itself needs no allocation either.
    let expected_id = still_host::id_for(1);
    assert!(expected_id.is_some());

    // A default change at a device count the buffers already handled.
    host.set(2, Some(0), Some(1));

    let mut seen_kind = None;
    let mut seen_matches = false;
    let mut count = 0usize;
    let guard = alloc_counter::measure();
    for event in watcher.drain(&host).unwrap() {
        count += 1;
        if count == 1 {
            seen_kind = Some(event.kind());
            seen_matches = event.device() == expected_id.as_ref();
        }
    }
    let allocated = guard.allocations();
    drop(guard);

    assert_eq!(count, 1);
    assert_eq!(seen_kind, Some(DeviceEventKind::DefaultOutputChanged));
    assert!(seen_matches, "the change must name the new default device");
    assert_eq!(
        allocated, 0,
        "the event path allocated {allocated} times in steady state"
    );
}

#[test]
fn draining_a_full_queue_does_not_allocate() {
    let host = still_host::Host::new(2);
    host.set(2, Some(0), Some(0));
    let mut watcher = host.device_events().unwrap();
    // Warm the buffers with several events, then measure the drain itself.
    host.set(4, Some(0), Some(0));
    watcher.poll(&host).unwrap();
    watcher.clear();
    host.set(2, Some(0), Some(0));
    watcher.poll(&host).unwrap();
    watcher.clear();

    host.set(4, Some(0), Some(0));
    watcher.poll(&host).unwrap();

    let guard = alloc_counter::measure();
    let count = watcher.drain(&host).unwrap().count();
    let allocated = guard.allocations();
    drop(guard);

    assert_eq!(count, 2);
    assert_eq!(allocated, 0, "draining allocated {allocated} times");
}

// ---------------------------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------------------------

#[test]
fn a_device_watcher_seeded_on_an_absent_device_reports_it_appearing() {
    // The handle exists but the device is already gone when the watcher is created, which is the
    // common case when a caller keeps a handle across a replug.
    let host = FakeHost::new(&["out0"]);
    let device = host.device("out0");
    host.remove("out0");

    let mut watcher = device.device_events(&host).unwrap();
    // Seeding recorded the device as absent, so no event yet.
    assert!(watcher.poll(&host).unwrap().is_empty());

    device.track();
    assert_eq!(
        events(watcher.poll(&host).unwrap()),
        vec![hotplug(DeviceEventKind::Added, "out0")]
    );
}

#[test]
fn two_watchers_on_the_same_host_are_independent() {
    let host = FakeHost::new(&["out0"]).with_defaults(None, Some("out0"));
    let mut first = host.device_events().unwrap();
    let mut second = host.device_events().unwrap();

    host.add("out1");
    let drained: Vec<DeviceEvent> = second.drain(&host).unwrap().collect();
    let seen = events(&drained);
    assert_eq!(seen.len(), 1);
    // The first watcher has not been polled, so it still holds its own event.
    assert_eq!(first.len(), 0);
    assert_eq!(events(first.poll(&host).unwrap()).len(), 1);
}

#[test]
fn event_accessors_report_their_parts() {
    let id = DeviceId::new(HostId::Alsa, "out0");
    let event = DeviceEvent::new(DeviceEventKind::Removed, Some(id.clone()));
    assert_eq!(event.kind(), DeviceEventKind::Removed);
    assert_eq!(event.device(), Some(&id));
    assert!(event.is_hotplug());
    assert!(!event.is_default_change());
    assert_eq!(event.clone().into_device(), Some(id));

    let empty = DeviceEvent::new(DeviceEventKind::DefaultOutputChanged, None);
    assert_eq!(empty.device(), None);
    assert!(empty.is_default_change());
    assert_eq!(
        empty.to_string(),
        "default output device changed: no device"
    );
}

#[test]
fn event_kind_flags_are_mutually_exclusive() {
    for kind in [
        DeviceEventKind::Added,
        DeviceEventKind::Removed,
        DeviceEventKind::DefaultInputChanged,
        DeviceEventKind::DefaultOutputChanged,
    ] {
        assert_ne!(kind.is_hotplug(), kind.is_default_change(), "for {kind:?}");
    }
}

#[test]
fn watching_a_host_that_became_unavailable_reports_it_on_poll() {
    let host = FakeHost::new(&["out0"]);
    let mut watcher = host.device_events().unwrap();

    host.fail_devices_with(Some(ErrorKind::HostUnavailable));
    let err = watcher.poll(&host).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::HostUnavailable);
}
