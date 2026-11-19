//! Watches for device hotplug and default-device changes.
//!
//! This is the fix for an application that keeps a [`cpal::Device`] handle open and has to be
//! restarted after headphones are plugged in or the system default changes: instead of holding
//! the stale handle, subscribe to device events and rebuild the device and stream when the
//! event says so.
//!
//! Two layers are shown:
//! - a host watcher, which reports every device appearing or disappearing and both defaults
//!   changing;
//! - a device watcher, which reports whether one device is still present.
//!
//! Run with: `cargo run --example device_events`
//!
//! Plug in or unplug a headset, or change the default output in the system settings, to see
//! events. Stop with Ctrl-C.

extern crate anyhow;
extern crate cpal;

use std::time::{Duration, Instant};

use cpal::{
    DeviceEvent, DeviceEventKind,
    traits::{DeviceTrait, HostTrait},
};

fn main() -> Result<(), anyhow::Error> {
    let host = cpal::default_host();
    println!("host: {:?}", host.id());

    // Seeded with the current state: this only reports what changes from here on.
    let mut host_events = host.device_events()?;
    println!(
        "watching {} device(s) for {:.0?}",
        host.devices()?.count(),
        Duration::from_secs(20)
    );

    // A device watcher, so the same example shows the second layer. It is optional: the host
    // watcher already reports this device disappearing.
    let mut device_events = match host.default_output_device() {
        Some(device) => {
            println!("also watching the default output device ({device})");
            Some(device.device_events(&host)?)
        }
        None => {
            println!("no default output device to watch");
            None
        }
    };

    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        for event in host_events.drain(&host)? {
            describe("host", &event);
        }
        if let Some(watcher) = device_events.as_mut() {
            for event in watcher.drain(&host)? {
                describe("device", &event);
            }
        }
        std::thread::sleep(Duration::from_millis(250));
    }

    println!("done");
    Ok(())
}

fn describe(scope: &str, event: &DeviceEvent) {
    let device = event
        .device()
        .map_or_else(|| "<none>".to_owned(), |id| id.to_string());
    match event.kind() {
        DeviceEventKind::Added => println!("[{scope}] device plugged in: {device}"),
        DeviceEventKind::Removed => println!("[{scope}] device unplugged: {device}"),
        DeviceEventKind::DefaultInputChanged => {
            println!("[{scope}] default input is now: {device}")
        }
        DeviceEventKind::DefaultOutputChanged => {
            println!("[{scope}] default output is now: {device}")
        }
        // `DeviceEventKind` is `#[non_exhaustive]`; a newer CPAL may report something else.
        _ => println!("[{scope}] {event}"),
    }
}
