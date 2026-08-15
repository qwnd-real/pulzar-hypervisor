//! What every controller on the machine is doing, on demand.
//!
//! What is worth having here is the state that says something is stuck: a
//! controller with interrupts requested and never taken, or one still owing real
//! hardware an acknowledgement, is the shape both an interrupt storm and a lost
//! wakeup show up as.

use log::info;

use crate::machine::registry::lapics;

/// Logs what each processor's controller is doing.
///
/// What is worth having here is the state that says something is stuck: a
/// controller with interrupts requested and never taken, or one still owing
/// real hardware an acknowledgement, is the shape both an interrupt storm and a
/// lost wakeup show up as.
pub fn describe(who: &str) {
    let Ok(page) = lapics() else {
        info!("{who}: the emulated controllers have not been installed");
        return;
    };
    for vlapic in page.all() {
        info!(
            "{who}: {} {} {} in {}{}, task priority {}, {} requested, {} in service{}",
            vlapic.index(),
            vlapic.apic_id(),
            if vlapic.running() {
                "running"
            } else {
                "waiting to be started"
            },
            vlapic.mode(),
            if vlapic.base().bootstrap() {
                " as the bootstrap processor"
            } else {
                ""
            },
            vlapic.task_priority(),
            vlapic.requested_count(),
            vlapic.in_service_count(),
            if vlapic.ledger().is_empty() {
                ""
            } else {
                ", owing hardware an acknowledgement"
            },
        );
        if let Some(vector) = vlapic.requested() {
            info!(
                "{who}: {} has {vector} requested at processor priority {}",
                vlapic.index(),
                vlapic.processor_priority()
            );
        }
    }
}
