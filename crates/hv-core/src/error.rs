//! Why the hypervisor could not finish bringing itself up.
//!
//! Every variant ends the same way — the boot stops and the processor halts —
//! so the point of naming them is the serial log, which is the only thing left
//! to diagnose from.

use acpi::AcpiError;
use apic::ApicError;
use clock::ClockError;
use cpu::CpuError;
use descriptors::DescriptorError;
use emulate::EmulateError;
use exits::ExitError;
use handoff::HandoffError;
use ipi::IpiError;
use paging::PagingError;
use partition::PartitionError;
use pci::PciError;
use portal::PortalError;
use thiserror::Error;
use vcpu::VcpuError;
use vlapic::VlapicError;

/// A failure during bring-up.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum CoreError {
    /// The boot protocol the loader passed cannot be used.
    #[error(transparent)]
    Handoff(#[from] HandoffError),
    /// The address space could not be adopted or edited.
    #[error(transparent)]
    Paging(#[from] PagingError),
    /// The address space was told twice how to reach the other processors, or
    /// twice how to count them.
    #[error(transparent)]
    Shootdown(#[from] paging::shootdown::AlreadyInstalled),
    /// The nested page tables were told twice how to reach the processors
    /// inside a guest, or twice how to count them.
    #[error(transparent)]
    Coherence(#[from] npt::coherence::AlreadyInstalled),
    /// The firmware tables could not be read.
    #[error(transparent)]
    Acpi(#[from] AcpiError),
    /// The processor's own descriptor tables could not be set up.
    #[error(transparent)]
    Descriptors(#[from] DescriptorError),
    /// No timebase could be established, so the hypervisor would have no way to
    /// tell how much time had passed.
    #[error(transparent)]
    Clock(#[from] ClockError),
    /// The machine's processors could not be described, or this one could not
    /// take a place among them.
    #[error(transparent)]
    Cpu(#[from] CpuError),
    /// An interrupt controller could not be set up or driven, or a processor
    /// could not be started.
    #[error(transparent)]
    Apic(#[from] ApicError),
    /// Interprocessor interrupts could not be set up.
    #[error(transparent)]
    Ipi(#[from] IpiError),
    /// The machine's devices could not be surveyed.
    #[error(transparent)]
    Pci(#[from] PciError),
    /// The guest could not be established, or this processor could not join it.
    #[error(transparent)]
    Partition(#[from] PartitionError),
    /// The virtualization extension could not be enabled on this processor.
    #[error(transparent)]
    Vcpu(#[from] VcpuError),
    /// The guest firmware portal could not be placed.
    #[error(transparent)]
    Portal(#[from] PortalError),
    /// Instruction decoding could not be prepared before guest execution.
    #[error(transparent)]
    Emulate(#[from] EmulateError),
    /// The guest's own interrupt controllers could not be built or driven.
    #[error(transparent)]
    Vlapic(#[from] VlapicError),
    /// The guest stopped, at an exit nothing could answer for or at a control
    /// block the processor refused.
    #[error(transparent)]
    Exit(#[from] ExitError),
    /// A processor came up before the boot processor had established the guest,
    /// which the order of bring-up is supposed to rule out.
    #[error("the guest was not established before this processor came up")]
    NoPartition,
    /// The allocator would not take the span reserved for the heap, which can
    /// only mean it is too small to hold the allocator's own bookkeeping.
    #[error("the allocator refused the {bytes:#x}-byte heap at {base:#x}")]
    HeapRefused {
        /// Where the heap was to start.
        base: u64,
        /// How large it was to be.
        bytes: u64,
    },
    /// An address in the handoff is not one this processor can form.
    #[error("{value:#x} is not a usable address")]
    BadAddress {
        /// The offending value.
        value: u64,
    },
    /// Something the hypervisor depends on did not survive the transition to an
    /// address space of its own.
    #[error("{what} stopped working after the firmware half of the address space was dropped")]
    SelfCheckFailed {
        /// What was checked.
        what: &'static str,
    },
}
