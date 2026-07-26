//! The tables the processor delivers a guest's interrupts through without the
//! hypervisor's help.
//!
//! Interrupts are the worst case for virtualization: a guest that takes ten
//! thousand a second and must exit for each one spends most of its time not
//! running. So the hardware can be given the guest's interrupt controller
//! outright — its registers live in a page of memory, and the processor
//! delivers to them, evaluates priorities, and even routes interrupts between
//! the guest's own processors, all without an exit.
//!
//! Three tables make that work, and this module is their layout. One page each,
//! all of them physically addressed by the processor.
//!
//! # Why the hypervisor is still involved
//!
//! Delivery only stays in hardware while every destination is currently running
//! on some physical processor. The moment one is not — descheduled, or never
//! started — the hardware cannot do anything with the interrupt and exits, and
//! the hypervisor delivers it the slow way. That is what
//! [`IpiFailure::TargetNotRunning`] is, and it is the common case rather than
//! an error: a guest with more virtual processors than the machine has physical
//! ones is in it constantly.
//!
//! The other reason for an exit is a register the hardware does not implement.
//! Only the parts of the controller worth accelerating are accelerated; the
//! rest come back as an unaccelerated access for the hypervisor to emulate.

use x86_64::PhysAddr;

/// Bits the page-aligned pointers in these structures are shifted by.
const PAGE_SHIFT: u64 = 12;

/// One virtual processor's entry in the table of where its interrupt controller
/// lives.
///
/// The guest's own identifier for a processor indexes this table, and the entry
/// says where that processor's controller registers are and whether it is
/// running anywhere at the moment.
///
/// The running flag is the interesting one, and it is a promise the hypervisor
/// makes to the hardware: while it is set, the hardware will send interrupts
/// straight to the physical processor named beside it. A stale set flag sends
/// them to a processor that is running something else entirely.
#[bitfield_struct::bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct PhysicalApicEntry {
    /// Which physical processor is currently running this virtual one.
    /// Meaningless unless [`PhysicalApicEntry::is_running`] is set.
    #[bits(12)]
    pub host_apic_id: u16,
    /// Where this virtual processor's controller registers live, as a physical
    /// page number. Reached through
    /// [`PhysicalApicEntry::backing_page_address`] rather than directly.
    #[bits(40)]
    pub backing_page: u64,
    #[bits(10)]
    __: u16,
    /// This virtual processor is scheduled on a physical one right now, so the
    /// hardware may deliver to it directly.
    pub is_running: bool,
    /// This entry describes a virtual processor at all. A clear bit means the
    /// index names nothing.
    pub valid: bool,
}

impl PhysicalApicEntry {
    /// Where this virtual processor's controller registers are.
    #[must_use]
    pub const fn backing_page_address(self) -> PhysAddr {
        PhysAddr::new(self.backing_page() << PAGE_SHIFT)
    }

    /// The same entry pointing at this page of controller registers.
    ///
    /// The address must be page aligned; its low twelve bits are not part of
    /// the field and are discarded.
    #[must_use]
    pub const fn with_backing_page_address(self, address: PhysAddr) -> Self {
        self.with_backing_page(address.as_u64() >> PAGE_SHIFT)
    }
}

/// One entry of the table translating a logical interrupt destination into a
/// virtual processor.
///
/// Interrupts can be addressed to a logical identifier rather than to a
/// specific processor, and this table is how the hardware resolves one. Because
/// each entry names exactly one virtual processor, every logical identifier in
/// a guest must be unique — a scheme where several processors share one is not
/// something this table can express.
#[bitfield_struct::bitfield(u32)]
#[derive(PartialEq, Eq)]
pub struct LogicalApicEntry {
    /// Which virtual processor this logical identifier names, in the guest's
    /// own numbering.
    pub guest_apic_id: u8,
    #[bits(23)]
    __: u32,
    /// This entry names a processor at all.
    pub valid: bool,
}

/// Where the table of virtual processors is, and how much of it is in use.
///
/// The one pointer in a guest's control block whose low twelve bits are not
/// reserved: they hold the highest index that is valid, which is what lets the
/// hardware avoid walking a whole page of mostly empty entries. Every other
/// pointer to these structures requires those bits to be zero.
#[bitfield_struct::bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct AvicPhysicalTable {
    /// Highest index in the table that describes a processor.
    #[bits(12)]
    pub max_index: u16,
    /// Where the table is, as a physical page number.
    #[bits(40)]
    pub pointer: u64,
    #[bits(12)]
    __: u16,
}

impl AvicPhysicalTable {
    /// Where the table is.
    #[must_use]
    pub const fn address(self) -> PhysAddr {
        PhysAddr::new(self.pointer() << PAGE_SHIFT)
    }

    /// The same value pointing at this page. The address must be page aligned;
    /// its low twelve bits are discarded.
    #[must_use]
    pub const fn with_address(self, address: PhysAddr) -> Self {
        self.with_pointer(address.as_u64() >> PAGE_SHIFT)
    }
}

/// Highest virtual processor index addressable with eight-bit identifiers.
///
/// One less than the field could hold, because the all-ones identifier means
/// "every processor" and so cannot name one.
pub const MAX_PHYSICAL_ID: u16 = 0xFE;

/// Highest virtual processor index addressable with 32-bit identifiers.
pub const X2_MAX_PHYSICAL_ID: u16 = 0x1FF;

/// Highest virtual processor index addressable with 32-bit identifiers on a
/// processor that supports the extended table.
///
/// Beyond one page's worth: such a table spans up to eight consecutive pages,
/// and how many is the top three bits of the maximum index plus one.
pub const X2_EXTENDED_MAX_PHYSICAL_ID: u16 = 0xFFF;

/// Why the hardware could not finish delivering an interrupt between a guest's
/// processors.
///
/// Only the second of these is routine. The rest mean the guest programmed its
/// controller in a way the hardware does not accelerate, or that the tables the
/// hypervisor maintains do not describe what the guest is trying to reach.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum IpiFailure {
    /// The guest asked for a delivery mode the hardware does not accelerate.
    /// The hypervisor emulates the request itself.
    InvalidInterruptType = 0,
    /// A destination is not scheduled on a physical processor at the moment,
    /// so there is nowhere to send it. Routine on any machine running more
    /// virtual processors than it has physical ones: the hypervisor makes the
    /// interrupt pending and delivers it when that processor next runs.
    TargetNotRunning = 1,
    /// The destination names no valid entry in the table of virtual
    /// processors.
    InvalidTarget = 2,
    /// The destination's entry is valid but points at no usable page of
    /// controller registers.
    InvalidBackingPage = 3,
    /// The vector the guest asked to deliver is not one the hardware will
    /// deliver.
    InvalidIpiVector = 4,
}

impl IpiFailure {
    /// The cause an encoding names, or `None` for one the architecture does not
    /// define.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Option<Self> {
        Some(match bits {
            0 => Self::InvalidInterruptType,
            1 => Self::TargetNotRunning,
            2 => Self::InvalidTarget,
            3 => Self::InvalidBackingPage,
            4 => Self::InvalidIpiVector,
            _ => return None,
        })
    }

    /// The encoding for this cause.
    #[must_use]
    pub const fn into_bits(self) -> u32 {
        self as u32
    }
}

/// Signals another physical processor that one of its guests has an interrupt
/// waiting.
///
/// Written when the hardware has put an interrupt into a guest's controller
/// registers but that guest is running elsewhere: the write pokes the physical
/// processor running it, which re-examines the guest's controller and delivers
/// what it finds. Write-only — reading it faults — and unlike most writes to a
/// model-specific register, this one is not fully serializing, precisely
/// because it sits on a path that has to be fast.
#[bitfield_struct::bitfield(u64)]
#[derive(PartialEq, Eq)]
pub struct Doorbell {
    /// Which physical processor to poke.
    #[bits(12)]
    pub host_apic_id: u16,
    #[bits(52)]
    __: u64,
}

const _: () = assert!(
    size_of::<PhysicalApicEntry>() == size_of::<u64>()
        && size_of::<AvicPhysicalTable>() == size_of::<u64>()
        && size_of::<Doorbell>() == size_of::<u64>(),
    "these are quadword structures",
);
const _: () = assert!(
    size_of::<LogicalApicEntry>() == size_of::<u32>(),
    "a logical table entry is one doubleword",
);
const _: () = assert!(
    MAX_PHYSICAL_ID < 0xFF,
    "the broadcast identifier cannot name one processor",
);
