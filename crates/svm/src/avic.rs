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
//! Three tables make that work, and this module is their layout. All of them
//! are physically addressed by the processor, and all but one are a single
//! page: the table of virtual processors is a run of up to eight, because the
//! field beside its address gives the hardware a largest valid index rather
//! than a length.
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
    /// The guest sent an interrupt the encrypted-virtualization extension
    /// refuses to deliver in hardware. Only a processor with that extension
    /// reports it; like [`IpiFailure::InvalidInterruptType`], the answer is
    /// for the hypervisor to emulate the request itself.
    UnacceleratedIpi = 5,
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
            5 => Self::UnacceleratedIpi,
            _ => return None,
        })
    }

    /// The encoding for this cause.
    #[must_use]
    pub const fn into_bits(self) -> u32 {
        self as u32
    }
}

/// What an `AVIC_INCOMPLETE_IPI` exit is saying.
///
/// The interrupt the guest asked for, decoded whole out of the two
/// exit-information fields: the request itself as the controller's command
/// register would have taken it, why the hardware could not deliver it — as the
/// cause a handler acts on and as the encoding the hardware actually named —
/// and the destination the tables were consulted for where the cause is one
/// about a destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IncompleteIpiExit {
    icr: u64,
    cause: IpiFailure,
    reported: u32,
    index: u16,
}

impl IncompleteIpiExit {
    /// The exit decoded out of the two exit-information fields the processor
    /// writes into the control block.
    ///
    /// Total rather than fallible: a cause encoding the architecture does not
    /// define decodes as [`IpiFailure::InvalidInterruptType`], because full
    /// software emulation is the one answer that is safe for a request
    /// nothing else understands.
    ///
    /// The encoding is kept beside the cause it decoded to, because that
    /// substitution is safe for a handler and wrong for anything counting: see
    /// [`IncompleteIpiExit::reported`].
    #[must_use]
    pub const fn from_exit_info(exit_info_1: u64, exit_info_2: u64) -> Self {
        let reported = (exit_info_2 >> u32::BITS) as u32;
        let cause = match IpiFailure::from_bits(reported) {
            Some(cause) => cause,
            None => IpiFailure::InvalidInterruptType,
        };
        Self {
            icr: exit_info_1,
            cause,
            reported,
            index: (exit_info_2 & INDEX_MASK) as u16,
        }
    }

    /// The request the guest made, as the controller's command register would
    /// have taken it.
    #[must_use]
    pub const fn icr(&self) -> u64 {
        self.icr
    }

    /// Why the hardware could not deliver it.
    #[must_use]
    pub const fn cause(&self) -> IpiFailure {
        self.cause
    }

    /// The encoding the hardware named the failure by, whether or not the
    /// architecture defines one.
    ///
    /// [`IncompleteIpiExit::cause`] answers what a *handler* should do, and an
    /// undefined encoding is deliberately answered there as the cause whose
    /// treatment is safe for a request nothing understands. That substitution
    /// is wrong for anything that counts or reports: an encoding above the
    /// defined range is silicon describing something this hypervisor does
    /// not model, and a log in which it is indistinguishable from the
    /// commonest legitimate cause hides the one machine worth knowing
    /// about.
    #[must_use]
    pub const fn reported(&self) -> u32 {
        self.reported
    }

    /// The destination the tables were consulted for. Only the causes about a
    /// destination give this meaning.
    ///
    /// An index into the physical table or into the logical one, and which of
    /// them is not this field's to say: the cause decides for two of the three
    /// it is defined for, and for a directed interprocessor interrupt it is the
    /// destination mode of the command in [`IncompleteIpiExit::icr`] — logical
    /// where that names a logical destination, physical otherwise. Reported raw
    /// here for that reason; a consumer that means to say which table has both
    /// halves to hand.
    #[must_use]
    pub const fn index(&self) -> u16 {
        self.index
    }
}

/// What an `AVIC_UNACCELERATED_ACCESS` exit is saying.
///
/// One access to one controller register the hardware does not accelerate:
/// its direction, the register's offset in the controller's page, and — only
/// for the one write where it is part of the request — the vector being
/// retired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnacceleratedAccessExit {
    write: bool,
    offset: u16,
    eoi_vector: Option<u8>,
}

impl UnacceleratedAccessExit {
    /// The exit decoded out of the two exit-information fields the processor
    /// writes into the control block.
    ///
    /// The vector names the interrupt an end-of-interrupt write retires, and
    /// is present only for that one access: for every other register the
    /// second field is not the guest's and is not decoded.
    #[must_use]
    pub const fn from_exit_info(exit_info_1: u64, exit_info_2: u64) -> Self {
        let write = exit_info_1 & ACCESS_IS_WRITE != 0;
        let offset = (exit_info_1 & OFFSET_MASK) as u16;
        let eoi_vector = if write && offset == EOI_OFFSET {
            Some((exit_info_2 & VECTOR_MASK) as u8)
        } else {
            None
        };
        Self {
            write,
            offset,
            eoi_vector,
        }
    }

    /// Whether the guest wrote the register rather than read it.
    #[must_use]
    pub const fn is_write(&self) -> bool {
        self.write
    }

    /// The register's offset in the controller's page.
    #[must_use]
    pub const fn offset(&self) -> u16 {
        self.offset
    }

    /// The vector an end-of-interrupt write retires, where that is what this
    /// was.
    #[must_use]
    pub const fn eoi_vector(&self) -> Option<u8> {
        self.eoi_vector
    }
}

/// Bits of the second exit-information field of an incomplete delivery that
/// name a destination.
const INDEX_MASK: u64 = 0xFFF;

/// The bit of the first exit-information field of an unaccelerated access that
/// says the access was a write.
const ACCESS_IS_WRITE: u64 = 1 << 32;

/// Bits of the first exit-information field of an unaccelerated access that
/// name a register offset.
const OFFSET_MASK: u64 = 0xFF0;

/// The end-of-interrupt register's offset in the controller's page.
const EOI_OFFSET: u16 = 0xB0;

/// The bits of the second exit-information field of an unaccelerated access
/// that an end-of-interrupt write carries its retired vector in.
const VECTOR_MASK: u64 = 0xFF;

/// The index of the register that rings [`Doorbell`].
///
/// It sits in the range of model-specific registers a hypervisor owns, and a
/// guest reaching it could poke whichever physical processor it named — so
/// every guest's permission map intercepts it, and the access is refused
/// rather than forwarded.
pub const AVIC_DOORBELL: u32 = 0xC001_011B;

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
/// Both 32-bit caps have to survive the twelve-bit field a control block
/// publishes a table's extent in, or a mode's own limit would be a number
/// nothing could tell the hardware — and raising one past the field is an edit
/// that would otherwise compile and only be found where the extent is written.
const _: () = assert!(
    AvicPhysicalTable::new()
        .with_max_index(X2_MAX_PHYSICAL_ID)
        .max_index()
        == X2_MAX_PHYSICAL_ID
        && AvicPhysicalTable::new()
            .with_max_index(X2_EXTENDED_MAX_PHYSICAL_ID)
            .max_index()
            == X2_EXTENDED_MAX_PHYSICAL_ID,
    "a mode's highest index must survive the field a table's extent is published in",
);

#[cfg(test)]
mod tests {
    use super::*;

    /// A second exit-information field carrying this cause and this index.
    fn exit_info_2(cause: IpiFailure, index: u64) -> u64 {
        (u64::from(cause.into_bits()) << u32::BITS) | index
    }

    #[test]
    fn ipi_failure_encodings_round_trip() {
        for cause in [
            IpiFailure::InvalidInterruptType,
            IpiFailure::TargetNotRunning,
            IpiFailure::InvalidTarget,
            IpiFailure::InvalidBackingPage,
            IpiFailure::InvalidIpiVector,
            IpiFailure::UnacceleratedIpi,
        ] {
            assert_eq!(IpiFailure::from_bits(cause.into_bits()), Some(cause));
        }
    }

    #[test]
    fn ipi_failure_refuses_undefined_encodings() {
        assert_eq!(IpiFailure::from_bits(6), None);
        assert_eq!(IpiFailure::from_bits(u32::MAX), None);
    }

    #[test]
    fn an_incomplete_ipi_exit_carries_the_request_cause_and_index() {
        let icr = 0x0000_0000_000C_4A20;
        let exit = IncompleteIpiExit::from_exit_info(
            icr,
            exit_info_2(IpiFailure::TargetNotRunning, 0x1FE),
        );
        assert_eq!(exit.icr(), icr);
        assert_eq!(exit.cause(), IpiFailure::TargetNotRunning);
        assert_eq!(exit.reported(), IpiFailure::TargetNotRunning.into_bits());
        assert_eq!(exit.index(), 0x1FE);
    }

    #[test]
    fn an_incomplete_ipi_exit_masks_the_index_to_twelve_bits() {
        let exit = IncompleteIpiExit::from_exit_info(
            0,
            exit_info_2(IpiFailure::InvalidTarget, 0xFFFF_FFFF),
        );
        assert_eq!(exit.index(), 0xFFF);
    }

    #[test]
    fn an_unknown_cause_decodes_as_full_emulation_and_keeps_its_encoding() {
        // The substitution is what a handler needs — emulating the request whole
        // is the one answer that is safe for something nothing here models — and
        // it is exactly what a count of it must not inherit, or the encoding
        // arrives in the log as the commonest legitimate cause. So both are
        // carried, and this is what says they are two answers rather than one.
        for reported in [6, 7, u32::MAX] {
            let exit =
                IncompleteIpiExit::from_exit_info(0, u64::from(reported) << u32::BITS | 0x1FE);
            assert_eq!(exit.cause(), IpiFailure::InvalidInterruptType);
            assert_eq!(exit.reported(), reported);
            assert_eq!(IpiFailure::from_bits(exit.reported()), None);
            assert_eq!(exit.index(), 0x1FE, "the index is decoded either way");
        }
        // And a defined encoding is the cause itself, so nothing reads the two
        // apart where the architecture named the failure.
        for cause in [
            IpiFailure::InvalidInterruptType,
            IpiFailure::UnacceleratedIpi,
        ] {
            let exit = IncompleteIpiExit::from_exit_info(0, exit_info_2(cause, 0));
            assert_eq!(exit.cause(), cause);
            assert_eq!(IpiFailure::from_bits(exit.reported()), Some(cause));
        }
    }

    #[test]
    fn an_unaccelerated_access_names_its_register_and_direction() {
        let read = UnacceleratedAccessExit::from_exit_info(0x830, 0);
        assert!(!read.is_write());
        assert_eq!(read.offset(), 0x830);
        assert_eq!(read.eoi_vector(), None);

        let write = UnacceleratedAccessExit::from_exit_info(ACCESS_IS_WRITE | 0x280, 0);
        assert!(write.is_write());
        assert_eq!(write.offset(), 0x280);
    }

    #[test]
    fn an_unaccelerated_access_ignores_bits_that_are_not_the_offset() {
        let exit = UnacceleratedAccessExit::from_exit_info(ACCESS_IS_WRITE | 0xFFF_FFFF, 0);
        assert_eq!(exit.offset(), 0xFF0);
    }

    #[test]
    fn an_unaccelerated_access_names_the_vector_only_for_an_eoi_write() {
        let eoi =
            UnacceleratedAccessExit::from_exit_info(ACCESS_IS_WRITE | u64::from(EOI_OFFSET), 0x2E);
        assert_eq!(eoi.eoi_vector(), Some(0x2E));

        let eoi_read = UnacceleratedAccessExit::from_exit_info(u64::from(EOI_OFFSET), 0x2E);
        assert_eq!(eoi_read.eoi_vector(), None);

        let other_write = UnacceleratedAccessExit::from_exit_info(ACCESS_IS_WRITE | 0x300, 0x2E);
        assert_eq!(other_write.eoi_vector(), None);
    }

    #[test]
    fn an_unaccelerated_access_masks_the_vector_to_eight_bits() {
        let eoi =
            UnacceleratedAccessExit::from_exit_info(ACCESS_IS_WRITE | u64::from(EOI_OFFSET), 0x1FF);
        assert_eq!(eoi.eoi_vector(), Some(0xFF));
    }

    /// A physical table entry as the architecture lays it out: the valid and
    /// running bits in the two highest positions, the backing page's number in
    /// the forty between, and the host's identifier in the lowest twelve
    /// (APM Table 15-23).
    #[test]
    fn a_physical_entry_encodes_where_the_architecture_says() {
        let entry = PhysicalApicEntry::new()
            .with_valid(true)
            .with_is_running(true)
            .with_backing_page_address(PhysAddr::new(0x0000_0ABC_D000))
            .with_host_apic_id(0x123);
        assert_eq!(
            entry.into_bits(),
            (1 << 63) | (1 << 62) | 0x0000_0ABC_D000 | 0x123,
        );
    }

    /// The backing page enters as its number and leaves as its address, and
    /// the low bits an entry has no room for are dropped on the way in.
    #[test]
    fn a_physical_entry_keeps_the_backing_page_page_aligned() {
        let entry = PhysicalApicEntry::new().with_backing_page_address(PhysAddr::new(0x0001_2345));
        assert_eq!(entry.backing_page_address(), PhysAddr::new(0x0001_2000));
    }

    /// A logical table entry: the guest identifier in the lowest byte and the
    /// valid bit in the highest, with nothing between them (APM Table 15-24).
    #[test]
    fn a_logical_entry_encodes_where_the_architecture_says() {
        let entry = LogicalApicEntry::new()
            .with_valid(true)
            .with_guest_apic_id(0x5A);
        assert_eq!(entry.into_bits(), (1 << 31) | 0x5A);
    }

    /// The control-block value naming the physical table: the table's number
    /// above the largest valid index beside it, with the twelve bits between
    /// them the only place the index fits.
    #[test]
    fn a_physical_table_pointer_keeps_the_index_beside_it() {
        let table = AvicPhysicalTable::new()
            .with_max_index(0x1FF)
            .with_address(PhysAddr::new(0xDEAD_E000));
        assert_eq!(table.into_bits(), 0xDEAD_E000 | 0x1FF);
        assert_eq!(table.address(), PhysAddr::new(0xDEAD_E000));
        assert_eq!(table.max_index(), 0x1FF);
    }

    /// The doorbell: twelve bits of physical identifier at the bottom of the
    /// register and nothing else in it.
    ///
    /// The one field in this module whose width is a judgement rather than a
    /// transcription — the architecture's figure draws eight and leaves the
    /// rest must-be-zero — so what it encodes is worth pinning: a narrower
    /// field would poke the wrong processor on a machine with more than 256
    /// of them, and a wider one would set a bit the architecture reserves.
    #[test]
    fn a_doorbell_carries_the_physical_identifier_and_nothing_else() {
        assert_eq!(Doorbell::new().with_host_apic_id(0xFFF).into_bits(), 0xFFF);
        assert_eq!(Doorbell::new().with_host_apic_id(1).into_bits(), 1);
        assert_eq!(Doorbell::new().into_bits(), 0);
    }
}
