//! What the emulator is allowed to touch, as narrow traits rather than as the
//! hypervisor's own types.
//!
//! Every module below this one works through [`Cpu`] and [`Guest`]. Neither
//! adds a layer for its own sake: they exist because the two things an emulator
//! acts on — a register file and a guest's memory — are reachable in production
//! only from a live virtual processor on a machine running a guest, and a great
//! deal of what this crate has to get exactly right is arithmetic that has
//! nothing to do with either.
//!
//! # What this buys, and what it costs
//!
//! It buys the whole of the test suite. Partial-register merge rules, address
//! wrapping at three address sizes, span planning across a page boundary, the
//! order of operations when a device read is followed by a failing store: all
//! of it is decided by code that now runs on the host against a register file
//! that is an array and a guest memory that is a map.
//!
//! It costs nothing at runtime. Every caller is generic over these traits and
//! monomorphizes to one implementation in the hypervisor image, so a call
//! through `Cpu` is the same direct call to [`Vcpu`] it was when the type was
//! named — there is no dynamic dispatch anywhere in this crate.
//!
//! # The vector registers are part of this, deliberately
//!
//! The guest's vector state is not in a control block; it is still in the
//! processor. That makes reading it the one operation here that is inherently
//! assembly, and putting it behind this trait is what confines that assembly to
//! a single implementation — so the emulator's own logic about *which* register
//! to read, and how much of it a move touches, is tested on the host without
//! executing a vector instruction at all.

use memory::{Addressing, MemoryError, Written};
use svm::SaveArea;
use x86_64::PhysAddr;

use crate::{
    value::{WIDEST, Width},
    xmm,
    xmm::Vector,
};

/// The register file an instruction is performed against.
///
/// Implemented by [`Vcpu`](vcpu::Vcpu) for the hypervisor and by a plain struct
/// for the tests. The methods are the architecture's own vocabulary — a
/// four-bit register number, the saved state, the vector registers — and not a
/// convenience layer over it: everything about *what* those numbers mean lives
/// in [`crate::gpr`], where it is one implementation rather than one per
/// backend.
pub(crate) trait Cpu {
    /// What the guest has in the register a four-bit number names.
    fn gpr(&self, number: u8) -> u64;

    /// Puts a value in the register a four-bit number names.
    ///
    /// The whole sixty-four bits. Partial-register rules are applied before
    /// this is reached, because they are the same rules for every backend.
    fn set_gpr(&mut self, number: u8, value: u64);

    /// The state the guest stopped in.
    fn save(&self) -> &SaveArea;

    /// The same, to write.
    fn save_mut(&mut self) -> &mut SaveArea;

    /// The instruction bytes the processor already fetched, if it left them.
    ///
    /// Empty where it did not, which is not a failure: it means the instruction
    /// has to be read back out of the guest, which is what [`crate::decode`]
    /// does next.
    fn fetched(&self) -> &[u8];

    /// What the guest has in one of its vector registers.
    fn vector(&self, register: Vector) -> [u8; WIDEST];

    /// Puts a value in one of the guest's vector registers.
    fn set_vector(&mut self, register: Vector, value: [u8; WIDEST]);
}

/// A guest's memory, at the addresses the guest itself uses.
///
/// Narrower than [`Linear`](memory::Linear) on purpose: this crate translates,
/// reads and writes, and does nothing else to a guest's memory. Anything the
/// trait does not offer is something the emulator has no business doing.
pub(crate) trait Guest {
    /// Where a linear address lands in the guest's physical memory.
    ///
    /// # Errors
    ///
    /// Whatever walking the guest's own tables reports.
    fn translate(&self, linear: u64) -> Result<PhysAddr, MemoryError>;

    /// Copies `into.len()` bytes of the guest's memory.
    ///
    /// # Errors
    ///
    /// As [`Guest::translate`], and whatever reaching the memory reports.
    fn read(&self, linear: u64, into: &mut [u8]) -> Result<(), MemoryError>;

    /// Copies `from.len()` bytes into the guest's memory.
    ///
    /// # Errors
    ///
    /// As [`Guest::read`].
    fn write(&self, linear: u64, from: &[u8]) -> Result<Written, MemoryError>;

    /// Whether a write of this width there would be committed rather than
    /// discarded.
    ///
    /// Asked before an irreversible read, so that a destination which would
    /// refuse the write is discovered while refusing still costs nothing. The
    /// answer is about the whole span, because a write is all-or-nothing: one
    /// unwritable byte discards the lot.
    ///
    /// # Errors
    ///
    /// As [`Guest::translate`].
    fn writable(&self, linear: u64, width: Width) -> Result<bool, MemoryError>;

    /// How this guest translates: its mode, its segment bases and its address
    /// widths.
    fn addressing(&self) -> &Addressing;
}

/// The hypervisor's own virtual processor, which is the only implementation
/// anything outside the tests ever sees.
impl Cpu for vcpu::Vcpu {
    fn gpr(&self, number: u8) -> u64 {
        Self::gpr(self, number)
    }

    fn set_gpr(&mut self, number: u8, value: u64) {
        Self::set_gpr(self, number, value);
    }

    fn save(&self) -> &SaveArea {
        Self::save(self)
    }

    fn save_mut(&mut self) -> &mut SaveArea {
        Self::save_mut(self)
    }

    fn fetched(&self) -> &[u8] {
        self.control().fetched_instruction()
    }

    /// Straight out of the processor, because that is where the guest's vector
    /// state still is — nothing saved it on the way out of the guest.
    fn vector(&self, register: Vector) -> [u8; WIDEST] {
        xmm::read(register)
    }

    /// Straight into the processor, which *is* the guest's register changing.
    fn set_vector(&mut self, register: Vector, value: [u8; WIDEST]) {
        xmm::write(register, value);
    }
}

impl Guest for memory::Linear<'_> {
    fn translate(&self, linear: u64) -> Result<PhysAddr, MemoryError> {
        Self::translate(self, linear)
    }

    fn read(&self, linear: u64, into: &mut [u8]) -> Result<(), MemoryError> {
        Self::read(self, linear, into)
    }

    fn write(&self, linear: u64, from: &[u8]) -> Result<Written, MemoryError> {
        Self::write(self, linear, from)
    }

    fn writable(&self, linear: u64, width: Width) -> Result<bool, MemoryError> {
        Self::writable(self, linear, width.bytes())
    }

    fn addressing(&self) -> &Addressing {
        Self::addressing(self)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use alloc::{collections::BTreeMap, vec, vec::Vec};
    use core::cell::{Cell, RefCell};

    use memory::{Addressing, MemoryError, Written};
    use svm::{SaveArea, SegmentAttributes};
    use x86_64::PhysAddr;

    use super::{Cpu, Guest};
    use crate::{
        value::{WIDEST, Width},
        xmm::Vector,
    };

    /// A register file that is an array, standing in for one that is a control
    /// block on a processor running a guest.
    ///
    /// The two registers a state-save area really holds are held there here
    /// too, because which side of that split a register falls on is exactly
    /// the sort of thing a merge rule can get wrong.
    #[derive(Clone, Debug)]
    pub(crate) struct Machine {
        registers: [u64; 16],
        save: SaveArea,
        vectors: [[u8; WIDEST]; 16],
        fetched: Vec<u8>,
    }

    /// Which encoded number the accumulator is, being one of the two the save
    /// area holds rather than the register block.
    const RAX: u8 = 0;
    /// Which encoded number the stack pointer is, being the other.
    const RSP: u8 = 4;
    /// Bits of an encoded register number that name a register, as the
    /// architecture reports them.
    const NUMBER: u8 = 0xF;

    impl Default for Machine {
        fn default() -> Self {
            Self {
                registers: [0; 16],
                save: SaveArea::zeroed(),
                vectors: [[0; WIDEST]; 16],
                fetched: Vec::new(),
            }
        }
    }

    impl Machine {
        /// A machine executing 64-bit code with paging on, which is what almost
        /// every test wants and none of them should have to spell.
        pub(crate) fn long_mode() -> Self {
            let mut machine = Self::default();
            machine.save.cs.attributes = SegmentAttributes::new().with_long(true);
            machine.save.cr0 = PG;
            machine.save.cr4 = PAE;
            machine.save.efer = LMA;
            machine
        }

        /// A machine executing 32-bit protected-mode code with paging on.
        pub(crate) fn protected() -> Self {
            let mut machine = Self::default();
            machine.save.cs.attributes = SegmentAttributes::new().with_default_size(true);
            machine.save.cr0 = PG;
            machine
        }

        /// A machine executing 16-bit code with no paging at all.
        pub(crate) fn real() -> Self {
            Self::default()
        }

        /// Where the guest is stopped.
        pub(crate) const fn at(mut self, rip: u64) -> Self {
            self.save.rip = rip;
            self
        }

        /// The same machine with the direction flag set, so the string
        /// instructions count downwards.
        pub(crate) const fn backwards(mut self) -> Self {
            self.save.rflags |= DIRECTION;
            self
        }

        /// The same machine with these bytes already fetched by the processor,
        /// as a control block carries them after a nested page fault.
        ///
        /// Named apart from [`Cpu::fetched`] deliberately: one puts bytes there
        /// and the other reads them back, and a builder that shadowed the trait
        /// method would make which of the two a call site meant depend on how
        /// many arguments it happened to pass.
        pub(crate) fn with_fetched(mut self, bytes: &[u8]) -> Self {
            self.fetched = bytes.to_vec();
            self
        }

        /// How this machine translates, which is what a guest memory is built
        /// from.
        pub(crate) fn addressing(&self) -> Addressing {
            Addressing::from_save(&self.save)
        }
    }

    /// Bit of `CR0` that turns the guest's own page tables on.
    const PG: u64 = 1 << 31;
    /// Bit of `CR4` that widens a page table entry and adds a level.
    const PAE: u64 = 1 << 5;
    /// Bit of `EFER` that says the processor really is in long mode.
    const LMA: u64 = 1 << 10;
    /// Bit of the flags that says the index registers count downwards.
    const DIRECTION: u64 = 1 << 10;

    impl Cpu for Machine {
        fn gpr(&self, number: u8) -> u64 {
            match number & NUMBER {
                RAX => self.save.rax,
                RSP => self.save.rsp,
                number => self.registers[usize::from(number)],
            }
        }

        fn set_gpr(&mut self, number: u8, value: u64) {
            match number & NUMBER {
                RAX => self.save.rax = value,
                RSP => self.save.rsp = value,
                number => self.registers[usize::from(number)] = value,
            }
        }

        fn save(&self) -> &SaveArea {
            &self.save
        }

        fn save_mut(&mut self) -> &mut SaveArea {
            &mut self.save
        }

        fn fetched(&self) -> &[u8] {
            &self.fetched
        }

        fn vector(&self, register: Vector) -> [u8; WIDEST] {
            self.vectors[usize::from(register.number())]
        }

        fn set_vector(&mut self, register: Vector, value: [u8; WIDEST]) {
            self.vectors[usize::from(register.number())] = value;
        }
    }

    /// Bytes in the smallest page, which is the granularity everything about a
    /// guest's memory is decided at.
    pub(crate) const PAGE: u64 = 4096;

    /// A guest's memory that is a map from page to contents, standing in for
    /// one reached through two levels of page table.
    ///
    /// Translation is deliberately not the identity. Each page is given a
    /// physical address unrelated to its linear one, so that code which
    /// conflates the two — or which assumes the page after a linear page is
    /// the physical page after it — fails here rather than on a machine.
    ///
    /// The pages sit behind a cell because a guest's memory is written through
    /// a shared reference: the real one is a handle onto tables the
    /// hypervisor reaches, not something the emulator owns exclusively.
    #[derive(Debug, Default)]
    pub(crate) struct Memory {
        pages: RefCell<BTreeMap<u64, Page>>,
        addressing: Option<Addressing>,
        /// How many times a linear address has been translated.
        ///
        /// A real translation is a walk of the guest's own page tables, which
        /// is the dominant cost of emulating anything and the thing a
        /// repeated instruction must not pay per element. Counting them
        /// is what lets a test assert that it does not.
        walks: Cell<usize>,
    }

    /// One page of a guest's memory: where it really is, what is in it, and
    /// what the guest may do to it.
    #[derive(Clone, Debug)]
    struct Page {
        gpa: u64,
        bytes: Vec<u8>,
        writable: bool,
    }

    impl Memory {
        /// A guest with nothing described at all, translating the way that
        /// machine does.
        pub(crate) fn new(machine: &Machine) -> Self {
            Self {
                pages: RefCell::new(BTreeMap::new()),
                addressing: Some(machine.addressing()),
                walks: Cell::new(0),
            }
        }

        /// Describes one page, at a guest physical address of this map's
        /// choosing.
        ///
        /// The address is derived from the linear page number by a fixed
        /// scattering rather than by adding a constant, so neighbouring linear
        /// pages are not neighbouring physical ones.
        pub(crate) fn map(&mut self, linear: u64) -> &mut Self {
            self.map_at(linear, scattered(linear))
        }

        /// Describes one page at a chosen guest physical address, for the tests
        /// that care which region a byte lands in.
        pub(crate) fn map_at(&mut self, linear: u64, gpa: u64) -> &mut Self {
            self.pages.borrow_mut().insert(
                linear / PAGE,
                Page {
                    gpa,
                    bytes: vec![0; as_usize(PAGE)],
                    writable: true,
                },
            );
            self
        }

        /// Describes one page the guest may read and not write.
        pub(crate) fn map_read_only(&mut self, linear: u64) -> &mut Self {
            self.map(linear);
            if let Some(page) = self.pages.borrow_mut().get_mut(&(linear / PAGE)) {
                page.writable = false;
            }
            self
        }

        /// Puts bytes in the guest's memory, describing whatever pages they
        /// cross.
        pub(crate) fn fill(&mut self, linear: u64, bytes: &[u8]) -> &mut Self {
            for (offset, byte) in bytes.iter().enumerate() {
                let at = linear.wrapping_add(offset as u64);
                if !self.pages.borrow().contains_key(&(at / PAGE)) {
                    self.map(at);
                }
                let mut pages = self.pages.borrow_mut();
                let page = pages.get_mut(&(at / PAGE)).expect("just described");
                page.bytes[as_usize(at % PAGE)] = *byte;
            }
            self
        }

        /// What the guest has at an address, as far as `bytes`.
        pub(crate) fn peek(&self, linear: u64, bytes: usize) -> Vec<u8> {
            let mut into = vec![0; bytes];
            self.read(linear, &mut into).expect("described memory");
            into
        }

        /// Where a linear page was put, for the tests that need the address a
        /// region has to cover.
        pub(crate) fn gpa_of(&self, linear: u64) -> u64 {
            self.pages
                .borrow()
                .get(&(linear / PAGE))
                .map(|page| page.gpa + linear % PAGE)
                .expect("the page must be described")
        }

        /// How many page-table walks have been made so far.
        pub(crate) fn walks(&self) -> usize {
            self.walks.get()
        }

        /// Forgets the count, so a test can measure one phase of a run.
        pub(crate) fn forget_walks(&self) {
            self.walks.set(0);
        }

        /// Whether every byte of a range is the guest's to write.
        fn writable(&self, linear: u64, bytes: usize) -> Result<bool, MemoryError> {
            let pages = self.pages.borrow();
            (0..bytes as u64).try_fold(true, |allowed, offset| {
                let at = linear.wrapping_add(offset);
                let page = pages
                    .get(&(at / PAGE))
                    .ok_or(MemoryError::Untranslated { linear: at })?;
                Ok(allowed && page.writable)
            })
        }
    }

    /// Where a linear page is put, as a function of which page it is.
    ///
    /// Deliberately not monotonic: page `n + 1` does not follow page `n`, so
    /// nothing can accidentally rely on a linear span being physically
    /// contiguous.
    fn scattered(linear: u64) -> u64 {
        const BASE: u64 = 0x4000_0000;
        const STRIDE: u64 = 0x10 * PAGE;
        BASE + (linear / PAGE).wrapping_mul(STRIDE) % 0x1000_0000
    }

    /// A length as `usize`, for a map whose pages are `Vec`s.
    fn as_usize(value: u64) -> usize {
        usize::try_from(value).expect("a page offset fits a host pointer")
    }

    impl Guest for Memory {
        fn translate(&self, linear: u64) -> Result<PhysAddr, MemoryError> {
            self.walks.set(self.walks.get() + 1);
            self.pages
                .borrow()
                .get(&(linear / PAGE))
                .map(|page| PhysAddr::new(page.gpa + linear % PAGE))
                .ok_or(MemoryError::Untranslated { linear })
        }

        fn read(&self, linear: u64, into: &mut [u8]) -> Result<(), MemoryError> {
            let pages = self.pages.borrow();
            for (offset, byte) in into.iter_mut().enumerate() {
                let at = linear.wrapping_add(offset as u64);
                let page = pages
                    .get(&(at / PAGE))
                    .ok_or(MemoryError::Untranslated { linear: at })?;
                *byte = page.bytes[as_usize(at % PAGE)];
            }
            Ok(())
        }

        /// All of it or none of it, as the real one is: a range with one
        /// unwritable byte in it is discarded whole rather than partly applied.
        fn write(&self, linear: u64, from: &[u8]) -> Result<Written, MemoryError> {
            if !self.writable(linear, from.len())? {
                return Ok(Written::Discarded);
            }
            let mut pages = self.pages.borrow_mut();
            for (offset, byte) in from.iter().enumerate() {
                let at = linear.wrapping_add(offset as u64);
                let page = pages
                    .get_mut(&(at / PAGE))
                    .ok_or(MemoryError::Untranslated { linear: at })?;
                page.bytes[as_usize(at % PAGE)] = *byte;
            }
            Ok(Written::Committed)
        }

        fn writable(&self, linear: u64, width: Width) -> Result<bool, MemoryError> {
            Self::writable(self, linear, width.bytes())
        }

        fn addressing(&self) -> &Addressing {
            self.addressing
                .as_ref()
                .expect("a guest memory must be built from a machine")
        }
    }
}
