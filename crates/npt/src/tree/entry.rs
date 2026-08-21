//! What the bits of one nested table entry mean.
//!
//! Every value stored into one of these tables comes from [`encode`], and
//! everything anything here needs of a value it read comes from the predicates
//! below. That is what keeps the format from being restated: a second place
//! computing `PRESENT | USER_ACCESSIBLE` would be a second place for it to be
//! wrong, and the two would not have to disagree loudly to be a guest that
//! faults for ever.
//!
//! # Why the bits are the ones they are
//!
//! **Present and user-accessible on every entry, leaf and table alike.** A
//! guest's own page-table walk is a *user* access at the nested level, so a
//! table that is not user-accessible turns every such walk into a fault.
//!
//! **Write-back everywhere**, which is the absence of both cache bits rather
//! than a bit of its own. The effective memory type of a guest access is the
//! guest's own combined with the nested one, and write-back is the identity
//! element of that combination: not a claim that the memory is cacheable, but
//! the only encoding that leaves a guest free to mark its own device apertures
//! uncacheable and be obeyed.
//!
//! **Writable from [`Access::WRITE`], and always on a table**, again because a
//! guest walking its own tables writes at this level.
//!
//! **No-execute from the absence of [`Access::EXECUTE`], and never on a
//! table.** The host has no-execute translation enabled, so the bit on a table
//! would deny execution of everything below it.
//!
//! **The large-page bit from the level, never passed in.** Whether an entry
//! describes memory is what its level already says, and saying it twice is how
//! the two come to disagree.
//!
//! **Nothing at all is the only encoding that denies a read.** No present entry
//! has a bit that does, which is why a region every access to which must fault
//! — and an address the machine does not have — encode as zero rather than as
//! something merely restrictive.
//!
//! Each of those is asserted at the bottom of this file over every kind an
//! entry can describe, because they are properties of a number that an edit can
//! break while leaving the crate compiling.

use x86_64::{PhysAddr, structures::paging::PageTableFlags};

use crate::{
    map::{Access, Kind, RegionTag, Trap},
    tree::walk::Level,
};

/// What one entry of a nested page table is to say.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Entry {
    /// Nothing: the addresses it describes have no translation at all, and
    /// every access to one of them faults.
    Absent,
    /// The table of the level below.
    Table {
        /// Where that table is.
        frame: PhysAddr,
    },
    /// A run of `kind` beginning at `frame`, as much of it as one entry of
    /// `level` describes.
    Leaf {
        /// What a guest finds there.
        kind: Kind,
        /// The level the entry sits at.
        level: Level,
        /// Where the run really is.
        frame: PhysAddr,
    },
}

/// The quadword the processor's page walker reads for one entry.
pub(crate) const fn encode(entry: Entry) -> u64 {
    match entry {
        Entry::Absent => 0,
        // A table permits everything and denies nothing, because what a guest
        // may do with the memory under it is each leaf's to say and a constraint
        // here would apply to all five hundred and twelve of them at once.
        Entry::Table { frame } => quadword(permissions(Access::all()), frame.as_u64()),
        Entry::Leaf { kind, level, frame } => match permitted(kind) {
            Some(access) => quadword(permissions(access).union(large(level)), frame.as_u64()),
            None => 0,
        },
    }
}

/// Whether the entry describes anything at all.
pub(crate) const fn present(value: u64) -> bool {
    value & PageTableFlags::PRESENT.bits() != 0
}

/// Whether the entry describes memory rather than the table below it.
///
/// The level is part of the question because every entry of a page table
/// describes memory whether or not it carries the bit that says so.
pub(crate) const fn leaf(value: u64, level: Level) -> bool {
    matches!(level, Level::Page) || value & PageTableFlags::HUGE_PAGE.bits() != 0
}

/// Where what the entry names begins: the table below it, or the memory a leaf
/// describes.
pub(crate) const fn frame(value: u64) -> PhysAddr {
    PhysAddr::new_truncate(value & ADDRESS)
}

/// Whether a guest may write what a leaf describes.
pub(crate) const fn writable(value: u64) -> bool {
    value & PageTableFlags::WRITABLE.bits() != 0
}

/// Whether two entries say the same thing.
///
/// The accessed and dirty bits are left out of it, because the processor writes
/// them and software does not: an entry differing only there is the entry this
/// crate wrote, with a note from the hardware about what the guest has done
/// since, and storing over it would discard the note and buy nothing.
pub(crate) const fn unchanged(value: u64, wanted: u64) -> bool {
    let hardware = PageTableFlags::ACCESSED.union(PageTableFlags::DIRTY).bits();
    value & !hardware == wanted & !hardware
}

/// The entry describing the `slot`th slice of what `value` describes, one level
/// down.
///
/// The same memory with the same permissions and the same memory type, which is
/// what makes splitting a large page change no translation. The only bit that
/// moves is the one saying an entry describes memory, and only where the level
/// below is the page table, whose entries describe memory without it.
pub(crate) const fn narrowed(value: u64, below: Level, slot: u64) -> u64 {
    let inherited = PageTableFlags::from_bits_retain(value)
        .difference(PageTableFlags::HUGE_PAGE)
        .union(large(below));
    quadword(inherited, frame(value).as_u64() + slot * below.span())
}

/// What a guest may do with a run of this kind, or `None` where nothing is
/// described at all.
const fn permitted(kind: Kind) -> Option<Access> {
    Some(match kind {
        // The machine's own memory, as the guest is allowed to use it.
        Kind::Ram { access, .. } => access,
        // What the interrupt acceleration needs of a page whose contents it
        // never reads: memory the guest may write, going nowhere.
        Kind::Sink { .. } => Access::all(),
        // Readable and executable, so that a guest walking physical memory sees
        // zeroes rather than a fault; never writable, because the one frame it
        // reads as stands in for all of the hypervisor's memory at once and a
        // write there would be a write every other page of it saw.
        Kind::Shadow => Access::EXECUTE,
        // A page of the hypervisor's own memory handed to the guest on purpose,
        // which is a way of giving it something rather than memory it owns. The
        // write is taken away here rather than trusted never to have been asked
        // for: a writable page of the chunk is hypervisor memory under the
        // guest's control.
        Kind::Exposed { access, .. } => access.difference(Access::WRITE),
        // The hardware really behind the region, so that a read costs nothing
        // and a write is the only access that faults — which is the whole of
        // what trapping writes alone means.
        Kind::Interposed {
            trap: Trap::Writes, ..
        } => Access::all().difference(Access::WRITE),
        // Not described at all. No present entry has a bit that denies a read,
        // so this is the only encoding that faults on one — and an address the
        // machine does not have must never be described as one it does.
        Kind::Interposed {
            trap: Trap::Everything,
            ..
        }
        | Kind::Unaddressable => return None,
    })
}

/// The bits that say what a guest may do with what an entry describes.
const fn permissions(access: Access) -> PageTableFlags {
    let reachable = PageTableFlags::PRESENT.union(PageTableFlags::USER_ACCESSIBLE);
    let writes = if access.contains(Access::WRITE) {
        PageTableFlags::WRITABLE
    } else {
        PageTableFlags::empty()
    };
    let executes = if access.contains(Access::EXECUTE) {
        PageTableFlags::empty()
    } else {
        PageTableFlags::NO_EXECUTE
    };
    reachable.union(writes).union(executes)
}

/// The bit that says an entry above the page table describes memory rather than
/// the table below it.
const fn large(level: Level) -> PageTableFlags {
    if matches!(level, Level::Page) {
        PageTableFlags::empty()
    } else {
        PageTableFlags::HUGE_PAGE
    }
}

/// One entry's quadword: what it says, and where what it names begins.
///
/// The address is masked rather than trusted, because an address that was not a
/// page boundary would otherwise put its low bits where the flags are.
const fn quadword(flags: PageTableFlags, frame: u64) -> u64 {
    (flags.bits() & !ADDRESS) | (frame & ADDRESS)
}

/// The bits of an entry that are the address of what it names: bits 51 down to
/// 12, which is the widest a processor may implement and the width a
/// [`PhysAddr`] holds.
const ADDRESS: u64 = 0x000F_FFFF_FFFF_F000;

/// A frame for the assertions and the tests to name. Where it is matters to
/// none of them, only that it is aligned for every level a leaf can sit at.
const SOMEWHERE: PhysAddr = PhysAddr::new_truncate(1 << 30);

/// A name for the assertions and the tests to give a region.
const NAMED: RegionTag = RegionTag(0);

/// Ordinary memory the guest may do anything with, which is the kind the
/// assertions about a leaf's level name.
const OPEN: Kind = Kind::Ram {
    spa: SOMEWHERE,
    access: Access::all(),
};

/// Every kind of run an entry can describe, which is what the assertions below
/// and the tests quantify over.
const KINDS: [Kind; 8] = [
    OPEN,
    Kind::Shadow,
    Kind::Exposed {
        spa: SOMEWHERE,
        access: Access::EXECUTE,
    },
    Kind::Exposed {
        spa: SOMEWHERE,
        access: Access::empty(),
    },
    Kind::Sink { spa: SOMEWHERE },
    Kind::Interposed {
        tag: NAMED,
        trap: Trap::Writes,
    },
    Kind::Interposed {
        tag: NAMED,
        trap: Trap::Everything,
    },
    Kind::Unaddressable,
];

/// Every level a leaf can sit at, which is every level but the root.
const LEVELS: [Level; 3] = [Level::Pointer, Level::Directory, Level::Page];

/// The entry a table gets, which is the same whatever level it sits at.
const TABLE: u64 = encode(Entry::Table { frame: SOMEWHERE });

/// The entry one page of a region only whose writes are trapped gets.
const TRAPPED: u64 = leaf_of(
    Kind::Interposed {
        tag: NAMED,
        trap: Trap::Writes,
    },
    Level::Page,
);

/// The entry a leaf of `kind` gets at `level`.
const fn leaf_of(kind: Kind, level: Level) -> u64 {
    encode(Entry::Leaf {
        kind,
        level,
        frame: SOMEWHERE,
    })
}

/// Whether `value` carries every one of `flags`.
const fn carries(value: u64, flags: PageTableFlags) -> bool {
    value & flags.bits() == flags.bits()
}

/// The bits every kind that is described at all carries, and the bits any of
/// them carries, over every kind at every level a leaf can sit at.
const fn carried() -> (u64, u64) {
    let mut every = u64::MAX;
    let mut any = 0;
    let mut level = 0;
    while level < LEVELS.len() {
        let mut kind = 0;
        while kind < KINDS.len() {
            let value = leaf_of(KINDS[kind], LEVELS[level]);
            if value != 0 {
                every &= value;
                any |= value;
            }
            kind += 1;
        }
        level += 1;
    }
    (every, any)
}

const _: () = assert!(
    carries(
        carried().0,
        PageTableFlags::PRESENT.union(PageTableFlags::USER_ACCESSIBLE)
    ),
    "every entry that describes anything must be present and user-accessible: a \
     guest's own page-table walk is a user access at this level, so anything \
     less turns every such walk into a fault",
);
const _: () = assert!(
    carried().1
        & PageTableFlags::WRITE_THROUGH
            .union(PageTableFlags::NO_CACHE)
            .bits()
        == 0,
    "every leaf must be write-back, which is the nested memory type that leaves \
     the guest's own choice in force",
);
const _: () = assert!(
    carries(
        TABLE,
        PageTableFlags::PRESENT
            .union(PageTableFlags::USER_ACCESSIBLE)
            .union(PageTableFlags::WRITABLE)
    ) && TABLE
        & PageTableFlags::NO_EXECUTE
            .union(PageTableFlags::HUGE_PAGE)
            .bits()
        == 0,
    "a table must permit what a guest's own page-table walk needs and deny \
     nothing: no-execute on one would deny execution of everything below it, and \
     the large-page bit would make it memory",
);
const _: () = assert!(
    !writable(leaf_of(Kind::Shadow, Level::Page))
        && !writable(leaf_of(
            Kind::Exposed {
                spa: SOMEWHERE,
                access: Access::all()
            },
            Level::Page
        )),
    "the one frame the hypervisor's own memory reads as must stay zero, and a \
     page shown to the guest is a way of handing it something rather than \
     memory it owns, whatever access is asked for it",
);
const _: () = assert!(
    present(TRAPPED) && !writable(TRAPPED),
    "a region only whose writes are trapped must be present, so a read costs \
     nothing, and not writable, so a write faults",
);
const _: () = assert!(
    leaf_of(
        Kind::Interposed {
            tag: NAMED,
            trap: Trap::Everything
        },
        Level::Page
    ) == 0
        && leaf_of(Kind::Unaddressable, Level::Page) == 0,
    "no present entry has a bit that denies a read, so nothing at all is the \
     only encoding that faults on one",
);
const _: () = assert!(
    !carries(leaf_of(OPEN, Level::Page), PageTableFlags::HUGE_PAGE)
        && carries(leaf_of(OPEN, Level::Directory), PageTableFlags::HUGE_PAGE)
        && carries(leaf_of(OPEN, Level::Pointer), PageTableFlags::HUGE_PAGE),
    "the bit saying an entry describes memory belongs to every leaf above the \
     page table and to no entry in one, where an entry describes memory without \
     it",
);
const _: () = assert!(
    ADDRESS.trailing_zeros() == Level::Page.span().trailing_zeros()
        && ADDRESS.leading_zeros() == 12,
    "the address in an entry must be a page number in the fifty-two bits a \
     physical address has",
);

#[cfg(test)]
mod tests {
    //! What every kind of run comes to as bits, at every level an entry can
    //! describe one at.
    //!
    //! The invariants every encoding upholds are asserted above, at compile
    //! time, rather than restated here: what these add is the table of what
    //! each kind in particular permits, written out by hand so that changing
    //! the encoding has to be a change here as well.

    use x86_64::{PhysAddr, structures::paging::PageTableFlags};

    use super::{
        Access, Entry, KINDS, Kind, LEVELS, Level, NAMED, OPEN, SOMEWHERE, Trap, encode, frame,
        leaf_of, narrowed, present, unchanged, writable,
    };

    #[test]
    fn each_kind_permits_exactly_what_the_guest_may_do_with_it() {
        // `None` is a kind nothing describes at all; the pair is whether the
        // guest may write it and whether it may execute from it.
        let expected = [
            (OPEN, Some((true, true))),
            (Kind::Shadow, Some((false, true))),
            (
                Kind::Exposed {
                    spa: SOMEWHERE,
                    access: Access::EXECUTE,
                },
                Some((false, true)),
            ),
            (
                Kind::Exposed {
                    spa: SOMEWHERE,
                    access: Access::empty(),
                },
                Some((false, false)),
            ),
            (Kind::Sink { spa: SOMEWHERE }, Some((true, true))),
            (
                Kind::Interposed {
                    tag: NAMED,
                    trap: Trap::Writes,
                },
                Some((false, true)),
            ),
            (
                Kind::Interposed {
                    tag: NAMED,
                    trap: Trap::Everything,
                },
                None,
            ),
            (Kind::Unaddressable, None),
        ];
        assert_eq!(
            expected.len(),
            KINDS.len(),
            "every kind an entry can describe must be accounted for here"
        );

        for (kind, permits) in expected {
            for level in LEVELS {
                let value = leaf_of(kind, level);
                let Some((writes, executes)) = permits else {
                    assert_eq!(value, 0, "{kind:?} must not be described at all");
                    continue;
                };
                assert!(present(value), "{kind:?} is described, so it is present");
                assert_eq!(
                    writable(value),
                    writes,
                    "whether the guest may write {kind:?}"
                );
                assert_eq!(
                    value & PageTableFlags::NO_EXECUTE.bits() == 0,
                    executes,
                    "whether the guest may execute {kind:?}"
                );
                assert_eq!(frame(value), SOMEWHERE, "where {kind:?} really is");
                assert_eq!(
                    value & PageTableFlags::HUGE_PAGE.bits() != 0,
                    !matches!(level, Level::Page),
                    "whether an entry of {level:?} describing {kind:?} says it is memory"
                );
            }
        }
    }

    #[test]
    fn narrowing_a_leaf_describes_the_same_memory_one_level_down() {
        let large = leaf_of(OPEN, Level::Directory);
        for slot in [0, 1, 511] {
            let page = narrowed(large, Level::Page, slot);
            assert_eq!(
                frame(page),
                SOMEWHERE + slot * Level::Page.span(),
                "the slice of the region this entry describes"
            );
            assert_eq!(
                writable(page),
                writable(large),
                "with the permissions the entry it replaces gave"
            );
            assert_eq!(
                page & PageTableFlags::HUGE_PAGE.bits(),
                0,
                "and without the bit, a page table's entries being memory anyway"
            );
        }

        let huge = leaf_of(OPEN, Level::Pointer);
        let kept = narrowed(huge, Level::Directory, 3);
        assert_eq!(frame(kept), SOMEWHERE + 3 * Level::Directory.span());
        assert_ne!(
            kept & PageTableFlags::HUGE_PAGE.bits(),
            0,
            "a slice of a gigabyte is still memory rather than a table"
        );
    }

    #[test]
    fn an_entry_the_processor_has_written_into_still_says_the_same_thing() {
        let value = leaf_of(OPEN, Level::Page);
        let touched = value | PageTableFlags::ACCESSED.union(PageTableFlags::DIRTY).bits();

        assert!(
            unchanged(touched, value),
            "the accessed and dirty bits are the processor's own note, not a \
             difference worth storing over"
        );
        assert!(
            !unchanged(value & !PageTableFlags::WRITABLE.bits(), value),
            "while a permission that differs is a different entry"
        );
        assert!(
            !unchanged(value, leaf_of(OPEN, Level::Page) + Level::Page.span()),
            "and so is memory somewhere else"
        );
    }

    #[test]
    fn an_address_is_never_mistaken_for_a_permission() {
        // What an entry names is a page, so the bits below one are not part of
        // the address and cannot reach the flags kept there.
        let value = leaf_of(OPEN, Level::Directory);
        let stray = Entry::Leaf {
            kind: OPEN,
            level: Level::Directory,
            frame: PhysAddr::new(SOMEWHERE.as_u64() + Level::Page.span() - 1),
        };

        assert_eq!(
            encode(stray),
            value,
            "an address inside a page names that page and nothing else"
        );
    }
}
