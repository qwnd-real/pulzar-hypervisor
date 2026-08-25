//! Getting tables out of uACPI, and giving them back.
//!
//! uACPI owns the machine's table directory: it reads the root pointer, decides
//! between the two directories a root pointer can name, checks every header and
//! every checksum, and keeps what passed. What it offers in return is a table
//! by position or by signature, mapped and reference-counted — a table stays
//! mapped for exactly as long as something holds a reference to it.
//!
//! That reference count is the whole reason this module exists rather than the
//! calls being made where the tables are wanted. A table taken and not given
//! back is a mapping that lives forever, and on the firmware side those
//! mappings come out of a window with a fixed number of slots in it. So a table
//! taken here is a [`Held`], which gives its reference back when it is dropped,
//! and there is no way to take one without getting a value that does.
//!
//! # What is read through a held table, and what is not
//!
//! Everything this crate parses is read while the table is held and copied out
//! before it is released. That is the same discipline the crate had when it
//! read firmware's memory directly, and it matters more now: the mapping is
//! uACPI's, and uACPI is entitled to take it down the moment the last reference
//! goes.

use core::{ffi::c_char, slice};

use uacpi_sys::{Status, raw};

use crate::{AcpiError, Signature, Table, as_u64, as_usize, raw::Fields, sdt, sdt::HEADER_BYTES};

/// A table uACPI has handed over, mapped for as long as this value lives.
pub struct Held {
    /// uACPI's own description of the table: where it is mapped, and the
    /// position it is kept at.
    table: raw::uacpi_table,
}

impl Held {
    /// Takes the first table with this signature, if the machine has one.
    ///
    /// The first, because a machine may legitimately list several tables under
    /// one signature — supplementary description tables being the usual case —
    /// and none of the ones this crate parses is ever among them.
    ///
    /// # Errors
    ///
    /// [`AcpiError::Uacpi`] for anything uACPI reported other than the table
    /// not being there, which is `Ok(None)`.
    pub fn find(signature: Signature) -> Result<Option<Self>, AcpiError> {
        let name = signature.terminated();
        Self::take(|table| {
            // SAFETY: the name is four characters and a terminator in this
            // function's own storage, which outlives the call, and `table` is
            // storage for one description.
            Status::new(unsafe {
                raw::uacpi_table_find_by_signature(name.as_ptr().cast::<c_char>(), table)
            })
        })
    }

    /// Takes the table uACPI keeps at `index`, if it keeps one there.
    ///
    /// # Errors
    ///
    /// As [`Held::find`].
    pub fn at(index: usize) -> Result<Option<Self>, AcpiError> {
        Self::take(|table| {
            // SAFETY: `table` is storage for one description, and an index uACPI
            // does not keep a table at is reported rather than read.
            Status::new(unsafe { raw::uacpi_table_get_by_index(index, table) })
        })
    }

    /// What the table's header says about it.
    ///
    /// # Errors
    ///
    /// [`AcpiError::Truncated`] if the table is shorter than a header, which
    /// for a table uACPI has checked cannot happen.
    pub fn describe(&self) -> Result<Table, AcpiError> {
        Table::read(self.index(), &self.head())
    }

    /// The whole of the table, header included, ready for a parser.
    ///
    /// The borrow is this value's, which is what ties every read of the table
    /// to the reference that keeps it mapped.
    ///
    /// # Errors
    ///
    /// As [`Held::describe`].
    pub fn fields(&self) -> Result<Fields<'_>, AcpiError> {
        let length = as_usize(u64::from(self.head().u32(sdt::LENGTH)?));
        // uACPI refuses a table whose header declares less than a header's worth
        // of bytes, so this is a restatement of one of its own checks rather than
        // a case a machine can reach.
        if length < HEADER_BYTES {
            return Err(AcpiError::Truncated {
                at: self.address(),
                len: length,
                offset: 0,
                wanted: HEADER_BYTES,
            });
        }
        Ok(self.view(length))
    }

    /// Which position uACPI keeps the table at.
    pub const fn index(&self) -> usize {
        self.table.index
    }

    /// The header, which is as much of the table as is safe to read before its
    /// declared length has been read out of it.
    fn head(&self) -> Fields<'_> {
        self.view(HEADER_BYTES)
    }

    /// The first `length` bytes of the table.
    fn view(&self, length: usize) -> Fields<'_> {
        // SAFETY: uACPI maps a table's whole declared length when it hands out a
        // reference to it, and this value holds such a reference for as long as
        // the returned borrow lives. `length` is either a header's worth, which
        // uACPI has already established is present, or the declared length read
        // out of that header. The bytes are ones firmware wrote, so each is a
        // valid `u8` and `u8` needs no alignment, and nothing in this crate ever
        // forms a mutable path to a table.
        let bytes = unsafe { slice::from_raw_parts(self.pointer(), length) };
        Fields::new(self.address(), bytes)
    }

    /// Where uACPI mapped the table.
    fn pointer(&self) -> *const u8 {
        // SAFETY: every arm of the union is the same address in a different type,
        // and uACPI fills it in for every table it hands out.
        unsafe { self.table.__bindgen_anon_1.ptr }.cast::<u8>()
    }

    /// The mapped address, as a number for a log line to name the table by.
    fn address(&self) -> u64 {
        // SAFETY: as in `pointer`.
        as_u64(unsafe { self.table.__bindgen_anon_1.virt_addr })
    }

    /// Runs one of uACPI's lookups and wraps whatever it found.
    fn take(
        lookup: impl FnOnce(*mut raw::uacpi_table) -> Status,
    ) -> Result<Option<Self>, AcpiError> {
        let mut table = raw::uacpi_table {
            __bindgen_anon_1: raw::uacpi_table__bindgen_ty_1 { virt_addr: 0 },
            index: 0,
        };
        let status = lookup(&raw mut table);
        if status.is_ok() {
            Ok(Some(Self { table }))
        } else if status == Status::NOT_FOUND {
            Ok(None)
        } else {
            Err(AcpiError::Uacpi { status })
        }
    }
}

impl Drop for Held {
    /// Gives the reference back, which is what lets uACPI take the mapping
    /// down.
    ///
    /// A refusal here cannot be reported — nothing is left to report it to —
    /// and cannot be acted on either: the only failure uACPI has for this
    /// is a description it does not recognize, and this one came from
    /// uACPI.
    fn drop(&mut self) {
        // SAFETY: the description is one uACPI filled in for a reference it
        // handed to this value, and this is the only place that reference is
        // given back.
        let _ = unsafe { raw::uacpi_table_unref(&raw mut self.table) };
    }
}

/// How many tables uACPI is keeping.
///
/// The positions are `0..count`, and a position uACPI has nothing at is
/// reported by [`Held::at`] rather than being ruled out by this: the count is a
/// bound, and a table can be dropped from the directory between the two calls.
pub fn count() -> usize {
    // SAFETY: a read of one field of uACPI's context, valid from the moment the
    // table subsystem is up and answering zero before that.
    unsafe { raw::uacpi_table_count() }
}
