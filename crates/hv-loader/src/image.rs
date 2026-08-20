//! PE32+ parsing and relocation for the one image this loader loads.
//!
//! Firmware would happily load the hypervisor image for us, but then firmware
//! would own where it lands, and the image has to be relocated into a
//! randomized high-half address that no UEFI service can express. So the loader
//! parses the image itself.
//!
//! This is deliberately not a general PE loader. It accepts exactly the shape
//! `rust-lld` produces for `x86_64-unknown-uefi` and refuses everything else
//! rather than interpreting it: unknown relocation types, overlapping sections,
//! a section that is both writable and executable, a missing relocation
//! directory. Every refusal is a named error, because an image this loader
//! cannot place correctly must stop the boot instead of being mapped wrongly.
//! `goblin` and `object` were both considered and both need an allocator.
//!
//! Loading is split from parsing: [`Image::parse`] reads the headers out of a
//! buffer, and the caller — which is the only thing holding the file handle and
//! the destination frames — copies the bytes. [`Image::relocate`] then patches
//! the copy in place.

use paging::Protection;
use thiserror::Error;

/// Sections this loader accepts in one image. `rust-lld` emits a handful, so
/// the bound is generous in practice; it exists so the section table fits a
/// fixed array instead of an allocation. An image that exceeds it is refused
/// with [`ImageError::TooManySections`], and raising the bound is the fix if a
/// future toolchain ever emits more.
const MAX_SECTIONS: usize = 24;

/// The only section alignment this loader supports, which is also the page size
/// its mappings use. A larger alignment would mean sections that cannot be
/// given independent protections at page granularity.
const SECTION_ALIGNMENT: u32 = 4096;

/// A parsed PE32+ image, ready to be copied and relocated.
#[derive(Debug)]
pub struct Image {
    size: u64,
    entry: u32,
    headers: u32,
    linked_base: u64,
    relocations: (u32, u32),
    sections: [Section; MAX_SECTIONS],
    count: usize,
}

/// One section of the image, as it will be mapped.
#[derive(Clone, Copy, Debug)]
pub struct Section {
    /// Kept as the linker wrote it, padding included, and trimmed by
    /// [`Section::name`] — the raw bytes are of no use to a caller.
    name: [u8; NAME_BYTES],
    /// Offset of the section from the image base, in both the virtual span and
    /// the destination frames.
    pub offset: u64,
    /// Bytes the section occupies once mapped.
    pub size: u64,
    /// Offset of the section's bytes in the file. Zero-length for uninitialized
    /// data, which is already zero in freshly allocated frames.
    pub file_offset: u64,
    /// Bytes to read from the file, never more than [`Section::size`].
    pub file_size: u64,
    /// What the section is mapped as, derived from its characteristics.
    pub protection: Protection,
}

impl Section {
    /// The section's name, for the boot log.
    ///
    /// A section name is eight bytes padded with NULs, and is not required to
    /// be valid UTF-8 — a name that is not is reported as `"?"` rather than
    /// stopping a boot over a diagnostic string.
    pub fn name(&self) -> &str {
        let end = self
            .name
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(NAME_BYTES);
        str::from_utf8(&self.name[..end]).unwrap_or("?")
    }

    /// Filler for the unused tail of the fixed-size section table.
    const EMPTY: Self = Self {
        name: [0; NAME_BYTES],
        offset: 0,
        size: 0,
        file_offset: 0,
        file_size: 0,
        protection: Protection::ReadOnly,
    };
}

impl Image {
    /// Parses the headers at the front of the image file.
    ///
    /// `headers` needs to hold the DOS header, the PE headers and the whole
    /// section table — one 4 KiB page covers all three for any image this
    /// loader will see, and a short buffer is reported rather than assumed
    /// away.
    ///
    /// # Errors
    ///
    /// [`ImageError::Truncated`] if `headers` ends inside a structure this
    /// reads, or one of the rejection variants for an image this loader
    /// will not place.
    pub fn parse(headers: &[u8]) -> Result<Self, ImageError> {
        if u16_at(headers, 0)? != DOS_MAGIC {
            return Err(ImageError::NotPortableExecutable);
        }
        let pe = u32_at(headers, 0x3C)? as usize;
        if u32_at(headers, pe)? != PE_MAGIC {
            return Err(ImageError::NotPortableExecutable);
        }

        let machine = u16_at(headers, pe + 4)?;
        if machine != MACHINE_AMD64 {
            return Err(ImageError::Machine { machine });
        }
        let count = u16_at(headers, pe + 6)? as usize;
        if count > MAX_SECTIONS {
            return Err(ImageError::TooManySections { count });
        }
        let optional = pe + 24;
        let optional_size = u16_at(headers, pe + 20)? as usize;
        let magic = u16_at(headers, optional)?;
        if magic != OPTIONAL_MAGIC_PE32_PLUS {
            return Err(ImageError::NotPe32Plus { magic });
        }
        let subsystem = u16_at(headers, optional + 68)?;
        if subsystem != SUBSYSTEM_EFI_APPLICATION {
            return Err(ImageError::Subsystem { subsystem });
        }
        let alignment = u32_at(headers, optional + 32)?;
        if alignment != SECTION_ALIGNMENT {
            return Err(ImageError::SectionAlignment { alignment });
        }

        let size = u32_at(headers, optional + 56)?;
        if size == 0 || !size.is_multiple_of(SECTION_ALIGNMENT) {
            return Err(ImageError::ImageSize { size });
        }
        let relocations = data_directory(headers, optional, optional_size, RELOCATION_DIRECTORY)?;
        if relocations.1 == 0 {
            return Err(ImageError::NoRelocations);
        }

        let mut image = Self {
            size: u64::from(size),
            entry: u32_at(headers, optional + 16)?,
            headers: u32_at(headers, optional + 60)?,
            linked_base: u64_at(headers, optional + 24)?,
            relocations,
            sections: [Section::EMPTY; MAX_SECTIONS],
            count,
        };
        if image.entry >= size {
            return Err(ImageError::EntryOutsideImage { entry: image.entry });
        }
        if image.headers == 0 || image.headers > size {
            return Err(ImageError::HeaderSize {
                headers: image.headers,
            });
        }
        image.read_sections(headers, optional + optional_size)?;
        if !image.sections().iter().any(|section| {
            u64::from(image.entry) >= section.offset
                && u64::from(image.entry) < section.offset + section.size
                && section.protection == Protection::ReadExecute
        }) {
            return Err(ImageError::EntryNotExecutable { entry: image.entry });
        }
        Ok(image)
    }

    /// Bytes of virtual address space the image occupies.
    pub const fn size(&self) -> u64 {
        self.size
    }

    /// Bytes of headers at the front of the image, mapped read-only so the
    /// image can describe itself.
    pub fn header_bytes(&self) -> u64 {
        u64::from(self.headers)
    }

    /// The sections, in ascending address order.
    pub fn sections(&self) -> &[Section] {
        &self.sections[..self.count]
    }

    /// Address of the entry point once the image is based at `base`.
    pub fn entry(&self, base: u64) -> u64 {
        base + u64::from(self.entry)
    }

    /// Rebases the copied image in `image` from its link-time base to `base`.
    ///
    /// The image must already hold the headers and every section's bytes at
    /// their virtual offsets: the relocation directory is read out of the
    /// copy, not out of the file, exactly as the processor will read the
    /// patched values back.
    ///
    /// # Errors
    ///
    /// [`ImageError::Truncated`] if a relocation block or target lies outside
    /// the image, [`ImageError::MalformedRelocationBlock`] for a block
    /// header that cannot be walked, or
    /// [`ImageError::UnsupportedRelocation`] for a fixup type this loader
    /// will not apply.
    pub fn relocate(&self, image: &mut [u8], base: u64) -> Result<(), ImageError> {
        let delta = base.wrapping_sub(self.linked_base);
        let (start, size) = self.relocations;
        let mut offset = start as usize;
        let end = offset + size as usize;
        while offset < end {
            let page = u32_at(image, offset)? as usize;
            let block = u32_at(image, offset + 4)? as usize;
            if block < RELOCATION_BLOCK_HEADER || offset + block > end {
                return Err(ImageError::MalformedRelocationBlock { offset, block });
            }
            for fixup in (RELOCATION_BLOCK_HEADER..block).step_by(size_of::<u16>()) {
                let entry = u16_at(image, offset + fixup)?;
                let target = page + usize::from(entry & 0x0FFF);
                match entry >> 12 {
                    // Padding to a four-byte block size, and nothing to apply.
                    RELOCATION_ABSOLUTE => {}
                    RELOCATION_DIR64 => {
                        let value = u64_at(image, target)?.wrapping_add(delta);
                        write_u64_at(image, target, value)?;
                    }
                    kind => return Err(ImageError::UnsupportedRelocation { kind }),
                }
            }
            offset += block;
        }
        Ok(())
    }

    /// Reads the section table and checks that it describes a span this loader
    /// can map: page-aligned, ascending, non-overlapping, inside the image, and
    /// never both writable and executable.
    fn read_sections(&mut self, headers: &[u8], table: usize) -> Result<(), ImageError> {
        let mut lowest = u64::from(self.headers).next_multiple_of(u64::from(SECTION_ALIGNMENT));
        for index in 0..self.count {
            let entry = table + index * SECTION_HEADER_SIZE;
            let offset = u64::from(u32_at(headers, entry + 12)?);
            let size = u64::from(u32_at(headers, entry + 8)?)
                .next_multiple_of(u64::from(SECTION_ALIGNMENT));
            let characteristics = u32_at(headers, entry + 36)?;
            if offset < lowest || offset + size > self.size {
                return Err(ImageError::SectionOutsideImage { offset, size });
            }
            lowest = offset + size;
            self.sections[index] = Section {
                name: array_at(headers, entry)?,
                offset,
                size,
                file_offset: u64::from(u32_at(headers, entry + 20)?),
                file_size: u64::from(u32_at(headers, entry + 16)?).min(size),
                protection: protection(characteristics)?,
            };
        }
        Ok(())
    }
}

/// `MZ`, the signature every PE image still starts with.
const DOS_MAGIC: u16 = 0x5A4D;

/// `PE\0\0`, at the offset the DOS header's `e_lfanew` field points to.
const PE_MAGIC: u32 = 0x0000_4550;

/// `IMAGE_FILE_MACHINE_AMD64`.
const MACHINE_AMD64: u16 = 0x8664;

/// `IMAGE_NT_OPTIONAL_HDR64_MAGIC`, which is what makes the optional header a
/// PE32+ one and its `ImageBase` field 64 bits wide.
const OPTIONAL_MAGIC_PE32_PLUS: u16 = 0x020B;

/// `IMAGE_SUBSYSTEM_EFI_APPLICATION`.
const SUBSYSTEM_EFI_APPLICATION: u16 = 10;

/// Index of the base relocation table among the data directories.
const RELOCATION_DIRECTORY: usize = 5;

/// Bytes of one entry in the section table.
const SECTION_HEADER_SIZE: usize = 40;

/// Bytes of a section's name, NUL-padded rather than NUL-terminated.
const NAME_BYTES: usize = 8;

/// Bytes of `VirtualAddress` and `SizeOfBlock` before a relocation block's
/// fixups.
const RELOCATION_BLOCK_HEADER: usize = 8;

/// `IMAGE_REL_BASED_ABSOLUTE`.
const RELOCATION_ABSOLUTE: u16 = 0;

/// `IMAGE_REL_BASED_DIR64`, the only fixup a 64-bit image needs and the only
/// one this loader applies.
const RELOCATION_DIR64: u16 = 10;

/// `IMAGE_SCN_MEM_EXECUTE`.
const SECTION_EXECUTE: u32 = 0x2000_0000;

/// `IMAGE_SCN_MEM_WRITE`.
const SECTION_WRITE: u32 = 0x8000_0000;

/// Why an image was rejected.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ImageError {
    /// A field lies outside the bytes available.
    #[error("the image ends inside the structure at offset {offset:#x}")]
    Truncated {
        /// Where the read would have started.
        offset: usize,
    },
    /// No `MZ` or no `PE\0\0` signature.
    #[error("the file is not a PE image")]
    NotPortableExecutable,
    /// Built for another architecture.
    #[error("the image is for machine {machine:#06x}, not x86-64")]
    Machine {
        /// The `Machine` field found.
        machine: u16,
    },
    /// A PE32 image, whose 32-bit `ImageBase` cannot express a high-half
    /// address.
    #[error("the image is not PE32+ (optional header magic {magic:#06x})")]
    NotPe32Plus {
        /// The optional header magic found.
        magic: u16,
    },
    /// Not a UEFI application, so its entry point ABI is not the one the loader
    /// jumps with.
    #[error("the image declares subsystem {subsystem}, not an EFI application")]
    Subsystem {
        /// The `Subsystem` field found.
        subsystem: u16,
    },
    /// Sections are aligned to something other than the page size, so they
    /// cannot be given independent protections.
    #[error("the image aligns sections to {alignment:#x}, not the page size")]
    SectionAlignment {
        /// The `SectionAlignment` field found.
        alignment: u32,
    },
    /// `SizeOfImage` is zero or not a whole number of pages.
    #[error("the image declares an unusable size of {size:#x} bytes")]
    ImageSize {
        /// The `SizeOfImage` field found.
        size: u32,
    },
    /// More sections than the fixed section table holds.
    #[error("the image has {count} sections, more than this loader accepts")]
    TooManySections {
        /// The `NumberOfSections` field found.
        count: usize,
    },
    /// The data directory the relocation table lives in is absent or empty, so
    /// the image cannot be moved away from its link-time base.
    #[error("the image carries no relocations and cannot be rebased")]
    NoRelocations,
    /// A section reaches outside the image or overlaps the one before it.
    #[error("a section at {offset:#x} spanning {size:#x} bytes does not fit the image")]
    SectionOutsideImage {
        /// Offset of the section from the image base.
        offset: u64,
        /// Bytes it would occupy.
        size: u64,
    },
    /// A section asks to be both writable and executable, which no mapping this
    /// loader makes will grant.
    #[error("a section is both writable and executable")]
    WritableAndExecutable,
    /// `AddressOfEntryPoint` is not inside the image.
    #[error("the entry point at {entry:#x} is outside the image")]
    EntryOutsideImage {
        /// The `AddressOfEntryPoint` field found.
        entry: u32,
    },
    /// `AddressOfEntryPoint` is not inside a section mapped executable, so the
    /// jump into it would run headers or data, or fault.
    #[error("the entry point at {entry:#x} is not in an executable section")]
    EntryNotExecutable {
        /// The `AddressOfEntryPoint` field found.
        entry: u32,
    },
    /// `SizeOfHeaders` is zero, or larger than the image the headers describe,
    /// so there is nothing to map or nowhere to map it.
    #[error("the image declares an unusable header size of {headers:#x} bytes")]
    HeaderSize {
        /// The `SizeOfHeaders` field found.
        headers: u32,
    },
    /// A relocation block's size makes it unwalkable.
    #[error("the relocation block at {offset:#x} declares an unusable size of {block:#x}")]
    MalformedRelocationBlock {
        /// Offset of the block in the image.
        offset: usize,
        /// The `SizeOfBlock` field found.
        block: usize,
    },
    /// A fixup type other than `ABSOLUTE` or `DIR64`. Applying it wrongly would
    /// corrupt the image silently, so it is refused.
    #[error("relocation type {kind} is not supported")]
    UnsupportedRelocation {
        /// The type nibble found.
        kind: u16,
    },
}

/// Protection for a section with these characteristics.
fn protection(characteristics: u32) -> Result<Protection, ImageError> {
    match (
        characteristics & SECTION_WRITE != 0,
        characteristics & SECTION_EXECUTE != 0,
    ) {
        (true, true) => Err(ImageError::WritableAndExecutable),
        (true, false) => Ok(Protection::ReadWrite),
        (false, true) => Ok(Protection::ReadExecute),
        (false, false) => Ok(Protection::ReadOnly),
    }
}

/// Address and size of one data directory, or `(0, 0)` if the optional header
/// is too short to have it.
fn data_directory(
    bytes: &[u8],
    optional: usize,
    optional_size: usize,
    index: usize,
) -> Result<(u32, u32), ImageError> {
    /// Offset of the data directory array within a PE32+ optional header.
    const DIRECTORIES: usize = 112;
    /// Bytes of one directory entry: an address and a size.
    const ENTRY: usize = 8;

    let count = u32_at(bytes, optional + 108)? as usize;
    if index >= count || DIRECTORIES + (index + 1) * ENTRY > optional_size {
        return Ok((0, 0));
    }
    let entry = optional + DIRECTORIES + index * ENTRY;
    Ok((u32_at(bytes, entry)?, u32_at(bytes, entry + 4)?))
}

/// A little-endian `u16` at `offset`.
fn u16_at(bytes: &[u8], offset: usize) -> Result<u16, ImageError> {
    array_at(bytes, offset).map(u16::from_le_bytes)
}

/// A little-endian `u32` at `offset`.
fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, ImageError> {
    array_at(bytes, offset).map(u32::from_le_bytes)
}

/// A little-endian `u64` at `offset`.
fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, ImageError> {
    array_at(bytes, offset).map(u64::from_le_bytes)
}

/// Overwrites the little-endian `u64` at `offset`.
fn write_u64_at(bytes: &mut [u8], offset: usize, value: u64) -> Result<(), ImageError> {
    bytes
        .get_mut(offset..)
        .and_then(|rest| rest.get_mut(..size_of::<u64>()))
        .ok_or(ImageError::Truncated { offset })?
        .copy_from_slice(&value.to_le_bytes());
    Ok(())
}

/// `N` bytes at `offset`.
///
/// Splitting the range in two `get` calls rather than forming
/// `offset..offset+N` keeps a large offset from overflowing into a range that
/// looks valid.
fn array_at<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N], ImageError> {
    bytes
        .get(offset..)
        .and_then(|rest| rest.get(..N))
        .and_then(|slice| slice.try_into().ok())
        .ok_or(ImageError::Truncated { offset })
}
