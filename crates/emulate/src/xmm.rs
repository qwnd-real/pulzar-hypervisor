//! The guest's vector registers, which are still in the processor.
//!
//! Nothing saves them across a world switch — `VMRUN` and `#VMEXIT` do not
//! carry vector state, and neither does the switch this hypervisor wrote. That
//! would be a bug in almost any hypervisor and is not one here, for a reason
//! that belongs to the build rather than to any code: this image is compiled
//! for a target whose feature string is `-mmx,-sse,+soft-float`, so no
//! instruction the compiler emits can name a vector register. The guest's are
//! therefore still sitting in the processor while the exit handler runs,
//! untouched.
//!
//! Which makes emulating a vector move remarkably direct. Reading the guest's
//! `XMM3` is reading `XMM3`. There is no saved copy to find, and no copy to
//! write back afterwards — a store into the register *is* the guest's register
//! changing.
//!
//! # The two instructions here are the only vector instructions in the image
//!
//! Everything below is assembly for that reason. It is also why
//! [`available`] exists: a processor executing `movdqu` needs the operating
//! system's vector support switched on, and this operating system has no vector
//! support and never enabled any. The bits are checked, and the one that can be
//! set is set, before anything here is reached.

use core::arch::asm;

use iced_x86::Register;
use x86_64::registers::control::{Cr0, Cr0Flags, Cr4, Cr4Flags};

use crate::{EmulateError, value::Width};

/// Turns on what a vector move needs, and refuses if the processor cannot.
///
/// The extension bit is the hypervisor's own to set: control register four is
/// swapped on every entry and exit, so the guest neither sees this nor is
/// affected by it. The other two are refusals rather than adjustments —
/// emulation is not the place to start changing how the processor handles
/// floating-point state, and a machine in either of those states has something
/// else wrong with it.
///
/// # Errors
///
/// [`EmulateError::NoVectors`] if the processor is set to trap vector
/// instructions rather than execute them.
pub(crate) fn available() -> Result<(), EmulateError> {
    let cr0 = Cr0::read();
    if cr0.contains(Cr0Flags::EMULATE_COPROCESSOR) || cr0.contains(Cr0Flags::TASK_SWITCHED) {
        return Err(EmulateError::NoVectors);
    }
    if !Cr4::read().contains(Cr4Flags::OSFXSR) {
        // SAFETY: this bit only says that the operating system is prepared to
        // handle vector state, which for this one means the two instructions in
        // this module and nothing else. It changes no translation and no
        // protection, and control register four is host state that `VMRUN`
        // swaps out, so no guest observes it.
        unsafe { Cr4::update(|flags| flags.insert(Cr4Flags::OSFXSR)) };
    }
    Ok(())
}

/// Declares the sixteen registers once, and everything that is one arm apiece.
///
/// A register number is not something an instruction can take as an operand, so
/// selecting one at runtime is a match however it is written. Writing the match
/// out by hand would be three of them, sixteen arms each, with the register
/// name repeated in an assembly string every time. This generates all three
/// from one list, which is also what makes [`Vector`] an enum rather than a
/// checked integer — and an enum is what lets each match be exhaustive with no
/// arm for a number that cannot happen.
macro_rules! vectors {
    ($($variant:ident => $register:ident => $number:literal,)*) => {
        /// One of the processor's sixteen vector registers.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum Vector {
            $(
                #[doc = concat!("Vector register ", $number, ".")]
                $variant,
            )*
        }

        impl Vector {
            /// The register an instruction named, or `None` if it named
            /// something else.
            ///
            /// `None` covers the wider vector registers as well as everything
            /// that is not one at all. Emulating a move of thirty-two or
            /// sixty-four bytes would mean saving state this hypervisor has
            /// never had to think about, so such a move is refused by name
            /// rather than quietly performed sixteen bytes at a time.
            pub(crate) const fn new(register: Register) -> Option<Self> {
                Some(match register {
                    $(Register::$register => Self::$variant,)*
                    _ => return None,
                })
            }

            /// What the guest has in it.
            pub(crate) fn read(self) -> [u8; Width::Vector.bytes()] {
                let mut value = [0; Width::Vector.bytes()];
                match self {
                    $(Self::$variant => {
                        // SAFETY: `available` established that the processor
                        // will execute this rather than trap it, and the
                        // destination is sixteen writable bytes of this frame.
                        // Naming a vector register is sound because no other
                        // instruction in this image names one, so the guest's
                        // value is what is in it.
                        unsafe {
                            asm!(
                                concat!("movdqu [{at}], xmm", $number),
                                at = in(reg) value.as_mut_ptr(),
                                options(nostack, preserves_flags),
                            );
                        }
                    })*
                }
                value
            }

            /// Puts a value in it, which is the guest's register changing.
            pub(crate) fn write(self, value: [u8; Width::Vector.bytes()]) {
                match self {
                    $(Self::$variant => {
                        // SAFETY: as in `read`, with the sixteen bytes read
                        // rather than written — hence `readonly`.
                        unsafe {
                            asm!(
                                concat!("movdqu xmm", $number, ", [{at}]"),
                                at = in(reg) value.as_ptr(),
                                options(nostack, readonly, preserves_flags),
                            );
                        }
                    })*
                }
            }
        }
    };
}

vectors! {
    Xmm0 => XMM0 => 0,
    Xmm1 => XMM1 => 1,
    Xmm2 => XMM2 => 2,
    Xmm3 => XMM3 => 3,
    Xmm4 => XMM4 => 4,
    Xmm5 => XMM5 => 5,
    Xmm6 => XMM6 => 6,
    Xmm7 => XMM7 => 7,
    Xmm8 => XMM8 => 8,
    Xmm9 => XMM9 => 9,
    Xmm10 => XMM10 => 10,
    Xmm11 => XMM11 => 11,
    Xmm12 => XMM12 => 12,
    Xmm13 => XMM13 => 13,
    Xmm14 => XMM14 => 14,
    Xmm15 => XMM15 => 15,
}

/// Moves sixteen bytes out of a device in one bus transaction.
///
/// A quadword pair would not do. Two eight-byte reads are two transactions, and
/// a device that answers a sixteen-byte read need not answer two halves of one
/// the same way — so the access the guest asked for is the access that has to
/// be made. Nothing narrower than a vector register can make it, and this image
/// has no other way to name one.
///
/// The register it borrows is saved and put back inside the same block, so the
/// guest's own value is unchanged by the time this returns.
///
/// # Safety
///
/// `from` must be sixteen readable bytes of a live mapping, and reading them
/// must be something the device behind it tolerates.
pub(crate) unsafe fn read_device(from: *const u8) -> [u8; Width::Vector.bytes()] {
    let mut borrowed = [0; Width::Vector.bytes()];
    let mut value = [0; Width::Vector.bytes()];
    // SAFETY: the caller vouches for the device address. The register is saved
    // before it is used and restored after, and nothing between the two can
    // observe it: an interrupt taken here would run this image's own handler,
    // which names no vector register either.
    unsafe {
        asm!(
            "movdqu [{borrowed}], xmm0",
            "movdqu xmm0, [{from}]",
            "movdqu [{value}], xmm0",
            "movdqu xmm0, [{borrowed}]",
            borrowed = in(reg) borrowed.as_mut_ptr(),
            from = in(reg) from,
            value = in(reg) value.as_mut_ptr(),
            options(nostack, preserves_flags),
        );
    }
    value
}

/// Moves sixteen bytes into a device in one bus transaction, borrowing a vector
/// register the same way [`read_device`] does.
///
/// # Safety
///
/// `to` must be sixteen writable bytes of a live mapping, and writing them must
/// be something the device behind it tolerates.
pub(crate) unsafe fn write_device(to: *mut u8, value: [u8; Width::Vector.bytes()]) {
    let mut borrowed = [0; Width::Vector.bytes()];
    // SAFETY: as in `read_device`, with the transfer the other way round.
    unsafe {
        asm!(
            "movdqu [{borrowed}], xmm0",
            "movdqu xmm0, [{value}]",
            "movdqu [{to}], xmm0",
            "movdqu xmm0, [{borrowed}]",
            borrowed = in(reg) borrowed.as_mut_ptr(),
            value = in(reg) value.as_ptr(),
            to = in(reg) to,
            options(nostack, preserves_flags),
        );
    }
}
