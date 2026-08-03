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
//! # Why each access is a call and not an assembly block
//!
//! A register the compiler cannot see is still a register the compiler has
//! rules about. An `asm!` block naming `xmm3` in its template and nowhere in
//! its operands tells rustc nothing about the register it reads or writes:
//! templates are opaque, so the effect is invisible and the code is outside the
//! language's defined semantics whatever the target's feature string says.
//! Being outside it happens to work today, which is the worst place for a
//! hypervisor to be.
//!
//! So every access crosses a real function boundary with a declared calling
//! convention. The System V convention makes all sixteen vector registers
//! caller-saved, so a call to one of these routines is *already* a call that
//! may clobber every one of them as far as rustc is concerned — the effect is
//! declared, by the ABI, at each call site. What the routine does inside is its
//! own business, which is what `naked` means: no prologue, no epilogue, nothing
//! the compiler generated, and so nothing for it to have assumed.
//!
//! The target's feature string is still load-bearing — it is why the value read
//! is the *guest's* rather than something the compiler spilled there — but it
//! is no longer doing the soundness argument's work on its own.
//!
//! # Nothing here reaches a device
//!
//! A vector move against a device register is performed by moving the bytes
//! between here and a buffer, and separately between that buffer and the
//! device. There used to be a pair of routines that borrowed `XMM0` to carry
//! sixteen bytes to a device in one instruction, and they are gone: one
//! assembly block is not exception-atomic, so an interrupt or a fault taken
//! between borrowing the guest's register and restoring it would abandon guest
//! state that has no other copy. A device that needs sixteen bytes in a single
//! bus transaction cannot be served without that borrow, and so is refused by
//! name — see [`Capability`](crate::mmio::Capability).

use core::arch::naked_asm;

use iced_x86::Register;
use x86_64::registers::control::{Cr0, Cr0Flags, Cr4, Cr4Flags};

use crate::{EmulateError, value::WIDEST};

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
        // handle vector state, which for this one means the routines in this
        // module and nothing else. It changes no translation and no protection,
        // and control register four is host state that `VMRUN` swaps out, so no
        // guest observes it.
        unsafe { Cr4::update(|flags| flags.insert(Cr4Flags::OSFXSR)) };
    }
    Ok(())
}

/// Declares the sixteen registers once, and everything that is one arm apiece.
///
/// A register number is not something an instruction can take as an operand, so
/// selecting one at runtime is a match however it is written. Writing the match
/// out by hand would be two of them, sixteen arms each, with the register name
/// repeated in an assembly string every time. This generates both from one
/// list, which is also what makes [`Vector`] an enum rather than a checked
/// integer — and an enum is what lets each match be exhaustive with no arm for
/// a number that cannot happen.
macro_rules! vectors {
    ($($variant:ident => $register:ident => $number:literal,)*) => {
        /// One of the processor's sixteen vector registers.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
        pub(crate) enum Vector {
            $(
                #[doc = concat!("Vector register ", $number, ".")]
                $variant,
            )*
        }

        impl Vector {
            /// Every one of them, in the architecture's own order.
            ///
            /// Exhaustive by construction: a register added to the list above
            /// appears here too, which is what lets a test walk all sixteen and
            /// still be complete.
            ///
            /// Nothing in the hypervisor iterates the registers — a move names
            /// exactly one — so this exists for the tests and is built only for
            /// them.
            #[cfg(test)]
            pub(crate) const ALL: [Self; 16] = [$(Self::$variant,)*];

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

            /// Which register this is, as the architecture numbers them.
            ///
            /// The hypervisor never needs this: a register number is not
            /// something an instruction can take as an operand, which is why the
            /// accesses below are a match rather than an index. It is how the
            /// test register file — which really is an array — finds its row.
            #[cfg(test)]
            pub(crate) const fn number(self) -> u8 {
                match self {
                    $(Self::$variant => $number,)*
                }
            }
        }

        /// What the guest has in one of its vector registers.
        ///
        /// The value is the guest's because nothing else in this image can have
        /// put anything there: the target emits no vector instruction, and the
        /// only ones that exist are here.
        pub(crate) fn read(register: Vector) -> [u8; WIDEST] {
            let mut value = [0; WIDEST];
            match register {
                $(Vector::$variant => {
                    /// Copies this register to sixteen bytes at the first
                    /// argument, which the convention puts in `RDI`.
                    #[unsafe(naked)]
                    unsafe extern "sysv64" fn store(_into: *mut u8) {
                        naked_asm!(concat!("movdqu [rdi], xmm", $number), "ret")
                    }
                    // SAFETY: `available` established that the processor will
                    // execute a vector move rather than trap it, and the
                    // destination is sixteen writable bytes of this frame that
                    // nothing else borrows. The call itself declares — through
                    // a convention in which every vector register is
                    // caller-saved — that vector state may change across it.
                    unsafe { store(value.as_mut_ptr()) };
                })*
            }
            value
        }

        /// Puts a value in one of them, which is the guest's register changing.
        pub(crate) fn write(register: Vector, value: [u8; WIDEST]) {
            match register {
                $(Vector::$variant => {
                    /// Loads this register from sixteen bytes at the first
                    /// argument.
                    #[unsafe(naked)]
                    unsafe extern "sysv64" fn load(_from: *const u8) {
                        naked_asm!(concat!("movdqu xmm", $number, ", [rdi]"), "ret")
                    }
                    // SAFETY: as in `read`, with the sixteen bytes read rather
                    // than written — and read out of a local this call cannot
                    // outlive.
                    unsafe { load(value.as_ptr()) };
                })*
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

#[cfg(test)]
mod tests {
    use iced_x86::Register;

    use super::Vector;

    #[test]
    fn every_register_is_listed_once_and_numbered_as_the_architecture_does() {
        assert_eq!(Vector::ALL.len(), 16);
        for (number, register) in Vector::ALL.into_iter().enumerate() {
            assert_eq!(
                u32::from(register.number()),
                u32::try_from(number).expect("sixteen fits"),
                "{register:?} must carry its own architectural number"
            );
        }
    }

    #[test]
    fn the_sixteen_legacy_registers_are_the_ones_recognized() {
        let named = [
            Register::XMM0,
            Register::XMM1,
            Register::XMM2,
            Register::XMM3,
            Register::XMM4,
            Register::XMM5,
            Register::XMM6,
            Register::XMM7,
            Register::XMM8,
            Register::XMM9,
            Register::XMM10,
            Register::XMM11,
            Register::XMM12,
            Register::XMM13,
            Register::XMM14,
            Register::XMM15,
        ];
        for (register, expected) in named.into_iter().zip(Vector::ALL) {
            assert_eq!(Vector::new(register), Some(expected));
        }
    }

    #[test]
    fn nothing_else_is_a_register_this_crate_reaches() {
        // The wider vector registers are refused rather than truncated to their
        // low sixteen bytes: a move of thirty-two would leave half the
        // destination holding whatever it held before.
        for register in [
            Register::XMM16,
            Register::XMM31,
            Register::YMM0,
            Register::YMM15,
            Register::ZMM0,
            Register::ZMM31,
            Register::MM0,
            Register::MM7,
            Register::RAX,
            Register::EAX,
            Register::AH,
            Register::CS,
            Register::CR0,
            Register::DR7,
            Register::K1,
            Register::BND0,
            Register::TMM0,
            Register::RIP,
            Register::None,
        ] {
            assert_eq!(
                Vector::new(register),
                None,
                "{register:?} is not one of the sixteen this crate reaches"
            );
        }
    }
}
