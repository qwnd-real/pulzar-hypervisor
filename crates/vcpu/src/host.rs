//! Everything a processor needs in place before it can run a guest at all.
//!
//! Three things, none of which belongs to any one guest, which is why they are
//! established once per processor rather than once per virtual processor:
//! permission to use the extension, somewhere for the processor to swap the
//! host's state through, and a snapshot of the host state the world switch does
//! not restore by itself.
//!
//! # Two pages, not one
//!
//! The address in `VM_HSAVE_PA` names a page the processor writes host state to
//! on the way into a guest and reads it back from on the way out, and the
//! architecture is explicit that its format is the implementation's business:
//! software must not rely on the contents or the layout. `VMSAVE` and `VMLOAD`,
//! by contrast, use a control block's state-save area, which is architectural.
//!
//! Those two happen to be the same size and could be made the same page — which
//! would be relying on a processor not to write the part of it the other
//! instruction reads, and that is precisely the reliance the architecture rules
//! out. One extra frame per processor buys the whole question away.
//!
//! # The snapshot, and the invariant it costs
//!
//! `#VMEXIT` restores the host's segment selectors but not `FS`, `GS`, `TR` or
//! `LDTR`, and not `KernelGsBase` or the fast-system-call registers; only
//! `VMLOAD` brings those back. The hypervisor cannot run without them — one
//! read through the `GS` base is how a processor answers which processor it is,
//! and the task register is what an interrupt taken on a switched stack needs —
//! so every exit must reload them.
//!
//! It need not re-save them. On a pulzar processor none of that state changes
//! after the processor has installed its descriptor tables and attached:
//! nothing executes `swapgs`, nothing reloads the task register or the local
//! descriptor table, and nothing writes a fast-system-call register. So
//! [`Host::install`] takes the snapshot once and the world switch reloads it
//! forever after, which takes a long-latency instruction out of every exit.
//!
//! That is an invariant rather than a fact about the hardware, and it is stated
//! as a precondition of [`Host::install`]: anything that ever does change one
//! of those registers on the host side has to snapshot again, or every
//! subsequent exit will restore a value that is no longer true.

use log::info;
use paging::{DirectMap, Frames};
use processor::{Svm, SvmFeatures};
use svm::{
    Vmcb,
    msr::{HostSaveAddress, TSC_RATIO, TscRatio, VM_CR, VM_HSAVE_PA, VmCr},
};
use x86_64::{
    PhysAddr,
    registers::{
        control::{Efer, EferFlags},
        model_specific::Msr,
    },
};

use crate::VcpuError;

/// The host's side of a world switch, on one processor.
///
/// One of these belongs to the processor that installed it and to no other. The
/// snapshot in it is that processor's `GS` base and task register, so restoring
/// it anywhere else would give a processor another one's identity — and
/// [`Host::install`] hands each processor its own and publishes it nowhere, so
/// the only way to reach one is to be the processor that made it. A [`Vcpu`]
/// holding one is not [`Send`], which is what keeps it that way.
///
/// [`Vcpu`]: crate::Vcpu
#[derive(Clone, Copy, Debug)]
pub struct Host {
    hsave: PhysAddr,
    snapshot: PhysAddr,
    svm: Svm,
}

impl Host {
    /// Turns the extension on for the calling processor and sets up both pages.
    ///
    /// In this order, and none of it is interchangeable: the extension has to
    /// be enabled before `VMSAVE` will execute at all rather than raise an
    /// invalid opcode, and the snapshot has to be taken while the host
    /// state it captures is the state this processor will keep.
    ///
    /// # Errors
    ///
    /// [`VcpuError::NoSvm`] on a processor without the extension,
    /// [`VcpuError::SvmDisabled`] or [`VcpuError::SvmLocked`] where firmware
    /// turned it off — the two differ in whether a key exists that could turn
    /// it back on, and only the second is something a firmware setting can
    /// fix — [`VcpuError::TooFewAsids`] on a processor that can tag no
    /// guest, [`VcpuError::OutOfFrames`] if the chunk cannot spare two
    /// pages, or [`VcpuError::Unreachable`] if the window does not reach
    /// one of them.
    ///
    /// # Safety
    ///
    /// The calling processor must already have installed its descriptor tables
    /// and attached, so that the task register and the `GS` base hold what this
    /// processor will keep using. Nothing may afterwards change `FS`, `GS`,
    /// `TR`, `LDTR`, `KernelGsBase`, `STAR`, `LSTAR`, `CSTAR`, `SFMASK` or any
    /// `SYSENTER` register on the host side without snapshotting again: every
    /// exit restores this snapshot, and a stale one is restored just as
    /// faithfully as a current one.
    ///
    /// This must be called at most once per processor. A second call would leak
    /// the first pair of pages and re-point the processor at the second while
    /// the first may still be named by a control block in flight.
    pub unsafe fn install(
        frames: &mut Frames,
        window: DirectMap,
    ) -> Result<&'static Self, VcpuError> {
        let svm = processor::svm().ok_or(VcpuError::NoSvm)?;
        if svm.asids < 2 {
            return Err(VcpuError::TooFewAsids { asids: svm.asids });
        }
        permitted(&svm)?;

        // SAFETY: the processor reports the extension and firmware has not
        // disabled it, so the bit is writable; enabling it changes nothing about
        // how any instruction this image already executes behaves.
        unsafe { Efer::update(|efer| efer.insert(EferFlags::SECURE_VIRTUAL_MACHINE_ENABLE)) };
        if !Efer::read().contains(EferFlags::SECURE_VIRTUAL_MACHINE_ENABLE) {
            return Err(VcpuError::SvmDisabled);
        }
        if svm.features.contains(SvmFeatures::TSC_RATE_MSR) {
            // SAFETY: the feature bit establishes that `TSC_RATIO` exists on
            // this processor, and one is a valid ratio that leaves the guest's
            // counter frequency unchanged.
            unsafe { Msr::new(TSC_RATIO).write(TscRatio::ONE.into_bits()) };
        }

        let hsave = page(frames, window)?;
        let address = HostSaveAddress::new(hsave).ok_or(VcpuError::Unreachable {
            phys: hsave.as_u64(),
        })?;
        // SAFETY: the address is page-aligned and non-zero — `HostSaveAddress`
        // refuses anything else — and names a frame of the reserved chunk, which
        // firmware described as memory and so lies below the largest physical
        // address this processor implements.
        unsafe { Msr::new(VM_HSAVE_PA).write(address.into_bits()) };

        let snapshot = page(frames, window)?;
        // SAFETY: the extension is enabled and this runs at ring 0, so `VMSAVE`
        // executes rather than faulting; `snapshot` is a page-aligned,
        // write-back frame of the chunk that nothing else names; and the caller
        // guarantees the state being captured is this processor's settled state.
        unsafe { core::arch::asm!("vmsave rax", in("rax") snapshot.as_u64(), options(nostack)) };

        Ok(&*alloc::boxed::Box::leak(alloc::boxed::Box::new(Self {
            hsave,
            snapshot,
            svm,
        })))
    }

    /// The control block a world switch restores this processor's own state
    /// from.
    #[must_use]
    pub const fn snapshot(&self) -> PhysAddr {
        self.snapshot
    }

    /// What this processor's extension can do, which is what decides how a
    /// guest's control block may be programmed.
    #[must_use]
    pub const fn svm(&self) -> Svm {
        self.svm
    }

    /// Logs what was established, and on what kind of processor.
    pub fn describe(&self, who: &str) {
        self.svm.describe(who);
        info!(
            "{who}: svm enabled, host state at {:#x}, snapshot at {:#x}",
            self.hsave, self.snapshot,
        );
    }
}

/// Whether firmware is willing to let this machine use the extension.
///
/// A processor can report the extension in its feature leaves and still refuse
/// every guest, because firmware disabled it — and reading the control register
/// back is the only way to tell that from a processor that simply cannot. The
/// two failures below are different problems for whoever is trying to boot: one
/// may be liftable with a key, the other needs a firmware setting changed by
/// hand.
fn permitted(svm: &Svm) -> Result<(), VcpuError> {
    // SAFETY: the processor reports the extension, so this register exists.
    // Reading it has no side effects.
    let vm_cr = VmCr::from_bits(unsafe { Msr::new(VM_CR).read() });
    if !vm_cr.svm_disabled() {
        return Ok(());
    }
    if svm.features.contains(SvmFeatures::SVM_LOCK) {
        Err(VcpuError::SvmLocked)
    } else {
        Err(VcpuError::SvmDisabled)
    }
}

/// A zeroed, page-aligned frame of the chunk the window can reach.
///
/// Both pages are handed to the processor by physical address and read by it
/// directly, so being reachable through the window is not what the hardware
/// needs — it is what lets anything here have written the zeroes first.
fn page(frames: &mut Frames, window: DirectMap) -> Result<PhysAddr, VcpuError> {
    let frame = frames
        .allocate(0)
        .map_err(|_| VcpuError::OutOfFrames)?
        .start_address();
    if window.virt(frame).is_none() {
        return Err(VcpuError::Unreachable {
            phys: frame.as_u64(),
        });
    }
    Ok(frame)
}

const _: () = assert!(
    size_of::<Vmcb>() == svm::PAGE_BYTES,
    "both pages are one frame, which is what makes a single allocation enough",
);
