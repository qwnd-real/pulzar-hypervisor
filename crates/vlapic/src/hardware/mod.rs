//! What reaches real silicon.
//!
//! The controller's register file is emulated in full; the hardware behind it is
//! not emulated at all. Its sources are the real sources, its timer is the real
//! timer, and the model it presents is the real processor's — because a
//! thermal sensor is a thermal sensor, a timer is a decrementing register, and a
//! guest that reads its own `CPUID` would catch any invention immediately.
//!
//! So this is the surface where the guest's registers become physical ones. Every
//! module here answers the same question about a different piece of hardware:
//! given what the guest has programmed, what should the real register hold — and
//! what must never be carried across, because carrying it would put the *host*
//! somewhere the guest chose.

pub(crate) mod mirror;
pub(crate) mod model;
pub(crate) mod sources;
pub(crate) mod timer;
