//! The hardware side of these tables: four levels of page table, and how a
//! guest physical address is walked down through them.
//!
//! What an address *means* is [`crate::map`]'s to answer. Everything here is
//! the machinery that writes such an answer where the processor's page walker
//! will find it, and it knows nothing about why an address means what it does.

pub(crate) mod walk;
