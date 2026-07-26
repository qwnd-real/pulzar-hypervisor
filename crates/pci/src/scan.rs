//! Finding every function the machine has, and what sits above each one.
//!
//! # Why every bus is swept rather than descended
//!
//! The obvious enumeration is recursive: read bus zero, and wherever a bridge
//! is found, descend into the bus it says is below it. It is also wrong on any
//! machine with more than one host bridge. A second root bus is not below
//! anything — no bridge points at it, because it *is* the top of a hierarchy —
//! and the only thing that describes it is the ACPI namespace, which needs an
//! interpreter for a bytecode this hypervisor has no other reason to implement.
//!
//! So every bus of every aperture is read instead, and the tree is rebuilt
//! afterwards from the bus numbers the bridges themselves carry. This costs
//! nothing: a bus has to be mapped before it can be found empty, so the buses a
//! recursive walk would have skipped are exactly the ones whose cost was
//! already paid. What it buys is a machine with four host bridges enumerating
//! as completely as a machine with one, no recursion depth on a deep switch
//! tree, and no need to guard against a bridge that points at its own bus.
//!
//! Sweeping is only safe because it stays inside the apertures. An entry in the
//! firmware table is a promise that the root complex decodes that range; an
//! address outside every entry is not, and is never read.
//!
//! # Two mappings per bus, for two different lifetimes
//!
//! Probing a bus needs the whole megabyte of it, because a function that is
//! absent has to be readable to be found absent. Keeping a bus needs only the
//! four kilobytes of each function that turned out to exist.
//!
//! Those are very different amounts. A machine with 200 populated buses would
//! spend 200 MiB of a 1 GiB mapping window on the first and 8 MiB on the
//! second, and the window also has to hold a stack for every processor. So the
//! megabyte is transient — mapped, read, and released before the next bus — and
//! only the pages of functions that exist are kept.
//!
//! The transient mapping is read-only, which is what makes "the survey never
//! writes to a device" a property of the page tables rather than a claim about
//! this code. The kept mappings are writable, because taking the devices over
//! later is what they are kept for.
//!
//! # Buses are swept in order, and that is load-bearing
//!
//! A bridge's secondary bus is always numbered above the bus the bridge is on,
//! so sweeping in ascending order means the port above a bus has already been
//! found by the time that bus is read. Two decisions need exactly that: whether
//! the port forwards the alternative routing interpretation, which changes how
//! the functions of a device are counted, and whether the port is a bridge down
//! onto plain PCI, below which extended configuration space does not answer.

use alloc::vec::Vec;

use log::warn;
use paging::{AddressSpace, CacheType, Mapping, Protection};
use x86_64::{PhysAddr, VirtAddr};

use crate::{
    PciError, Segment,
    access::{Aperture, Config, Reach, Sweep},
    address::{self, Address, Bus, DEVICES, FUNCTIONS},
    bar::{self, Bar, Rom, SLOTS},
    capability::{self, Capabilities},
    express::{Express, Role},
    extended::{Acs, Aer, Ari, ResizableBars, Sriov, serial_number},
    header::{self, Bridge, Class, Command, Common, Layout, Status},
    msi::{self, Msi, MsiX},
};

/// Where one function sits in the collection everything else indexes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Index(u32);

impl Index {
    /// The position itself.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The position as a subscript.
    const fn slot(self) -> usize {
        self.0 as usize
    }
}

/// The top of one bus hierarchy: a bus that no bridge claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Root {
    segment: Segment,
    bus: Bus,
}

impl Root {
    /// The segment group the hierarchy is in.
    #[must_use]
    pub const fn segment(&self) -> Segment {
        self.segment
    }

    /// The bus at the top of it.
    #[must_use]
    pub const fn bus(&self) -> Bus {
        self.bus
    }
}

/// Everything one function said about itself.
///
/// Flat and `Copy` on purpose: what eventually reads these is a nested paging
/// fault, which can neither allocate nor fail, so a record that owned a
/// collection would be the wrong shape however convenient it looked here.
#[derive(Clone, Copy, Debug)]
pub struct Function {
    address: Address,
    phys: Option<PhysAddr>,
    virt: Option<VirtAddr>,
    extended: bool,
    vendor: u16,
    device: u16,
    subsystem_vendor: u16,
    subsystem_device: u16,
    revision: u8,
    class: Class,
    layout: Layout,
    multifunction: bool,
    command: Command,
    status: Status,
    interrupt_line: u8,
    interrupt_pin: u8,
    power_state: u8,
    serial: Option<u64>,
    bars: [Bar; SLOTS],
    rom: Option<Rom>,
    capabilities: Capabilities,
    msi: Option<Msi>,
    msi_x: Option<MsiX>,
    express: Option<Express>,
    bridge: Option<Bridge>,
    aer: Option<Aer>,
    ari: Option<Ari>,
    acs: Option<Acs>,
    resizable: Option<ResizableBars>,
    sriov: Option<Sriov>,
    parent: Option<Index>,
}

impl Function {
    /// Which function this is.
    #[must_use]
    pub const fn address(&self) -> Address {
        self.address
    }

    /// Where its configuration space is in physical memory.
    ///
    /// `None` for a function reached through the legacy ports, which name a
    /// register without there being an address it lives at.
    #[must_use]
    pub const fn phys(&self) -> Option<PhysAddr> {
        self.phys
    }

    /// Where its configuration space is mapped, for as long as pulzar runs.
    ///
    /// `None` if the mapping window could not hold it, in which case reaching
    /// the function again means mapping it again.
    #[must_use]
    pub const fn virt(&self) -> Option<VirtAddr> {
        self.virt
    }

    /// Whether extended configuration space was reachable, and so whether the
    /// extended capabilities below are absent or merely unread.
    #[must_use]
    pub const fn extended(&self) -> bool {
        self.extended
    }

    /// Who made the silicon.
    #[must_use]
    pub const fn vendor(&self) -> u16 {
        self.vendor
    }

    /// What the silicon is.
    #[must_use]
    pub const fn device(&self) -> u16 {
        self.device
    }

    /// Who made the board, where the function says.
    #[must_use]
    pub const fn subsystem_vendor(&self) -> u16 {
        self.subsystem_vendor
    }

    /// What the board is, where the function says.
    #[must_use]
    pub const fn subsystem_device(&self) -> u16 {
        self.subsystem_device
    }

    /// The vendor's revision of the device.
    #[must_use]
    pub const fn revision(&self) -> u8 {
        self.revision
    }

    /// What kind of device it is.
    #[must_use]
    pub const fn class(&self) -> Class {
        self.class
    }

    /// Which shape its header has.
    #[must_use]
    pub const fn layout(&self) -> Layout {
        self.layout
    }

    /// Whether its device has more than one function.
    #[must_use]
    pub const fn multifunction(&self) -> bool {
        self.multifunction
    }

    /// What it is allowed to do on its bus.
    #[must_use]
    pub const fn command(&self) -> Command {
        self.command
    }

    /// What it has noticed going wrong.
    #[must_use]
    pub const fn status(&self) -> Status {
        self.status
    }

    /// Which interrupt line firmware wired its pin to.
    #[must_use]
    pub const fn interrupt_line(&self) -> u8 {
        self.interrupt_line
    }

    /// Which of the four pins it asserts, or zero for none.
    #[must_use]
    pub const fn interrupt_pin(&self) -> u8 {
        self.interrupt_pin
    }

    /// Which power state it is in. Anything but zero means its memory
    /// registers do not answer.
    #[must_use]
    pub const fn power_state(&self) -> u8 {
        self.power_state
    }

    /// Whether it answers memory accesses at all, which every region below
    /// depends on.
    #[must_use]
    pub const fn decoding(&self) -> bool {
        self.command.memory_space() && self.power_state == 0
    }

    /// Its serial number, if it has the capability that carries one.
    #[must_use]
    pub const fn serial(&self) -> Option<u64> {
        self.serial
    }

    /// Its base address registers.
    #[must_use]
    pub const fn bars(&self) -> &[Bar; SLOTS] {
        &self.bars
    }

    /// Its expansion ROM, if it has one.
    #[must_use]
    pub const fn rom(&self) -> Option<Rom> {
        self.rom
    }

    /// Where each capability it has was found.
    #[must_use]
    pub const fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    /// Its older message interrupt capability.
    #[must_use]
    pub const fn msi(&self) -> Option<Msi> {
        self.msi
    }

    /// Its newer message interrupt capability, which is where a vector table's
    /// physical address comes from.
    #[must_use]
    pub const fn msi_x(&self) -> Option<MsiX> {
        self.msi_x
    }

    /// What it is in a PCI Express hierarchy.
    #[must_use]
    pub const fn express(&self) -> Option<Express> {
        self.express
    }

    /// What it bridges, if it is a bridge.
    #[must_use]
    pub const fn bridge(&self) -> Option<Bridge> {
        self.bridge
    }

    /// What it has reported through advanced error reporting.
    #[must_use]
    pub const fn aer(&self) -> Option<Aer> {
        self.aer
    }

    /// How its device numbers its functions, when there are more than eight.
    #[must_use]
    pub const fn ari(&self) -> Option<Ari> {
        self.ari
    }

    /// What it will let the devices below it do to each other.
    #[must_use]
    pub const fn acs(&self) -> Option<Acs> {
        self.acs
    }

    /// Which of its base address registers can be resized.
    #[must_use]
    pub const fn resizable(&self) -> Option<ResizableBars> {
        self.resizable
    }

    /// The virtual functions it can conjure.
    #[must_use]
    pub const fn sriov(&self) -> Option<Sriov> {
        self.sriov
    }

    /// The bridge immediately above it, or `None` if it is on a root bus.
    #[must_use]
    pub const fn parent(&self) -> Option<Index> {
        self.parent
    }

    /// Access to this function's configuration space through the mapping that
    /// was kept for it.
    ///
    /// # Errors
    ///
    /// [`PciError::Unmapped`] if no mapping was kept, which the log will have
    /// said more about at the time.
    pub(crate) fn config(&self) -> Result<Config, PciError> {
        let virt = self.virt.ok_or(PciError::Unmapped {
            address: self.address,
        })?;
        // SAFETY: the mapping this address came from is owned by the `Pci` in
        // the crate's global and is never released, so it is live for as long as
        // anything can hold a reference to this function.
        Ok(unsafe { Config::new(self.address, Reach::Mapped(virt)) })
    }
}

/// Every function the machine has, and how they sit under one another.
#[derive(Debug, Default)]
pub struct Topology {
    functions: Vec<Function>,
    regions: Vec<Owned>,
    roots: Vec<Root>,
}

impl Topology {
    /// Every function, ordered as a machine is conventionally listed.
    #[must_use]
    pub fn functions(&self) -> &[Function] {
        &self.functions
    }

    /// How many there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.functions.len()
    }

    /// Whether the machine turned out to have none at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }

    /// The buses nothing claims, which is where each hierarchy starts.
    #[must_use]
    pub fn roots(&self) -> &[Root] {
        &self.roots
    }

    /// The function at `index`.
    #[must_use]
    pub fn get(&self, index: Index) -> Option<&Function> {
        self.functions.get(index.slot())
    }

    /// The function at `address`.
    #[must_use]
    pub fn find(&self, address: Address) -> Option<&Function> {
        self.functions
            .binary_search_by_key(&address, Function::address)
            .ok()
            .and_then(|slot| self.functions.get(slot))
    }

    /// Every function immediately below the bridge at `index`.
    pub fn children(&self, index: Index) -> impl Iterator<Item = &Function> {
        self.functions
            .iter()
            .filter(move |function| function.parent == Some(index))
    }

    /// The function whose registers `phys` falls in.
    ///
    /// Only the regions whose extent is known without writing to a device are
    /// indexed, which today means the two that message interrupts use. A base
    /// address register reports where it answers but not how far, so its range
    /// joins this index when sizing runs and not before.
    #[must_use]
    pub fn owner(&self, phys: PhysAddr) -> Option<&Function> {
        let target = phys.as_u64();
        let above = self.regions.partition_point(|owned| owned.start <= target);
        self.regions[..above]
            .iter()
            .rev()
            .find(|owned| owned.last >= target)
            .and_then(|owned| self.get(owned.owner))
    }

    /// How many physical ranges are indexed.
    #[must_use]
    pub fn regions(&self) -> usize {
        self.regions.len()
    }
}

/// One physical range a function owns, as the owner index holds it.
#[derive(Clone, Copy, Debug)]
struct Owned {
    start: u64,
    last: u64,
    owner: Index,
}

/// What one sweep of the machine produced.
#[derive(Debug, Default)]
pub(crate) struct Survey {
    pub(crate) topology: Option<Topology>,
    pub(crate) mappings: Vec<Mapping>,
    pub(crate) buses: u32,
    pub(crate) unreachable: u32,
    pub(crate) unmapped: u32,
    pub(crate) leaked: u32,
}

/// Reads every bus of every aperture, and every segment-zero bus no aperture
/// covers.
///
/// The second part is what keeps a machine with no firmware table at all from
/// enumerating as empty: with no apertures, every bus falls into it and the
/// whole machine is read through the legacy ports.
///
/// # Errors
///
/// Whatever the configuration mechanism reported. Every failure that belongs to
/// the machine rather than to this crate — an aperture that cannot be mapped, a
/// bus whose pages the window would not hold — is a warning and a counter
/// instead.
pub(crate) fn sweep(space: &mut AddressSpace, apertures: &[Aperture]) -> Result<Survey, PciError> {
    let mut survey = Survey::default();
    let mut functions = Vec::new();
    for aperture in apertures {
        for number in aperture.first_bus().get()..=aperture.last_bus().get() {
            let bus = Bus::new(number);
            let Some(phys) = aperture.bus_base(bus) else {
                continue;
            };
            mapped(
                space,
                &mut survey,
                &mut functions,
                aperture.segment(),
                bus,
                phys,
            )?;
        }
    }
    for number in 0..=u8::MAX {
        let bus = Bus::new(number);
        if apertures
            .iter()
            .any(|aperture| aperture.covers(Segment::ZERO, bus))
        {
            continue;
        }
        survey.buses += 1;
        let found = probe(&functions, Sweep::Ports, Segment::ZERO, bus, None)?;
        functions.extend(found);
    }

    survey.topology = Some(assemble(functions));
    Ok(survey)
}

/// Probes one bus through a mapping of its own, and keeps the pages of whatever
/// was on it.
fn mapped(
    space: &mut AddressSpace,
    survey: &mut Survey,
    functions: &mut Vec<Function>,
    segment: Segment,
    bus: Bus,
    phys: PhysAddr,
) -> Result<(), PciError> {
    // SAFETY: the range is one bus of an aperture firmware described and this
    // crate checked, so it is configuration space and not memory anything else
    // owns. It is mapped read-only, so no aliasing write is even expressible,
    // and uncached, because a cached read of a device's registers would answer
    // from whatever was there before.
    let mapping = match unsafe {
        space.map_physical(
            phys,
            address::BUS_BYTES,
            Protection::ReadOnly,
            CacheType::Uncached,
        )
    } {
        Ok(mapping) => mapping,
        Err(error) => {
            warn!("pci: could not map segment {segment} bus {bus} to probe it: {error}");
            survey.unreachable += 1;
            return Ok(());
        }
    };

    let found = probe(
        functions,
        Sweep::Mapped(mapping.addr()),
        segment,
        bus,
        Some(phys),
    );
    // SAFETY: the probe has returned and nothing derived from the mapping's
    // address outlived it — a record carries physical addresses and the values
    // that were read, never a pointer into the window.
    if let Err(error) = unsafe { space.unmap(mapping) } {
        warn!("pci: could not release the probe mapping of segment {segment} bus {bus}: {error}");
        survey.leaked += 1;
    }

    survey.buses += 1;
    let mut found = found?;
    retain(space, survey, &mut found);
    functions.append(&mut found);
    Ok(())
}

/// Keeps a mapping of each function's own configuration space.
///
/// A function whose page the window would not hold is left with none. It is
/// still a function the machine has and still worth reporting; what it loses is
/// the ability to be reached again without mapping it afresh.
fn retain(space: &mut AddressSpace, survey: &mut Survey, found: &mut [Function]) {
    for function in found {
        let Some(phys) = function.phys else {
            continue;
        };
        // SAFETY: as in `mapped`, for the four kilobytes of one function. This
        // one is writable because taking the device over later is what it is
        // kept for, and nothing else in this image maps configuration space.
        match unsafe {
            space.map_physical(
                phys,
                address::FUNCTION_BYTES,
                Protection::ReadWrite,
                CacheType::Uncached,
            )
        } {
            Ok(mapping) => {
                function.virt = Some(mapping.addr());
                survey.mappings.push(mapping);
            }
            Err(error) => {
                warn!(
                    "pci: could not keep a mapping for {}: {error}",
                    function.address
                );
                survey.unmapped += 1;
            }
        }
    }
}

/// Reads every function of one bus.
///
/// `known` is everything found on lower-numbered buses, which is where the two
/// facts about the port above this bus come from.
fn probe(
    known: &[Function],
    sweep: Sweep,
    segment: Segment,
    bus: Bus,
    base: Option<PhysAddr>,
) -> Result<Vec<Function>, PciError> {
    let above = port_above(known, segment, bus);
    let hidden = above.is_some_and(|port| {
        port.express
            .is_some_and(|express| express.role().hides_extended_space())
    });
    let flat = above.is_some_and(|port| port.express.is_some_and(|express| express.ari_enabled()));

    let mut found = Vec::new();
    if flat {
        // Under the alternative routing interpretation there are no devices,
        // only 256 function numbers, and a gap in them says nothing about what
        // comes after. The numbers still land on the same addresses, so the
        // sweep is the same one with the gating taken out.
        for number in 0..u16::from(DEVICES) * u16::from(FUNCTIONS) {
            let device = u8::try_from(number / u16::from(FUNCTIONS)).unwrap_or_default();
            let function = u8::try_from(number % u16::from(FUNCTIONS)).unwrap_or_default();
            let at = Address::at(segment, bus, device, function);
            if let Some(function) = read(sweep, at, base, hidden)? {
                found.push(function);
            }
        }
        return Ok(found);
    }

    for device in 0..DEVICES {
        let at = Address::at(segment, bus, device, 0);
        // Function zero gates the device. A device that does not answer at its
        // first function has no others, and probing them anyway is how phantom
        // functions get invented on hardware that aliases its decode.
        let Some(first) = read(sweep, at, base, hidden)? else {
            continue;
        };
        let more = first.multifunction;
        found.push(first);
        if !more {
            continue;
        }
        for number in 1..FUNCTIONS {
            if let Some(function) = read(sweep, at.with_function(number), base, hidden)? {
                found.push(function);
            }
        }
    }
    Ok(found)
}

/// The bridge immediately above `bus`, among the functions already found.
///
/// Sound because buses are swept in ascending order and a bridge's secondary
/// bus is always numbered above the bridge's own, so anything that could be the
/// port above this bus has already been read.
fn port_above(known: &[Function], segment: Segment, bus: Bus) -> Option<&Function> {
    known.iter().find(|function| {
        function.address.segment() == segment
            && function
                .bridge
                .is_some_and(|bridge| bridge.routes() && bridge.secondary() == bus)
    })
}

/// Reads one function, or finds that it is not there.
fn read(
    sweep: Sweep,
    address: Address,
    base: Option<PhysAddr>,
    hidden: bool,
) -> Result<Option<Function>, PciError> {
    // SAFETY: for a mapped sweep this address is inside the megabyte the caller
    // has mapped for this bus, because `within_bus` cannot exceed it; the
    // mapping is live for the whole of this call and nothing derived from it
    // escapes. For a port sweep there is no address at all.
    let config = unsafe { Config::new(address, sweep.reach(address)) };
    let vendor = config.u16(header::VENDOR)?;
    if header::ABSENT.contains(&vendor) {
        return Ok(None);
    }

    let common = Common::decode(&config)?;
    let extended = config.reaches_extended() && !hidden;
    let capabilities = capability::walk(
        &config,
        common.layout,
        common.status.capabilities(),
        extended,
    )?;
    let bars = bar::decode(&config, header::BARS, common.layout.bars())?;
    let express = capabilities
        .express()
        .map(|at| Express::decode(&config, at))
        .transpose()?;
    let rooted = express
        .is_some_and(|express| matches!(express.role(), Role::RootPort | Role::EventCollector));

    Ok(Some(Function {
        address,
        phys: base.map(|base| base + address.within_bus()),
        virt: None,
        extended,
        vendor: common.vendor,
        device: common.device,
        subsystem_vendor: common.subsystem_vendor,
        subsystem_device: common.subsystem_device,
        revision: common.revision,
        class: common.class,
        layout: common.layout,
        multifunction: common.multifunction,
        command: common.command,
        status: common.status,
        interrupt_line: common.interrupt_line,
        interrupt_pin: common.interrupt_pin,
        power_state: capabilities
            .power_management()
            .map(|at| capability::power_state(&config, at))
            .transpose()?
            .unwrap_or_default(),
        serial: capabilities
            .serial_number()
            .map(|at| serial_number(&config, at))
            .transpose()?,
        bars,
        rom: common
            .layout
            .rom()
            .map(|at| bar::rom(&config, at))
            .transpose()?
            .flatten(),
        capabilities,
        msi: capabilities
            .msi()
            .map(|at| msi::msi(&config, at))
            .transpose()?,
        msi_x: capabilities
            .msi_x()
            .map(|at| msi::msi_x(&config, at, &bars))
            .transpose()?,
        express,
        bridge: matches!(common.layout, Layout::Bridge)
            .then(|| Bridge::decode(&config))
            .transpose()?,
        aer: capabilities
            .advanced_error()
            .map(|at| Aer::decode(&config, at, rooted))
            .transpose()?,
        ari: capabilities
            .routing_interpretation()
            .map(|at| Ari::decode(&config, at))
            .transpose()?,
        acs: capabilities
            .access_control()
            .map(|at| Acs::decode(&config, at))
            .transpose()?,
        resizable: capabilities
            .resizable_bar()
            .map(|at| ResizableBars::decode(&config, at))
            .transpose()?,
        sriov: capabilities
            .virtualization()
            .map(|at| Sriov::decode(&config, at))
            .transpose()?,
        parent: None,
    }))
}

/// Sorts what was found and works out what sits above what.
///
/// The parent of a function is the bridge whose secondary bus is the bus the
/// function is on. That is the whole rule: a bridge's secondary bus *is* the
/// bus immediately below it, so nothing deeper needs searching and nothing
/// needs to recurse.
fn assemble(mut functions: Vec<Function>) -> Topology {
    functions.sort_unstable_by_key(Function::address);

    let mut claims = Vec::new();
    for (slot, function) in functions.iter().enumerate() {
        if let Some(bridge) = function.bridge
            && bridge.routes()
        {
            claims.push((
                function.address.segment(),
                bridge.secondary(),
                Index(index(slot)),
            ));
        }
    }
    claims.sort_unstable();
    // Two bridges claiming one bus is a machine describing itself incorrectly.
    // The first wins, which is at least a stable answer.
    claims.dedup_by_key(|claim| (claim.0, claim.1));

    for function in &mut functions {
        let key = (function.address.segment(), function.address.bus());
        function.parent = claims
            .binary_search_by_key(&key, |claim| (claim.0, claim.1))
            .ok()
            .and_then(|slot| claims.get(slot))
            .map(|claim| claim.2);
    }

    let mut roots: Vec<Root> = functions
        .iter()
        .filter(|function| function.parent.is_none())
        .map(|function| Root {
            segment: function.address.segment(),
            bus: function.address.bus(),
        })
        .collect();
    roots.sort_unstable();
    roots.dedup();

    let mut regions = Vec::new();
    for (slot, function) in functions.iter().enumerate() {
        let Some(msi_x) = function.msi_x else {
            continue;
        };
        for region in [msi_x.table(), msi_x.pending()].into_iter().flatten() {
            regions.push(Owned {
                start: region.phys().as_u64(),
                last: region.last(),
                owner: Index(index(slot)),
            });
        }
    }
    regions.sort_unstable_by_key(|owned| owned.start);

    Topology {
        functions,
        regions,
        roots,
    }
}

/// A position in the collection, as an index holds it.
///
/// Saturating rather than wrapping: a machine with four billion functions
/// cannot exist, and if one somehow did, every record past the limit pointing
/// at the last one is a wrong answer that stays inside the collection.
fn index(slot: usize) -> u32 {
    u32::try_from(slot).unwrap_or(u32::MAX)
}
