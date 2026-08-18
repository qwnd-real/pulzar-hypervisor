//! Host-side task runner for the pulzar workspace.
//!
//! Builds the UEFI boot media and drives QEMU so contributors never manage
//! artifacts by hand: the EFI system partition is staged as a plain directory
//! under `dist/esp/` and served to QEMU via its virtual-FAT driver, while
//! guest OS disks, OVMF firmware, and TPM state live in a per-user cache.
//! Nothing binary is ever created inside the repository, and every artifact
//! is reproducible from a single command.

mod disk;
mod esp;
mod paths;
mod proc;
mod vm;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

/// Task runner for building pulzar boot media and running it under QEMU.
#[derive(Parser)]
#[command(bin_name = "cargo xtask")]
enum Cli {
    /// Build the loader and hypervisor image and stage the ESP directory.
    Build {
        /// Build with the release profile.
        #[arg(long)]
        release: bool,
        /// Compile every log record out of both images.
        ///
        /// For a real machine, where a byte down a 16550 costs 260 µs with
        /// interrupts held off and a board with a serial header has a working
        /// one behind it whether or not a cable is plugged in. Counters kept in
        /// memory are unaffected and still readable from a debugger.
        #[arg(long)]
        silent: bool,
    },
    /// Provision a guest OS disk image (a one-time setup step per machine).
    #[command(subcommand)]
    Disk(DiskCommand),
    /// Build everything and boot it in QEMU.
    Run {
        /// Guest OS disk to attach behind the hypervisor boot media.
        #[arg(long, value_enum, default_value_t = Guest::None)]
        os: Guest,
        /// Build with the release profile.
        #[arg(long)]
        release: bool,
        /// Expose QEMU's guest GDB stub on 127.0.0.1:1234.
        #[arg(long)]
        gdb: bool,
        /// Compile every log record out of both images, as `build --silent`.
        #[arg(long)]
        silent: bool,
        /// Give the guest no debug console and no serial port at all.
        ///
        /// What a machine with no serial header presents, which is the machine
        /// this hypervisor has to run on: every log record is still in the
        /// image and still decides whether it has anything to say, and
        /// the answer has nowhere to go. The one configuration in which
        /// logging cannot be what changed the timing — `--silent`
        /// removes the records instead, which is a different question.
        #[arg(long, conflicts_with = "serial_log")]
        no_console: bool,
        /// Capture a guest log output to a file, created fresh each run.
        /// Repeatable: the first use maps to the debug console, which is where
        /// the guest logs, and further uses to COM1 upwards. Without it, the
        /// debug console goes to stdio.
        #[arg(long, value_name = "FILE")]
        serial_log: Vec<PathBuf>,
        /// Boot the guest disk directly, with no hypervisor in front of it.
        ///
        /// The same machine, the same disk and the same processor topology,
        /// with pulzar taken out — which is the only way to tell a fault of the
        /// hypervisor's from one the guest has on this hardware anyway. Nothing
        /// is built, because nothing of pulzar's is used.
        #[arg(long)]
        no_hypervisor: bool,
    },
}

/// Guest OS disk provisioning subcommands.
#[derive(Subcommand)]
enum DiskCommand {
    /// Fetch a pre-installed Debian image (no installer, no interaction).
    Linux {
        /// Recreate the disk even if it already exists.
        #[arg(long)]
        force: bool,
    },
    /// Fetch the `CachyOS` ISO and run its installer (interactive, once).
    Cachyos {
        /// Recreate the disk (and its firmware state) even if it exists.
        #[arg(long)]
        force: bool,
    },
    /// Create a blank disk and run the Windows installer from an ISO
    /// (interactive, once; the ISO must be supplied by you).
    Windows {
        /// Path to a Windows 11 installation ISO.
        #[arg(long)]
        iso: PathBuf,
        /// Recreate the disk (and its firmware/TPM state) even if it exists.
        #[arg(long)]
        force: bool,
    },
}

/// Guest OS selection for `run`.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Guest {
    /// Attach the Debian disk created by `disk linux`.
    Linux,
    /// Attach the `CachyOS` disk created by `disk cachyos`.
    #[value(name = "cachyos")]
    Cachyos,
    /// Attach the Windows disk created by `disk windows`.
    Windows,
    /// Boot the hypervisor media alone, with no guest OS disk.
    None,
}

impl Guest {
    /// Stable name used for the disk image and associated firmware/TPM state.
    fn label(self) -> &'static str {
        match self {
            Self::Linux => "linux",
            Self::Cachyos => "cachyos",
            Self::Windows => "windows",
            Self::None => "none",
        }
    }
}

fn main() -> Result<()> {
    match Cli::parse() {
        Cli::Build { release, silent } => esp::stage(release, silent).map(|_| ()),
        Cli::Disk(DiskCommand::Linux { force }) => disk::linux(force),
        Cli::Disk(DiskCommand::Cachyos { force }) => disk::cachyos(force),
        Cli::Disk(DiskCommand::Windows { iso, force }) => disk::windows(&iso, force),
        Cli::Run {
            os,
            release,
            gdb,
            silent,
            no_console,
            serial_log,
            no_hypervisor,
        } => {
            // Three ways of asking, and they collapse to one answer here so that
            // nothing downstream has to hold both a flag and a list and decide
            // which of them means what.
            let console = match (no_console, serial_log) {
                (true, _) => vm::Console::None,
                (false, files) if files.is_empty() => vm::Console::Stdio,
                (false, files) => vm::Console::Files(files),
            };
            let layering = if no_hypervisor {
                vm::Layering::Bare
            } else {
                vm::Layering::Hypervisor
            };
            vm::run(os, release, gdb, silent, console, layering)
        }
    }
}
