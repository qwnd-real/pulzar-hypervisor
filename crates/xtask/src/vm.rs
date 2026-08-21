//! QEMU orchestration: OVMF firmware, the virtual-FAT ESP drive, guest OS
//! disks, and the software TPM that Windows 11 requires.
//!
//! Firmware variable stores are kept per guest in the user cache, so NVRAM
//! boot entries written while an OS installs survive into later runs.

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Child, Command},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use ovmf_prebuilt::{Arch, FileType, Prebuilt, Source};

use crate::{Guest, disk, esp, paths, proc};

/// Hint attached to errors when a QEMU binary is missing from the host.
pub const QEMU_INSTALL_HINT: &str = "install QEMU (it provides qemu-system-x86_64 and qemu-img)";

/// The processors the guest is given, and how they are arranged: one socket of
/// eight cores with two threads each.
///
/// The arrangement is the point rather than the total. Two logical processors
/// sharing one physical core is a case nothing in this hypervisor's emulation
/// distinguishes — it has no notion of a core — and everything about the
/// hardware shares, so it is the one topology a guest has to be run on before
/// the emulation is believed. A flat `-smp 16` gives sixteen single-threaded
/// cores and never exercises it.
const TOPOLOGY: &str = "8,sockets=1,cores=4,threads=2";

/// The processor the guest is shown.
///
/// `topoext` is what makes the topology above mean anything: it is the bit that
/// says AMD's own topology leaf is there, and without it a guest cannot tell
/// which logical processors share a core. Linux then treats all sixteen as
/// separate cores and never brings a sibling up as one.
///
/// `invtsc` says the timestamp counter runs at a constant rate, which is what
/// lets a guest use it as a timebase rather than as a cycle counter.
const PROCESSOR: &str = "host,invtsc=on,topoext=on";

/// Memory the guest is given. Enough for a desktop installer to run in.
const MEMORY: &str = "8G";

/// Everything needed to assemble one QEMU invocation.
///
/// `esp` and `installer` are mutually exclusive: whichever is present claims
/// first boot priority.
pub struct Spec {
    /// Names the per-guest firmware-variable store and TPM state.
    pub label: &'static str,
    /// Staged ESP directory to boot first, if any.
    pub esp: Option<PathBuf>,
    /// Guest OS disk to attach behind the boot media, if any.
    pub disk: Option<PathBuf>,
    /// Installer ISO to boot from, if any.
    pub installer: Option<PathBuf>,
    /// Attach an emulated TPM 2.0 backed by `swtpm`.
    pub tpm: bool,
    /// Expose QEMU's guest GDB stub on the local machine.
    pub gdb: bool,
    /// Where the guest's own log output goes, and whether it has anywhere to go
    /// at all.
    pub console: Console,
}

/// What output devices the guest is given.
///
/// A choice rather than a list, because "no device" and "the default device"
/// are not points on the same scale: one of them is a machine the guest can
/// describe itself on and the other is a machine where every record it produces
/// is discarded at the source.
pub enum Console {
    /// The debug console, to this terminal. What a run says when nothing asked
    /// otherwise.
    Stdio,
    /// The debug console to the first file, and COM1 upwards to any others.
    Files(Vec<PathBuf>),
    /// Neither a debug console nor a serial port.
    ///
    /// What a machine with no serial header presents, which is the machine this
    /// hypervisor has to run on: the log records are still in the image and
    /// still decide whether they have anything to say, and there is nowhere for
    /// the answer to go. Worth being able to ask for under QEMU, because it is
    /// the one configuration where logging cannot be what changed the timing.
    None,
}

/// Whether pulzar stands in front of the guest.
///
/// A choice rather than a flag, because the two are not settings of one thing.
/// One boots the media this workspace builds, with the guest disk behind it;
/// the other boots that disk on bare firmware, on the same machine with the
/// same processor topology. The second exists so that a fault of the
/// hypervisor's can be told from one the guest has on this hardware anyway,
/// which is not a question any amount of looking at the first can answer.
#[derive(Clone, Copy)]
pub enum Layering {
    /// Boot the staged hypervisor media, with the guest disk behind it.
    Hypervisor,
    /// Boot the guest disk directly. Nothing is built, because nothing of
    /// pulzar's is used.
    Bare,
}

/// Builds the boot media and runs it in QEMU, optionally with a guest disk.
///
/// `silent` compiles every log record out of both images, which is a strange
/// thing to ask of a QEMU run — the debug console there costs one port write a
/// byte — and is offered anyway, because a bare-metal image is worth being able
/// to try under QEMU before it is written to a stick.
pub fn run(
    os: Guest,
    release: bool,
    gdb: bool,
    silent: bool,
    console: Console,
    layering: Layering,
) -> Result<()> {
    ensure!(
        matches!(layering, Layering::Hypervisor) || os != Guest::None,
        "nothing to boot: --no-hypervisor leaves only the guest disk, and --os none leaves no disk"
    );
    let disk = match os {
        Guest::None => None,
        Guest::Linux => Some(existing_image(os, "cargo xtask disk linux")?),
        Guest::Cachyos => Some(existing_image(os, "cargo xtask disk cachyos")?),
        Guest::Windows => Some(existing_image(
            os,
            "cargo xtask disk windows --iso <windows.iso>",
        )?),
    };
    let esp = match layering {
        Layering::Hypervisor => Some(esp::stage(release, silent)?),
        Layering::Bare => {
            println!("booting {} with no hypervisor in front of it", os.label());
            None
        }
    };
    launch(&Spec {
        label: os.label(),
        esp,
        disk,
        installer: None,
        tpm: matches!(os, Guest::Windows),
        gdb,
        console,
    })
}

/// Launches QEMU as described by `spec` and waits for it to exit.
pub fn launch(spec: &Spec) -> Result<()> {
    let (code, vars) = firmware(spec.label)?;
    let mut qemu = Command::new("qemu-system-x86_64");
    qemu.args([
        "-machine",
        "q35,accel=kvm,kernel-irqchip=on",
        "-cpu",
        PROCESSOR,
    ]);
    qemu.args(["-smp", TOPOLOGY, "-m", MEMORY]);
    qemu.args(["-no-shutdown", "-no-reboot"]);
    qemu.args(["-overcommit", "cpu-pm=on"]);
    if spec.gdb {
        qemu.args(["-gdb", "tcp:127.0.0.1:1234"]);
    }
    output_args(&mut qemu, &spec.console)?;
    qemu.arg("-drive").arg(format!(
        "if=pflash,format=raw,readonly=on,file={}",
        drive_path(&code)
    ));
    qemu.arg("-drive")
        .arg(format!("if=pflash,format=raw,file={}", drive_path(&vars)));
    if let Some(dir) = &spec.esp {
        qemu.arg("-drive").arg(format!(
            "if=none,id=esp,format=raw,file=fat:rw:{}",
            drive_path(dir)
        ));
        qemu.args(["-device", "ide-hd,drive=esp,bus=ide.0,bootindex=0"]);
    }
    if let Some(image) = &spec.disk {
        qemu.arg("-drive").arg(format!(
            "if=none,id=os,format=qcow2,file={}",
            drive_path(image)
        ));
        qemu.args(["-device", "ide-hd,drive=os,bus=ide.1,bootindex=1"]);
    }
    if let Some(iso) = &spec.installer {
        qemu.arg("-drive").arg(format!(
            "if=none,id=installer,format=raw,media=cdrom,file={}",
            drive_path(iso)
        ));
        qemu.args(["-device", "ide-cd,drive=installer,bus=ide.2,bootindex=0"]);
    }
    let _tpm = if spec.tpm {
        Some(start_swtpm(spec.label, &mut qemu)?)
    } else {
        None
    };
    proc::run(&mut qemu, QEMU_INSTALL_HINT)
}

/// Wires the guest's log outputs into `qemu`.
///
/// The debug console is the guest's own preferred backend and gets the first
/// destination, because that is where its output will actually appear: it is a
/// single I/O port with no line rate, so a hypervisor describing its own
/// interrupt path is affordable through it and is not through a UART.
///
/// With no capture files requested that destination is stdio, as COM1 always
/// was. Otherwise the first file becomes the debug console's and any further
/// ones become COM ports in request order, so a run that wants both still gets
/// both. Files are created empty up front — appending to a previous run's log,
/// or silently keeping one around when QEMU fails to start, would be a
/// debugging hazard.
///
/// [`Console::None`] passes neither, and the debug console's absence is its
/// absence from the command line: a port nothing is attached to is a port the
/// guest's probe does not find, which is the whole point of asking for it.
fn output_args(qemu: &mut Command, console: &Console) -> Result<()> {
    let files = match console {
        Console::None => {
            qemu.args(["-serial", "none"]);
            return Ok(());
        }
        Console::Stdio => &[][..],
        Console::Files(files) => files,
    };
    let Some((first, ports)) = files.split_first() else {
        qemu.args(["-debugcon", "stdio", "-serial", "none"]);
        return Ok(());
    };
    ensure!(
        ports.len() <= 4,
        "QEMU's PC machines expose at most four serial ports (COM1–COM4) beyond the debug console"
    );
    qemu.arg("-chardev")
        .arg(format!("file,id=debugcon,path={}", chardev_path(first)?));
    qemu.args(["-debugcon", "chardev:debugcon"]);
    for (index, path) in ports.iter().enumerate() {
        qemu.arg("-chardev")
            .arg(format!("file,id=char{index},path={}", chardev_path(path)?));
        qemu.arg("-serial").arg(format!("chardev:char{index}"));
    }
    Ok(())
}

/// Creates one capture file empty and formats its path for a chardev option.
fn chardev_path(path: &Path) -> Result<String> {
    fs::File::create(path)
        .with_context(|| format!("failed to create log file {}", path.display()))?;
    Ok(drive_path(path))
}

/// Removes a guest's firmware-variable store and TPM state, so a recreated
/// disk starts from factory-fresh NVRAM.
pub fn discard_state(guest: Guest) -> Result<()> {
    let vars = vars_path(guest.label())?;
    if vars.exists() {
        fs::remove_file(&vars).with_context(|| format!("failed to remove {}", vars.display()))?;
    }
    let tpm = paths::cache_dir("tpm")?.join(guest.label());
    if tpm.exists() {
        fs::remove_dir_all(&tpm).with_context(|| format!("failed to remove {}", tpm.display()))?;
    }
    Ok(())
}

fn existing_image(guest: Guest, hint: &str) -> Result<PathBuf> {
    let image = disk::image_path(guest)?;
    ensure!(
        image.exists(),
        "no {} guest disk yet — create one with `{hint}`",
        guest.label()
    );
    Ok(image)
}

/// Returns the OVMF code image and this guest's writable variable store,
/// fetching the prebuilt firmware and seeding the store on first use.
fn firmware(label: &str) -> Result<(PathBuf, PathBuf)> {
    let prebuilt = Prebuilt::fetch(Source::LATEST, paths::cache_dir("ovmf")?)
        .context("failed to fetch prebuilt OVMF firmware")?;
    let vars = vars_path(label)?;
    if !vars.exists() {
        fs::copy(prebuilt.get_file(Arch::X64, FileType::Vars), &vars)
            .with_context(|| format!("failed to seed {}", vars.display()))?;
    }
    Ok((prebuilt.get_file(Arch::X64, FileType::Code), vars))
}

fn vars_path(label: &str) -> Result<PathBuf> {
    Ok(paths::cache_dir("firmware-vars")?.join(format!("{label}.fd")))
}

/// Kills the background `swtpm` process when the QEMU session it served is
/// over.
struct TpmGuard {
    child: Child,
}

impl Drop for TpmGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Starts `swtpm` for `label`, waits for its control socket, and wires the
/// matching TPM 2.0 device options into `qemu`.
fn start_swtpm(label: &str, qemu: &mut Command) -> Result<TpmGuard> {
    let state = paths::cache_dir(&format!("tpm/{label}"))?;
    let socket = state.join("swtpm.sock");
    if socket.exists() {
        fs::remove_file(&socket)
            .with_context(|| format!("failed to remove stale socket {}", socket.display()))?;
    }
    let mut swtpm = Command::new("swtpm");
    swtpm.args(["socket", "--tpm2", "--terminate"]);
    swtpm
        .arg("--tpmstate")
        .arg(format!("dir={}", state.display()));
    swtpm
        .arg("--ctrl")
        .arg(format!("type=unixio,path={}", socket.display()));
    let child = proc::spawn(
        &mut swtpm,
        "install swtpm (required to emulate the TPM 2.0 that Windows 11 demands)",
    )?;
    let mut guard = TpmGuard { child };
    for _ in 0..50 {
        if socket.exists() {
            qemu.arg("-chardev")
                .arg(format!("socket,id=chrtpm,path={}", socket.display()));
            qemu.args(["-tpmdev", "emulator,id=tpm0,chardev=chrtpm"]);
            qemu.args(["-device", "tpm-crb,tpmdev=tpm0"]);
            return Ok(guard);
        }
        if let Some(status) = guard.child.try_wait().context("failed to poll swtpm")? {
            bail!("swtpm exited early with {status}");
        }
        thread::sleep(Duration::from_millis(100));
    }
    bail!(
        "swtpm did not create its control socket at {}",
        socket.display()
    )
}

/// Formats a path for embedding in a QEMU option string, where a literal
/// comma must be escaped by doubling it.
fn drive_path(path: &Path) -> String {
    path.display().to_string().replace(',', ",,")
}
