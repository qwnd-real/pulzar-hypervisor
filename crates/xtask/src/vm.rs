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
    /// Serial ports to capture: entry N becomes COM(N+1), written to the
    /// named file. Empty means the default single COM1-on-stdio port.
    pub serial_logs: Vec<PathBuf>,
}

/// Builds the boot media and runs it in QEMU, optionally with a guest disk.
pub fn run(os: Guest, release: bool, serial_logs: Vec<PathBuf>) -> Result<()> {
    let staged = esp::stage(release)?;
    let disk = match os {
        Guest::None => None,
        Guest::Linux => Some(existing_image(os, "cargo xtask disk linux")?),
        Guest::Windows => Some(existing_image(
            os,
            "cargo xtask disk windows --iso <windows.iso>",
        )?),
    };
    launch(&Spec {
        label: os.label(),
        esp: Some(staged),
        disk,
        installer: None,
        tpm: matches!(os, Guest::Windows),
        serial_logs,
    })
}

/// Launches QEMU as described by `spec` and waits for it to exit.
pub fn launch(spec: &Spec) -> Result<()> {
    let (code, vars) = firmware(spec.label)?;
    let mut qemu = Command::new("qemu-system-x86_64");
    qemu.args(["-machine", "q35,accel=kvm", "-cpu", "host"]);
    qemu.args(["-smp", "1", "-m", "4G"]);
    serial_args(&mut qemu, &spec.serial_logs)?;
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

/// Wires the guest serial ports into `qemu`: with no capture files
/// requested, COM1 goes to stdio as always; otherwise each file becomes a
/// chardev feeding one COM port, in request order. Files are created empty
/// up front — appending to a previous run's log, or silently keeping one
/// around when QEMU fails to start, would be a debugging hazard.
fn serial_args(qemu: &mut Command, logs: &[PathBuf]) -> Result<()> {
    if logs.is_empty() {
        qemu.args(["-serial", "stdio"]);
        return Ok(());
    }
    ensure!(
        logs.len() <= 4,
        "QEMU's PC machines expose at most four serial ports (COM1–COM4)"
    );
    for (index, path) in logs.iter().enumerate() {
        fs::File::create(path)
            .with_context(|| format!("failed to create serial log {}", path.display()))?;
        qemu.arg("-chardev")
            .arg(format!("file,id=char{index},path={}", drive_path(path)));
        qemu.arg("-serial").arg(format!("chardev:char{index}"));
    }
    Ok(())
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
