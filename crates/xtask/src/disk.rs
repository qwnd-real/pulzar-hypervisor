//! Guest OS disk provisioning.
//!
//! Disks live in the per-user cache, never in the repository. The Linux disk is
//! Debian's pre-installed "nocloud" cloud image (root logs in with no
//! password), pinned to an exact build and verified against its published
//! SHA-512, so no installer ever runs. The `CachyOS` disk is installed once
//! from the distribution's own ISO, which is fetched and pinned the same way —
//! it is a desktop image with no unattended mode, so setup is interactive and
//! happens with no hypervisor in front of it. Windows cannot be redistributed,
//! so its disk is created blank and installed interactively from an ISO
//! supplied by the contributor.

use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256, Sha512};

use crate::{Guest, paths, proc, vm};

const DEBIAN_URL: &str = "https://cloud.debian.org/images/cloud/trixie/20260722-2547/debian-13-nocloud-amd64-20260722-2547.qcow2";
const DEBIAN_SHA512: &str = "cb22bf0acb0718a2d9a8c88534f950937bb8439116c0bf8eff52792e88da982a8e2930b6ae25c175db0bce4db6a1c3be5a2f05aa1960e5fc442a3e0a70f8a042";

const CACHYOS_URL: &str =
    "https://cdn77.cachyos.org/ISO/desktop/260809/cachyos-desktop-linux-260809.iso";

/// The digest the distribution publishes beside that ISO, in the `.sha256` file
/// next to it.
///
/// It comes from the same server as the image, so it establishes that the three
/// gigabytes arrived intact and not that they are the distribution's — which is
/// what a signature would say and this does not.
const CACHYOS_SHA256: &str = "959f6577f45e25ee9fd8c220fd221b08e4ea79412c7315c0f922dd6d86d5e33c";

/// Room for a desktop installation and the packages that follow it. The qcow2
/// is sparse, so only written sectors consume host space.
const CACHYOS_DISK_SIZE: &str = "48G";

/// Windows 11 setup refuses disks smaller than 64 GB; the qcow2 is sparse,
/// so only written sectors consume host space.
const WINDOWS_DISK_SIZE: &str = "64G";

/// Room for a scratch filesystem, if a guest puts one on the `NVMe` disk at
/// all. The qcow2 is sparse, so an untouched disk consumes none of it.
const NVME_DISK_SIZE: &str = "1G";

/// How much of a download goes by between progress lines.
const PROGRESS_STEP: usize = 256 << 20;

/// Bytes read from the network at a time.
const CHUNK: usize = 1 << 20;

/// Which digest a download is checked against.
///
/// One enum rather than two functions because the streaming, the progress
/// reporting and the move-into-place are the same either way, and which
/// algorithm a distribution happens to publish is not a reason to have two
/// copies of them.
#[derive(Clone, Copy)]
enum Checksum {
    /// SHA-256, which is what `CachyOS` publishes beside its ISOs.
    Sha256(&'static str),
    /// SHA-512, which is what Debian publishes beside its cloud images.
    Sha512(&'static str),
}

impl Checksum {
    /// The digest as published, in lowercase hexadecimal.
    const fn expected(self) -> &'static str {
        match self {
            Self::Sha256(digest) | Self::Sha512(digest) => digest,
        }
    }
}

/// Location of a guest's disk image in the per-user cache.
pub fn image_path(guest: Guest) -> Result<PathBuf> {
    Ok(paths::cache_dir("disks")?.join(format!("{}.qcow2", guest.label())))
}

/// The blank disk the machine's `NVMe` controller holds, created on first use.
///
/// The controller is the one device this hypervisor answers for in place of
/// the hardware, so every machine has one — the guest an OS was provisioned
/// onto is attached alongside it, not moved onto it, which keeps the boot
/// media and the interposed device two separate questions. The disk is blank
/// and stays that way unless a guest partitions it, and it is kept in the
/// cache so whatever a guest puts on it survives between runs.
pub fn nvme_disk() -> Result<PathBuf> {
    let image = paths::cache_dir("disks")?.join("nvme.qcow2");
    if !image.exists() {
        create_disk(&image, NVME_DISK_SIZE)?;
    }
    Ok(image)
}

/// Fetches the pre-installed Debian image as the Linux guest disk.
pub fn linux(force: bool) -> Result<()> {
    let image = image_path(Guest::Linux)?;
    if image.exists() {
        if !force {
            println!(
                "linux guest disk already exists at {} (use --force to re-fetch)",
                image.display()
            );
            return Ok(());
        }
        vm::discard_state(Guest::Linux)?;
    }
    fetch_verified(DEBIAN_URL, Checksum::Sha512(DEBIAN_SHA512), &image)?;
    println!("linux guest disk ready at {}", image.display());
    Ok(())
}

/// Fetches the `CachyOS` ISO and boots its installer onto a blank disk, with no
/// hypervisor in front of it.
///
/// Interactive and once. It is a desktop image with no unattended installation
/// mode, so there is a person at the installer either way — and running it on
/// bare firmware rather than behind pulzar is deliberate: an installation is
/// what every later comparison is made against, so it must not have been made
/// through the thing being tested.
pub fn cachyos(force: bool) -> Result<()> {
    let image = image_path(Guest::Cachyos)?;
    if image.exists() {
        if !force {
            println!(
                "cachyos guest disk already exists at {} (use --force to start over)",
                image.display()
            );
            return Ok(());
        }
        fs::remove_file(&image).with_context(|| format!("failed to remove {}", image.display()))?;
        vm::discard_state(Guest::Cachyos)?;
    }
    let iso = iso_path("cachyos")?;
    if iso.exists() {
        println!("using the cached installer at {}", iso.display());
    } else {
        fetch_verified(CACHYOS_URL, Checksum::Sha256(CACHYOS_SHA256), &iso)?;
    }
    create_disk(&image, CACHYOS_DISK_SIZE)?;

    println!(
        "booting the CachyOS installer with no hypervisor — install to the blank disk, then shut the guest down"
    );
    vm::launch(&vm::Spec {
        label: Guest::Cachyos.label(),
        esp: None,
        disk: Some(image),
        installer: Some(iso),
        tpm: false,
        gdb: false,
        console: vm::Console::Stdio,
    })?;
    println!(
        "installer session ended — if setup completed, boot it behind the hypervisor with `cargo xtask run --os cachyos`, or without one with `cargo xtask run --os cachyos --no-hypervisor`; otherwise rerun with --force"
    );
    Ok(())
}

/// Creates a blank Windows disk and boots the installer from `iso` for a
/// one-time interactive installation.
pub fn windows(iso: &Path, force: bool) -> Result<()> {
    ensure!(iso.is_file(), "no ISO file at {}", iso.display());
    let image = image_path(Guest::Windows)?;
    if image.exists() {
        if !force {
            println!(
                "windows guest disk already exists at {} (use --force to start over)",
                image.display()
            );
            return Ok(());
        }
        fs::remove_file(&image).with_context(|| format!("failed to remove {}", image.display()))?;
        vm::discard_state(Guest::Windows)?;
    }
    create_disk(&image, WINDOWS_DISK_SIZE)?;

    println!(
        "booting the Windows installer — complete setup in the QEMU window, then shut the guest down"
    );
    vm::launch(&vm::Spec {
        label: Guest::Windows.label(),
        esp: None,
        disk: Some(image),
        installer: Some(iso.to_path_buf()),
        tpm: true,
        gdb: false,
        console: vm::Console::Stdio,
    })?;
    println!(
        "installer session ended — if setup completed, boot the guest behind the hypervisor with `cargo xtask run --os windows`; otherwise rerun with --force"
    );
    Ok(())
}

/// Where an installer ISO is kept once fetched, so a reinstall does not fetch
/// gigabytes again.
fn iso_path(label: &str) -> Result<PathBuf> {
    Ok(paths::cache_dir("isos")?.join(format!("{label}.iso")))
}

/// Creates a sparse qcow2 of `size` for an installer to write into.
fn create_disk(image: &Path, size: &str) -> Result<()> {
    let mut create = Command::new("qemu-img");
    create.args(["create", "-f", "qcow2"]).arg(image).arg(size);
    proc::run(&mut create, vm::QEMU_INSTALL_HINT)
}

/// Downloads `url` to `destination`, streaming it through its digest and only
/// moving it into place once that matches `checksum`.
fn fetch_verified(url: &str, checksum: Checksum, destination: &Path) -> Result<()> {
    let partial = destination.with_extension("partial");
    println!("downloading {url}");
    let response = ureq::get(url)
        .call()
        .with_context(|| format!("failed to download {url}"))?;
    let mut body = response.into_body().into_reader();
    let mut file = File::create(&partial)
        .with_context(|| format!("failed to create {}", partial.display()))?;
    let digest = match checksum {
        Checksum::Sha256(_) => stream::<Sha256>(&mut body, &mut file, &partial),
        Checksum::Sha512(_) => stream::<Sha512>(&mut body, &mut file, &partial),
    };
    drop(file);
    let digest = match digest {
        Ok(digest) => digest,
        Err(error) => {
            let _ = fs::remove_file(&partial);
            return Err(error);
        }
    };

    if digest != checksum.expected() {
        let _ = fs::remove_file(&partial);
        bail!(
            "checksum mismatch for {url}: expected {}, got {digest}",
            checksum.expected()
        );
    }
    fs::rename(&partial, destination)
        .with_context(|| format!("failed to move image into {}", destination.display()))?;
    Ok(())
}

/// Copies `body` into `file`, reporting progress, and answers its digest.
fn stream<D: Digest>(body: &mut impl Read, file: &mut File, path: &Path) -> Result<String> {
    let mut hasher = D::new();
    let mut buffer = vec![0_u8; CHUNK];
    let mut total = 0_usize;
    let mut reported = 0_usize;
    loop {
        let read = body.read(&mut buffer).context("download interrupted")?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        file.write_all(&buffer[..read])
            .with_context(|| format!("failed to write {}", path.display()))?;
        total += read;
        if total - reported >= PROGRESS_STEP {
            println!("  {} MiB", total >> 20);
            reported = total;
        }
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(digest: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(out, "{byte:02x}").expect("writing to a String cannot fail");
    }
    out
}
