//! Guest OS disk provisioning.
//!
//! Disks live in the per-user cache, never in the repository. The Linux disk
//! is Debian's pre-installed "nocloud" cloud image (root logs in with no
//! password), pinned to an exact build and verified against its published
//! SHA-512, so no installer ever runs. Windows cannot be redistributed, so
//! its disk is created blank and installed interactively, once, from an ISO
//! supplied by the contributor.

use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha512};

use crate::{Guest, paths, proc, vm};

const DEBIAN_URL: &str = "https://cloud.debian.org/images/cloud/trixie/20260722-2547/debian-13-nocloud-amd64-20260722-2547.qcow2";
const DEBIAN_SHA512: &str = "cb22bf0acb0718a2d9a8c88534f950937bb8439116c0bf8eff52792e88da982a8e2930b6ae25c175db0bce4db6a1c3be5a2f05aa1960e5fc442a3e0a70f8a042";

/// Windows 11 setup refuses disks smaller than 64 GB; the qcow2 is sparse,
/// so only written sectors consume host space.
const WINDOWS_DISK_SIZE: &str = "64G";

/// Location of a guest's disk image in the per-user cache.
pub fn image_path(guest: Guest) -> Result<PathBuf> {
    Ok(paths::cache_dir("disks")?.join(format!("{}.qcow2", guest.label())))
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
    fetch_verified(DEBIAN_URL, DEBIAN_SHA512, &image)?;
    println!("linux guest disk ready at {}", image.display());
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
    let mut create = Command::new("qemu-img");
    create
        .args(["create", "-f", "qcow2"])
        .arg(&image)
        .arg(WINDOWS_DISK_SIZE);
    proc::run(&mut create, vm::QEMU_INSTALL_HINT)?;

    println!(
        "booting the Windows installer — complete setup in the QEMU window, then shut the guest down"
    );
    vm::launch(&vm::Spec {
        label: Guest::Windows.label(),
        esp: None,
        disk: Some(image),
        installer: Some(iso.to_path_buf()),
        tpm: true,
        serial_logs: Vec::new(),
    })?;
    println!(
        "installer session ended — if setup completed, boot the guest behind the hypervisor with `cargo xtask run --os windows`; otherwise rerun with --force"
    );
    Ok(())
}

/// Downloads `url` to `destination`, streaming it through SHA-512 and only
/// moving it into place once the digest matches `sha512`.
fn fetch_verified(url: &str, sha512: &str, destination: &Path) -> Result<()> {
    let partial = destination.with_extension("partial");
    println!("downloading {url}");
    let response = ureq::get(url)
        .call()
        .with_context(|| format!("failed to download {url}"))?;
    let mut body = response.into_body().into_reader();
    let mut file = File::create(&partial)
        .with_context(|| format!("failed to create {}", partial.display()))?;
    let mut hasher = Sha512::new();
    let mut buffer = vec![0_u8; 1 << 20];
    let mut total = 0_usize;
    let mut reported = 0_usize;
    loop {
        let read = body.read(&mut buffer).context("download interrupted")?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        file.write_all(&buffer[..read])
            .with_context(|| format!("failed to write {}", partial.display()))?;
        total += read;
        if total - reported >= 256 << 20 {
            println!("  {} MiB", total >> 20);
            reported = total;
        }
    }
    drop(file);

    let digest = hex(hasher.finalize().as_slice());
    if digest != sha512 {
        let _ = fs::remove_file(&partial);
        bail!("checksum mismatch for {url}: expected {sha512}, got {digest}");
    }
    fs::rename(&partial, destination)
        .with_context(|| format!("failed to move image into {}", destination.display()))?;
    Ok(())
}

fn hex(digest: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(out, "{byte:02x}").expect("writing to a String cannot fail");
    }
    out
}
