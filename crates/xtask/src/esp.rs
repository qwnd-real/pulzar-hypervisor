//! Builds the UEFI crates and stages the EFI system partition.
//!
//! The ESP is a plain directory rather than a filesystem image: QEMU's
//! virtual-FAT driver serves it to the guest directly, so boot media is
//! regenerated from source on every run and no image artifact ever needs to
//! be built, tracked, or cleaned up.

use std::{env, fs, path::PathBuf, process::Command};

use anyhow::{Context, Result};

use crate::{paths, proc};

/// UEFI crates staged into the ESP: package name, image file produced under
/// `target/`, and destination path inside the ESP.
const IMAGES: [(&str, &str, &str); 2] = [
    ("hv-loader", "hv-loader.efi", "EFI/BOOT/BOOTX64.EFI"),
    ("hv-core", "hv-core.efi", "pulzar.efi"),
];

/// Compiles the UEFI crates and repopulates `dist/esp/` from scratch,
/// returning its path.
///
/// `silent` builds both images with their `quiet` feature, which takes `log`'s
/// static maximum level to `Off` and so compiles every record out of the whole
/// image. It is asked of both packages rather than one, even though `log` is
/// compiled once for the build and either would do it: a flag whose effect
/// depends on feature unification is one that stops working the moment the
/// dependency graph changes.
pub fn stage(release: bool, silent: bool) -> Result<PathBuf> {
    let root = paths::workspace_root();
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut build = Command::new(cargo);
    build.current_dir(&root).arg("build");
    for (package, _, _) in IMAGES {
        build.args(["--package", package]);
    }
    if release {
        build.arg("--release");
    }
    if silent {
        let features: Vec<String> = IMAGES
            .iter()
            .map(|(package, _, _)| format!("{package}/quiet"))
            .collect();
        build.args(["--features", &features.join(",")]);
    }
    proc::run(&mut build, "it ships with the Rust toolchain")?;

    let profile = if release { "release" } else { "debug" };
    let artifacts = root.join("target/x86_64-unknown-uefi").join(profile);
    let dir = paths::esp_dir();
    if dir.exists() {
        fs::remove_dir_all(&dir)
            .with_context(|| format!("failed to clear stale ESP at {}", dir.display()))?;
    }
    for (_, image, destination) in IMAGES {
        let to = dir.join(destination);
        let parent = to.parent().expect("every ESP destination has a parent");
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
        fs::copy(artifacts.join(image), &to)
            .with_context(|| format!("failed to stage {image} as {}", to.display()))?;
    }
    println!("staged ESP at {}", dir.display());
    Ok(dir)
}
