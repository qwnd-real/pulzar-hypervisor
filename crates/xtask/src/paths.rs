//! Filesystem locations used by the task runner.
//!
//! Two roots: the workspace itself (source, build output, and the staged ESP
//! under `dist/`) and a per-user cache (`~/.cache/pulzar` on Linux) for
//! artifacts that are expensive to recreate and independent of any one
//! checkout — guest OS disks, OVMF firmware, and TPM state. Keeping those in
//! the user cache means they survive re-clones, are shared between git
//! worktrees, and can never end up in the repository.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};

/// Repository root, derived from this crate's location at compile time.
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/xtask sits two levels below the workspace root")
        .to_path_buf()
}

/// Staged EFI-system-partition directory, served to QEMU as a virtual FAT
/// drive.
pub fn esp_dir() -> PathBuf {
    workspace_root().join("dist/esp")
}

/// Subdirectory of the per-user cache, created on first use.
pub fn cache_dir(subdir: &str) -> Result<PathBuf> {
    let dir = dirs::cache_dir()
        .context("no per-user cache directory is defined on this platform")?
        .join("pulzar")
        .join(subdir);
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create cache directory {}", dir.display()))?;
    Ok(dir)
}
