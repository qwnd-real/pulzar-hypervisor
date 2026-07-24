//! Child-process helpers shared by the task runner's commands.

use std::{
    io,
    process::{Child, Command},
};

use anyhow::{Result, anyhow, ensure};

/// Runs `command` to completion, failing if it exits non-zero.
///
/// A missing executable is reported with `install_hint` so a contributor
/// learns what to install instead of seeing a bare "No such file" error.
pub fn run(command: &mut Command, install_hint: &str) -> Result<()> {
    let program = program_name(command);
    let status = command
        .status()
        .map_err(|err| spawn_error(&program, install_hint, &err))?;
    ensure!(status.success(), "`{program}` failed with {status}");
    Ok(())
}

/// Spawns `command` in the background, with the same missing-executable
/// reporting as [`run`].
pub fn spawn(command: &mut Command, install_hint: &str) -> Result<Child> {
    let program = program_name(command);
    command
        .spawn()
        .map_err(|err| spawn_error(&program, install_hint, &err))
}

fn program_name(command: &Command) -> String {
    command.get_program().to_string_lossy().into_owned()
}

fn spawn_error(program: &str, install_hint: &str, err: &io::Error) -> anyhow::Error {
    if err.kind() == io::ErrorKind::NotFound {
        anyhow!("`{program}` not found in PATH — {install_hint}")
    } else {
        anyhow!("failed to run `{program}`: {err}")
    }
}
