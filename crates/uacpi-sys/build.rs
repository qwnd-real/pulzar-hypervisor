//! Compiles uACPI out of the pinned submodule and generates the Rust
//! declarations for its headers.
//!
//! Both halves read the same checked-out tree, so the declarations can never
//! describe a different uACPI than the one that gets linked. That is the whole
//! reason the bindings are generated here rather than committed: the submodule
//! pin is the single source of truth for both.
//!
//! # Why the C target is spelled out
//!
//! `x86_64-unknown-uefi` is a PE/COFF target with no C toolchain of its own, so
//! the C compiler is pointed at the triple that produces the same objects and
//! the same fundamental type widths. Getting that wrong is not a build failure
//! but a silent ABI mismatch: uACPI types `uacpi_cpu_flags` as `unsigned long`,
//! which is 32 bits under a Windows data model and 64 under a Unix one, and the
//! generated declarations and the compiled code have to agree about it.

use std::{
    env, fs,
    path::{Path, PathBuf},
};

/// The submodule, relative to the workspace root.
const SUBMODULE: &str = "third_party/uacpi";

/// A file that only exists if the submodule has actually been checked out, used
/// to tell an uninitialized submodule from a broken one.
const WITNESS: &str = "include/uacpi/uacpi.h";

/// Makes `uacpi_kernel_free` take the size of the allocation being released.
///
/// Not an optimization. Rust's deallocation requires the layout the allocation
/// was made with, and this is what supplies it — without it the host would have
/// to over-allocate every block to keep a size header of its own in front of
/// it.
const SIZED_FREES: &str = "UACPI_SIZED_FREES";

/// Makes uACPI take its architecture helpers from this crate's own
/// `uacpi_arch_helpers.h` instead of its default ones.
///
/// The default types a spinlock's saved interrupt state as `unsigned long`,
/// whose width the C compiler and Rust disagree about on the firmware target.
/// See that header.
const OVERRIDE_ARCH_HELPERS: &str = "UACPI_OVERRIDE_ARCH_HELPERS";

/// Strips the parts of uACPI that own ACPI's hardware.
///
/// The interpreter and the namespace stay; the event subsystem, the global lock
/// and the fixed-event machinery go. That is not a reduction in what pulzar
/// reads out of a machine — it is a statement of what pulzar is. A pass-through
/// hypervisor hands the platform on to firmware and then to an operating
/// system, and that operating system enters ACPI mode, enables the general
/// purpose events it wants, takes the global lock and services the system
/// control interrupt. Two owners of a shared, level-triggered line is a lost
/// interrupt, not a configuration.
///
/// Compiling those parts out rather than declining them at run time is what
/// makes the decision checkable: without this, bringing the namespace up asks
/// the host to route the control interrupt and fails when it will not, so the
/// namespace a machine describes would depend on a refusal several layers away.
const REDUCED_HARDWARE: &str = "UACPI_REDUCED_HARDWARE";

/// The replacement architecture helpers, and the widths they pin, both of which
/// live beside this build script rather than in the submodule.
const OWN_SOURCES: [&str; 1] = ["widths.c"];

fn main() {
    let uacpi = submodule();
    let sources = sources(&uacpi);
    let include = uacpi.join("include");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=uacpi_arch_helpers.h");
    for source in OWN_SOURCES {
        println!("cargo:rerun-if-changed={source}");
    }
    println!("cargo:rerun-if-changed={}", uacpi.join("source").display());
    println!("cargo:rerun-if-changed={}", include.display());

    let target = env::var("TARGET").expect("cargo sets TARGET for a build script");
    let clang_target = clang_target(&target);

    compile(&sources, &include, clang_target, &target);
    generate(&include, clang_target);
}

/// Builds the static library the crate links against.
fn compile(sources: &[PathBuf], include: &Path, clang_target: &str, target: &str) {
    let mut build = cc::Build::new();
    build
        .files(sources)
        .files(OWN_SOURCES)
        .include(include)
        // This crate's own directory, so that uACPI's quoted include of
        // `uacpi_arch_helpers.h` resolves to the replacement beside this script.
        .include(".")
        .std("c11")
        .define(SIZED_FREES, None)
        .define(OVERRIDE_ARCH_HELPERS, None)
        .define(REDUCED_HARDWARE, None)
        // uACPI reaches for nothing outside stdint, stddef, stdbool and stdarg,
        // all of which the compiler provides itself. Saying so is what keeps a
        // host build from resolving a header out of the host's sysroot that the
        // firmware build would not have.
        .flag("-ffreestanding")
        .flag("-fno-strict-aliasing")
        .flag("-fno-stack-protector")
        .warnings(false);
    if clang_target != target {
        // A target cargo names and the C compiler does not. Pointing the
        // compiler at the triple by hand also means it is clang doing the
        // compiling, since no host `gcc` can produce objects for it.
        build
            .compiler("clang")
            .flag(format!("--target={clang_target}"));
        build.archiver("llvm-ar");
    }
    if is_firmware(target) {
        // The red zone is unusable anywhere an interrupt or a fault can arrive
        // on the stack it would occupy, which is everywhere this code runs on
        // the firmware side.
        build.flag("-mno-red-zone");
    }
    build.compile("uacpi");
}

/// Writes the Rust declarations for uACPI's headers into the output directory.
fn generate(include: &Path, clang_target: &str) {
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args([
            format!("--target={clang_target}"),
            format!("-I{}", include.display()),
            "-I.".to_owned(),
            "-ffreestanding".to_owned(),
            "-std=c11".to_owned(),
        ])
        .clang_arg(format!("-D{SIZED_FREES}"))
        .clang_arg(format!("-D{OVERRIDE_ARCH_HELPERS}"))
        .clang_arg(format!("-D{REDUCED_HARDWARE}"))
        // The bindings are used from a `no_std` crate, and the generated layout
        // assertions are `#[test]` functions that would need one.
        .use_core()
        .ctypes_prefix("core::ffi")
        .layout_tests(false)
        // uACPI's own headers are the interface; what they include of the
        // compiler's is an implementation detail whose declarations would only
        // collide with `core`'s.
        .allowlist_file(".*uacpi.*")
        // Constants keep the names C gives them. Bindgen would otherwise prefix
        // each one with the name of the enum it came out of, which for uACPI is
        // pure noise: every variant is already prefixed by the subject it
        // belongs to, and a name that matches the header is a name that can be
        // looked up in the header.
        .prepend_enum_name(false)
        .derive_debug(true)
        .derive_copy(true)
        .generate()
        .expect("uACPI's headers describe a C interface bindgen can read");
    let out = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR for a build script"));
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("the output directory is writable");
}

/// The workspace's copy of uACPI, or a refusal naming what to do about it.
///
/// A missing submodule is by far the most common way this build fails, and it
/// fails in the compiler otherwise — on a source list that came back empty,
/// which says nothing about the cause.
fn submodule() -> PathBuf {
    let manifest =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(Path::parent)
        .expect("the manifest lives two directories below the workspace root");
    let uacpi = root.join(SUBMODULE);
    assert!(
        uacpi.join(WITNESS).is_file(),
        "{} is empty; run `git submodule update --init --recursive`",
        uacpi.display()
    );
    uacpi
}

/// Every translation unit uACPI is made of.
///
/// Read from the directory rather than listed here, because a list would be one
/// more thing to keep in step with the pin — and a source added by a version
/// bump would go missing without any build failure to say so. Every file is
/// compiled unconditionally: uACPI's own configuration guards live inside them,
/// so the ones a configuration strips out compile to nothing.
fn sources(uacpi: &Path) -> Vec<PathBuf> {
    let directory = uacpi.join("source");
    let mut sources: Vec<_> = fs::read_dir(&directory)
        .expect("the submodule has a source directory")
        .map(|entry| entry.expect("the source directory is readable").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "c"))
        .collect();
    assert!(
        !sources.is_empty(),
        "{} holds no C sources",
        directory.display()
    );
    // `read_dir` yields in whatever order the filesystem does, and the archive
    // member order follows it. Sorting is what makes the same checkout produce
    // the same static library twice.
    sources.sort();
    sources
}

/// The triple to compile C for, given the triple cargo is building for.
///
/// UEFI is the only one that needs translating: it has no C toolchain under its
/// own name, and the objects it wants are the PE/COFF ones a bare Windows
/// target produces. Everything else is a triple a C compiler already knows.
fn clang_target(target: &str) -> &str {
    match target {
        "x86_64-unknown-uefi" => "x86_64-unknown-windows-coff",
        other => other,
    }
}

/// Whether the target is one this code runs on a machine's own processor for,
/// as opposed to the host builds that exist to run the parsers' tests.
fn is_firmware(target: &str) -> bool {
    target.ends_with("-unknown-uefi")
}
