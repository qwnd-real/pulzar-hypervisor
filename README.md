# pulzar

[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-nightly-orange?style=flat-square&logo=rust)](rust-toolchain.toml)

A type-1 hypervisor for x86-64, written in `no_std` Rust on AMD SVM.

pulzar boots as a UEFI application and takes the machine over before any
operating system exists. The firmware it started under becomes its first
guest; the OS boot manager runs on inside that guest; and Windows or Linux
boots unmodified and unaware, with every processor it brings up virtualized.
Nothing is paravirtualized and no guest tooling is required — a provisioned
disk boots the same way behind pulzar as in front of it.

The machine is passed through, not emulated: a guest's devices are its real
ones. The two exceptions are deliberate. The interrupt controller is emulated,
because a guest that reached the real one could mask the host's interrupts,
acknowledge them, or reset the host's processors. And the NVMe controller's
identify responses are answered for, so a guest reading its drives' serial
numbers learns nothing that names the hardware — the data path itself is
never touched.

The codebase is meant as a clean, complete baseline for building your own
type-1 hypervisor: every part such a project needs — loader, handoff, paging,
nested page tables, exit handling, instruction emulation, SMP bring-up, an
interrupt controller, device interposition — is here, finished, with no stubs
and nothing placeholder. Below is exactly what is present, and how to run it.

## How it boots

1. Firmware starts `hv-loader` (`EFI/BOOT/BOOTX64.EFI`). It captures the
   state firmware is running with, reserves a 64 MiB chunk of physical memory
   for the hypervisor, and preloads the guest boot manager
   (`\EFI\Limine\limine_x64.efi`, searched on every filesystem volume — a
   missing or ambiguous path refuses the boot rather than guessing).
2. It captures the memory map, loads and relocates the hypervisor image at a
   randomized high-half address alongside a guarded stack, a direct map of
   physical memory and a mapping window, publishes a `Handoff` describing all
   of it, and jumps.
3. `hv-core` adopts that address space, establishes host descriptor tables
   and SVM state, and drops firmware's identity-mapped half. It enters the
   firmware snapshot as a guest through a two-page portal that starts the
   preloaded boot manager inside the guest and wraps `ExitBootServices`.
4. When the wrapped `ExitBootServices` succeeds, the other processors join
   the same guest — held exactly as a processor still in reset would be,
   until the OS's startup messages release them, in real mode, in reset
   state, and held again if the OS resets them once more.

The portal pages are the only hypervisor-owned pages a guest is ever shown,
and they are taken back at the first exit taken outside them. From then on
the whole of hypervisor memory presents as an immutable zero page through
nested paging, for the rest of the guest's life. `CPUID` reports no
virtualization extension, the model-specific registers answer consistently
with that, and the memory-type range registers are answered from a copy —
there is nothing for a guest to find.

## What is implemented

| Area | What is there |
|---|---|
| World switch | Full SVM run loop per processor, VMCB layouts asserted at compile time, clean-bit tracking |
| Nested paging | Second translation for guest memory, per-region device adoption, traps as fine as a single page |
| Hypervisor memory | KASLR of the image, guarded stacks, direct physical map, buddy allocator over the reserved chunk |
| Exit handling | `CPUID`/MSR concealment, MTRR virtualization, MSR permission bitmaps, nested page faults, exit census counters — an exit nothing answers stops the guest, never resumes it blindly |
| Instruction emulation | Decoder on the exit path for the instructions hardware cannot resume on its own |
| Interrupt controller | Full vlapic register-file emulation (see below), timer and LVT sources passed through with host-claimed vectors |
| AVIC | Hardware-driven interrupt delivery where the processor supports it, register page handed to hardware only while that is safe |
| SMP | All processors join the guest, held in reset until the OS starts them, re-holdable on INIT |
| Interrupts | Host interrupt entry path, inter-processor delivery between host processors, injection into a guest |
| ACPI | uACPI (pinned submodule) in reduced-hardware mode — the namespace is read, the hardware is the OS's to own |
| PCI | Configuration-space enumeration and recognition of devices to interpose |
| NVMe spoofing | Identify responses answered with spoofed identity, data path untouched (see below) |
| Hypercall | Versioned `VMMCALL` interface, answerable from ring 3 — a tool inside the guest needs no driver |
| Logging | 16550 UART, QEMU debug console and frame-buffer backends; every record can be compiled out |

### The interrupt controller

Every register a guest can reach is emulated: the request, in-service and
trigger-mode banks, the priorities, the error status with its write-then-read
protocol, the interrupt command register with both destination models, the
whole base-register state machine, and both of its faces — the memory-mapped
page and the x2APIC model-specific registers. Every interrupt on the machine
arrives in a host handler first and reaches the guest only as a decision the
emulator made. Where the processor can deliver a guest's interrupts itself
(AVIC), the tables are built and the delivery runs without exits.

### NVMe identity spoofing

A guest reads its drives' identities a handful of times a boot, through the
NVMe identify commands. Those are answered for: every identifying field —
serial number, model, firmware revision, volume identifiers — is replaced by
what the `spoof` transform makes of it under a per-machine 16-byte seed. The
replacement keeps the field's exact shape (a digit stays a digit, a
hexadecimal letter stays a hexadecimal letter in its case, separators and
padding stay where they were), is stable across boots and across processors,
and breaks the link to the hardware's real identity. The data path never
faults: reads of the register file are answered by hardware, and a doorbell
write costs one exit, forwarded with no lock held and nothing allocated.
Every way the driver can lose its place fails open — a loud warning and the
response passing through — because a guest that does not boot is a worse
failure than one that read one real serial number.

## Workspace layout

Two UEFI images, built for `x86_64-unknown-uefi`:

| Crate | Role |
|---|---|
| `hv-loader` | First-stage UEFI application: snapshot firmware, reserve memory, preload the guest boot manager, map the hypervisor, jump |
| `hv-core` | The hypervisor image: bring-up, the firmware guest, the exit loop, uACPI's host |

Firmware-side libraries, target-agnostic `no_std` so their tests run natively:

| Crate | Role |
|---|---|
| `acpi` | Firmware's ACPI tables, turned into data the hypervisor owns |
| `apic` | The machine's own interrupt controllers: driving, capturing, starting processors |
| `clock` | The machine's counters, turned into time |
| `config` | The machine's own choices, the spoof seed among them |
| `cpu` | Which processors the machine has, and which one is running |
| `descriptors` | The host's own tables and the interrupt entry path |
| `emulate` | Instruction decoding on the exit path |
| `exits` | What every guest exit means, and what the host does about it |
| `handoff` | The ABI the loader and the hypervisor image agree on |
| `hypercall` | The `VMMCALL` interface, shared by the exit path and guest-side tools |
| `inject` | Interrupt injection into a guest |
| `ipi` | How one host processor reaches another |
| `memory` | How anything reaches a guest's memory through the nested tables |
| `npt` | The second set of page tables a guest's memory is described by |
| `paging` | The hypervisor's own address space: KASLR, direct map, the reserved chunk |
| `partition` | The guest: its memory description and the devices answering for its regions |
| `pci` | Configuration space, and recognizing a device to interpose |
| `portal` | The pages firmware is entered at and leaves through |
| `probe` | MSR access that reports a refusal rather than taking one |
| `processor` | What the processor underneath all of them can do |
| `serial` | 16550 UART, debug console and frame-buffer logging backends |
| `snapshot` | The firmware context, read before any of it is overwritten |
| `spoof` | The keyed, format-preserving transform behind identity replacement |
| `svm` | AMD's SVM structures stated once, with layouts checked at compile time |
| `uacpi-sys` | The uACPI submodule, compiled and bound |
| `vcpu` | Turning SVM on and running a guest on one processor |
| `vlapic` | The interrupt controller a guest sees in place of the machine's |
| `drivers/nvme` | The storage driver answering a guest's identify commands |

Host-side tooling:

| Crate | Role |
|---|---|
| `xtask` | Builds the images, stages boot media, provisions guest disks, runs QEMU |
| `tools/apic-dump` | Runs inside a guest as an ordinary process and prints its interrupt controllers over the hypercall |

## Getting started

### Prerequisites

- Rust nightly — pinned in `rust-toolchain.toml`; rustup installs it on demand.
- A C compiler and `libclang` — uACPI is compiled out of the submodule and its
  Rust declarations generated at build time.
- To run guests: Linux with QEMU (`qemu-system-x86_64`, `qemu-img`) and KVM on
  an AMD processor, so SVM reaches the guest.
- `swtpm` for Windows 11 guests, which require a TPM.

### Clone and build

```bash
git clone --recurse-submodules https://github.com/qwnd-real/pulzar-hypervisor
cd pulzar-hypervisor
cargo xtask build
```

In an existing clone, `git submodule update --init` fetches uACPI. The build
produces `target/x86_64-unknown-uefi/debug/` and stages the EFI system
partition as a plain directory under `dist/esp/`, served to QEMU through its
virtual-FAT driver — no filesystem image is ever built or committed.

The disk a guest OS boots from must carry Limine at
`\EFI\Limine\limine_x64.efi`: the loader preloads it and the portal starts it
inside the first guest, and Limine then chainloads the OS.

### Provision a guest disk (once per machine)

```bash
cargo xtask disk linux    # fetch a pinned, checksum-verified Debian 13 nocloud image
cargo xtask disk cachyos  # fetch the pinned CachyOS ISO, install interactively on bare firmware
cargo xtask disk windows --iso <windows-11.iso>  # blank disk, interactive Windows 11 setup, TPM included
```

Windows ISOs cannot be redistributed, so you supply your own. Disks, OVMF
firmware and TPM state live in a per-user cache (`~/.cache/pulzar`), never in
the repository, and survive re-clones.

### Run it

```bash
cargo xtask run --os linux
```

| Flag | Effect |
|---|---|
| `--os none\|linux\|cachyos\|windows` | Which guest disk to attach; `none` boots the hypervisor media alone |
| `--release` | Build with the release profile |
| `--gdb` | Expose QEMU's guest GDB stub on `127.0.0.1:1234` |
| `--silent` | Compile every log record out of both images, for a real machine |
| `--no-console` | Give the guest no debug console and no serial port at all |
| `--serial-log FILE` | Capture guest log output to a file (repeatable: debug console first, then COM ports) |
| `--no-hypervisor` | Boot the guest disk directly, pulzar taken out — the same machine and disk, to tell a fault of the hypervisor's from one the guest has anyway |

Every run gets an NVMe controller with a blank scratch disk, so the one
interposed device is exercised on every boot. The default machine is q35
with 8 GB of RAM and 8 logical processors (one socket, four cores, two
threads) — the SMT sharing is part of what the interrupt-controller
emulation has to get right.

### Tests

The library crates are target-agnostic and their unit tests run natively on
the host, no firmware needed:

```bash
cargo test -p spoof
cargo test -p vlapic
cargo test -p exits
```

## Roadmap

Device interposition beyond NVMe, following the pattern the NVMe driver sets:

| Feature | Status |
|---|---|
| Micro-kernel (paging, nested paging, exits, SMP, ACPI, PCI, ...) | ✅ Implemented |
| Full vlapic, both faces, with AVIC | ✅ Implemented |
| NVMe identity spoofing | ✅ Implemented |
| xHCI shadowing | ⬜ Planned |
| AHCI shadowing | ⬜ Planned |
| Ethernet NIC shadowing (Intel & Realtek first) | ⬜ Planned |
| Wi-Fi NIC shadowing | ⬜ Low priority / TBD |

## Contributing

`cargo clippy --all-targets` must pass with zero warnings (`clippy::pedantic`
is denied workspace-wide), `cargo fmt --all` must produce no diff, and every
`unsafe` block carries a `// SAFETY:` comment stating why it is sound at that
call site. Keep commits focused and self-contained.

## License

Distributed under the terms of the [MIT license](LICENSE). The uACPI
submodule under `third_party/uacpi` is fetched from its own repository and
carries its own MIT license.
