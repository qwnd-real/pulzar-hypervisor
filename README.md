# Pulzar Hypervisor

[![License](https://img.shields.io/github/license/yourusername/pulzar?style=flat-square)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/yourusername/pulzar/ci.yml?branch=main&style=flat-square)](https://github.com/yourusername/pulzar/actions)
[![Rust](https://img.shields.io/badge/rust-nightly-orange?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![Issues](https://img.shields.io/github/issues/yourusername/pulzar?style=flat-square)](https://github.com/yourusername/pulzar/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)](CONTRIBUTING.md)

A minimal pass-through type-1 hypervisor for x86_64 with nested virtualization support.

Pulzar continues the UEFI boot cycle transparently, jumping into the UEFI context inside a virtualized environment and handing off to the next bootloader (Windows, Linux). The guest OS then runs entirely under Pulzar's control, with device access shadowed at the hypervisor level.

## Table of Contents

- [How It Works](#how-it-works)
- [Feature Roadmap](#feature-roadmap)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)

## How It Works

Pulzar boots as a UEFI application, initializes a minimal hypervisor context, and re-enters the UEFI boot cycle inside a virtualized guest. From there, the normal OS boot process (Windows or Linux) proceeds unmodified, with Pulzar transparently intercepting and shadowing hardware access as needed.

## Feature Roadmap

| Feature | Status |
|---|---|
| Micro-kernel (PCI, ACPI, IPIs, paging, ...) | ✅ Implemented |
| Full VLAPIC implementation | ⬜ Planned |
| xHCI shadowing | ⬜ Planned |
| NVMe shadowing | ⬜ Planned |
| AHCI shadowing | ⬜ Planned |
| Ethernet NIC shadowing (Intel & Realtek priority) | ⬜ Planned |
| Wi-Fi NIC shadowing | ⬜ Low priority / TBD |

## Getting Started

### Prerequisites

- Linux host
- [QEMU](https://www.qemu.org/)
- Rust toolchain (nightly)

### Testing with Windows

Windows ISOs are subject to Microsoft licensing, so `xtask` cannot download one for you — you'll need to supply your own.

**1. Provision the disk (one-time setup):**

```bash
cargo xtask disk windows --iso <path-to-windows.iso>
```

**2. Boot Windows under Pulzar:**

```bash
cargo xtask run --os windows
```

### Testing with Linux

Linux distributions can be fetched directly from a mirror, so no ISO is required.

**1. Provision the disk:**

```bash
cargo xtask disk linux
```

**2. Boot Linux under Pulzar:**

```bash
cargo xtask run --os linux
```

## Contributing

Contributions are welcome. Before opening a pull request:

- Ensure `cargo clippy` passes with no warnings
- Ensure `cargo fmt` has been run
- Keep commits focused and well-described

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

## License

Distributed under the terms of the [LICENSE](LICENSE) file.