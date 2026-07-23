# AGENTS.md — pulzar

Operating manual for AI agents working in this repository. Read it fully before
writing or changing any code. Every rule below is binding unless the user
explicitly overrides it in the current conversation.

Pulzar is a type-1 hypervisor booted via UEFI. It has two parts: a UEFI loader
application (`hv-loader`) and the hypervisor image proper, which the loader
maps and transfers control to. This file is not a project summary — it defines
how you must behave and what the code must look like.

## 1. Prime directives

1. **Never resolve ambiguity silently.** If the task, the requirements, or your
   own understanding admit more than one reasonable interpretation, STOP and
   ask the user before implementing. Present the interpretations you see and
   what each would mean. Guessing and "picking the sensible one" is a failure,
   even if the guess turns out right.
2. **Touch only what the task requires.** Do not modify, "improve", reformat,
   or refactor code outside the scope of the request. Every changed line must
   trace directly to the user's request. If you notice unrelated problems
   (dead code, a bug, an ugly API), report them — do not fix them unprompted.
3. **Reuse before you write.** Before implementing anything, check (a) whether
   this workspace already has a function/type/module that does it, and
   (b) whether a well-maintained crate already provides it. Never hand-roll an
   assembly wrapper, bit-manipulation helper, or data structure that an
   existing dependency or existing project code already offers, and never add
   a second function that duplicates an existing one. If you believe existing
   infrastructure is inadequate, say why and ask before replacing it.
4. **No unfinished work in committed code.** No `todo!()`, no `unimplemented!()`,
   no `// TODO` markers, no stubbed-out branches, no silently omitted cases.
   Work is either complete, or its remaining part is surfaced to the user as an
   explicit, named open question — never buried in the code.
5. **No shortcuts on quality gates.** You may not suppress, downgrade, or
   disable a warning or lint to make a build pass. The only exception is a lint
   that is genuinely wrong for legitimate low-level systems code and has no
   compliant alternative; see §5 for the required procedure.
6. **Act as a senior Rust systems engineer.** Simplicity first, DRY, idiomatic,
   production-ready, performance-conscious. If a senior engineer would call
   your solution overcomplicated, rewrite it before presenting it.

## 2. Workspace shape

```
pulzar/
├── AGENTS.md            ← this file (CLAUDE.md is a symlink to it)
├── Cargo.toml           ← workspace root; lints and shared deps live HERE
├── rust-toolchain.toml  ← pinned nightly; do not float the channel
├── rustfmt.toml         ← formatting policy (uses unstable options → nightly)
├── .cargo/config.toml   ← default build target: x86_64-unknown-uefi
└── crates/
    └── hv-loader/       ← UEFI application: first-stage loader for the hypervisor
```

- `hv-loader` is a `no_std`/`no_main` UEFI PE application built for
  `x86_64-unknown-uefi`, using the rust-osdev `uefi` crate. It runs under
  firmware boot services; its job (eventually) is to locate, map, and jump into
  the hypervisor image.
- The hypervisor core will later live in its own crate(s) built for a custom
  freestanding target with `-Z build-std` — this is why the toolchain is
  nightly. Do not create those crates until asked.
- Future host-side tooling (e.g. an `xtask` runner) must pass
  `--target x86_64-unknown-linux-gnu` explicitly, because the workspace default
  target is UEFI.

## 3. Toolchain and build

Agreed decisions, encoded in the config files — change them only with explicit
user approval:

- **Nightly Rust, pinned to a date** in `rust-toolchain.toml` (never channel
  `"nightly"` floating). Required for future `-Z build-std` on the freestanding
  hypervisor target and for the unstable rustfmt options in use. To bump the
  pin: ask first.
- **Edition 2024, workspace-wide**, via `workspace.package`.
- **Dependencies are declared in `[workspace.dependencies]`** at the root and
  inherited by member crates with `{ workspace = true }`. `Cargo.lock` is
  committed.
- Member crates set `[lints] workspace = true`. Never define per-crate lint
  levels.

Canonical commands (rustup picks up the pinned toolchain automatically):

```sh
cargo build                      # produces target/x86_64-unknown-uefi/debug/hv-loader.efi
cargo clippy --all-targets       # must exit 0 with zero warnings
cargo fmt --all                  # must produce no diff on committed code
cargo fmt --all -- --check       # verification form
```

## 4. Coding style — hard rules

These are hard rules, not preferences:

- **No section-banner comments.** Never write `// ----- VMCB -----` or
  `// ===== Intercepts =====`. Structure the code (modules, functions, types,
  ordering) so it stays readable without banners. Order items top-down: public
  entry points first, helpers after, so a reader never needs a banner to
  navigate.
- **Every `unsafe` block requires a `// SAFETY:` comment immediately above
  it**, explaining why the operation is sound *at that call site* — the
  invariants that hold and who guarantees them — not a restatement of what the
  code does. Every `unsafe fn` documents its contract in a `# Safety` doc
  section. This is mechanically enforced (§5).
- **Every module gets a `//!` doc comment** describing its purpose and, where
  relevant, the invariants it upholds. **Every public item gets a `///` doc
  comment.** Comment anything a competent reader would otherwise find unclear
  — and nothing that is already obvious from the code.
- **No unclean code.** No dead code, no commented-out code, no debug leftovers,
  no orphaned imports. If *your* change makes something unused, remove it; do
  not remove pre-existing dead code unless asked (report it instead).
- **Idiomatic Rust throughout.** Prefer `Result`/`Option` combinators over
  manual matching where they read better. Use newtypes over bare integers for
  anything with semantic meaning (physical/virtual addresses, register field
  offsets, vector numbers). No magic numbers and no stringly-typed code — use
  named constants, `enum`s, and `bitflags`-style types.
- **`unsafe` stays confined.** Raw hardware access (MSRs, control registers,
  port I/O, raw pointers into physical memory, privileged instructions)
  belongs in dedicated low-level modules/crates and is exposed to the rest of
  the codebase through safe wrappers wherever a sound safe wrapper is
  possible. Higher-level code should very rarely contain `unsafe` directly.
- **No panics in steady-state hypervisor code.** Anything on the eventual
  VMEXIT/runtime path must not panic during normal operation — use `Result`
  and propagate. Panics (`expect`, asserts) are acceptable only during early
  boot/init inside `hv-loader` and hypervisor bring-up, before guest execution
  begins, where aborting the boot is the correct response to a broken
  invariant.
- **Performance is a requirement, not an afterthought** — but never at the
  cost of correctness or clarity without measurement. Avoid allocation and
  copying on hot paths; if you trade clarity for speed, justify it in a
  comment.

## 5. Lint policy — zero warnings, no exceptions by default

Enforced at the workspace level in the root `Cargo.toml`:

- `clippy::pedantic` is **deny**. All rustc warnings are **deny**. There is no
  "warnings allowed" mode: a build or clippy run that emits any warning or
  error is a failed build, full stop.
- `clippy::undocumented_unsafe_blocks` and `clippy::missing_safety_doc` are
  **deny** — they mechanically enforce the `SAFETY:` rule.
- `missing_docs` (rustc) is **deny** — it mechanically enforces public-item
  and crate docs.
- `clippy::allow_attributes_without_reason` is **deny** — every exception must
  carry a written reason.

When a pedantic lint is *actively wrong* for legitimate low-level work (e.g.
numeric-cast lints firing on intentional pointer/field-width truncation) and no
compliant rewrite exists:

1. Prefer `#[expect(lint_name, reason = "one-line why")]` on the **narrowest
   possible item** — never at module or crate level, never a blanket group
   downgrade.
2. Tell the user which lint you excepted and why, in your summary.

Suppressing a lint to hide a real problem, to save time, or because a fix is
inconvenient is prohibited.

## 6. Dependency policy

- Prefer established ecosystem crates over hand-rolled code: `uefi`
  (rust-osdev) for all UEFI protocol/boot-services work, and for future
  low-level work crates like `x86_64`, `raw-cpuid`, `bitflags`, `spin` — check
  what already exists before writing register/instruction wrappers by hand.
- Before adding a dependency: confirm nothing in the workspace or the existing
  dependency tree already covers the need, confirm it works in `no_std` for
  firmware-side crates, and state in your summary why it was added.
- Add dependencies to `[workspace.dependencies]` with an explicit version;
  enable only the features actually needed.
- Do not vendor, fork, or copy-paste code out of crates.

## 7. How to work (behavioral rules)

**Think before coding.**
- State your assumptions explicitly before implementing. Uncertain → ask.
- Multiple interpretations → present them; do not pick silently (§1.1).
- If a simpler approach than the requested one exists, say so — push back
  when warranted, before writing code.

**Simplicity first.**
- Minimum code that solves the stated problem. No speculative features,
  no abstractions for single-use code, no unrequested configurability,
  no error handling for impossible states.

**Surgical changes.**
- Match the existing style even where you would personally differ.
- Do not reformat or restructure untouched code.
- Clean up only the orphans your own change created.

**Goal-driven execution.**
- Turn every task into a verifiable goal before starting ("builds a working
  `.efi`", "clippy clean", "boots in QEMU and prints X") and state a brief
  step → verify plan for multi-step work.
- Loop until the verification actually passes; never claim success you have
  not observed. If a gate fails and you cannot fix it within the rules above,
  report the failure honestly with the output.

## 8. Definition of done

A change may be presented as finished only when ALL of the following hold:

1. `cargo build` succeeds with **zero** warnings.
2. `cargo clippy --all-targets` succeeds with **zero** warnings.
3. `cargo fmt --all -- --check` reports no diff.
4. No `unsafe` block lacks a `SAFETY:` comment; no public item or module lacks
   docs (the lints in §5 verify this mechanically — do not rely on memory).
5. No TODOs, stubs, dead code, or commented-out code were introduced.
6. Any lint exception added is `#[expect(..., reason = "...")]`, maximally
   narrow, and disclosed in your summary.
7. The diff contains nothing unrelated to the request.

If any item fails, the task is not done — say so explicitly.
