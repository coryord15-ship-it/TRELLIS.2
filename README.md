# TRELLIS.2 on AMD — the ROCm port

**This fork:** upstream [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
is Linux + NVIDIA only (its README asks for an NVIDIA GPU with 24 GB
and CUDA 12.4). This fork documents and carries the changes that run
it **end-to-end on an AMD RX 9070 XT (gfx1201) via ROCm**: the PyTorch
body unmodified, the six native extensions on community HIP builds,
and the components a recompile can't fix replaced or corrected
locally.

The AMD work lives on this branch (`amd-port`); upstream is untouched
on the fork's upstream branch. The full layered write-up is
[`docs/AMD-PORT.md`](docs/AMD-PORT.md); every driver change and its
measured reason is in
[`docs/generation-driver-changes.md`](docs/generation-driver-changes.md).

## What works — proven, with the numbers

| Check | Result |
|---|---|
| End-to-end generation on RX 9070 XT | Working (ROCm torch 2.11.0, SDPA attention, FlexGEMM sparse conv) |
| CuMesh fill — R1 replacement (companion CuMesh fork) | Calibration zero failures; equivalence 2,355 chunks, **0 violations** |
| Batch fill mode (job 046) | **386,195 chunks in 1,024.1 s = 377.1 chunks/s** (vs ~40.6 h projected for the per-chunk design it replaced) |
| Sectioned export (job 049, Oct 27, 2026) | **PASS** — the extent-pin fallback fired once, on section 3, and the merged bounds verified **full span on every axis** |

## What's not working / known limits

- **Crossing-sheets extraction property:** upstream's sectioned
  cleanup judges components manifold-only, so a class of real output
  (thin perpendicular sheets crossing on shared edges) reads as
  thousands of fragments. Characterized in
  [`docs/sectioned-export-findings.md`](docs/sectioned-export-findings.md);
  the remedy belongs to the extraction rework, not this bundle.
- **torch is pinned to 2.11.0 under WSL:** newer builds (2.12.1+)
  abort at device enumeration in WSL (no `/sys/class/kfd`). The pin
  is deliberate and recorded in
  [`docs/known-issues-and-planned-work.md`](docs/known-issues-and-planned-work.md).
- **One architecture proven:** gfx1201 only. Other RDNA targets are
  untested, not claimed.

## What's left to do

- **Extent-preserving decimation:** the proven extent-pin fallback
  re-exports a failing section at near-full density (2.58M faces vs a
  ~300k budget) — correct first, optimized later.
- The remaining port queue (R2–R5), the V4 parallel proof harness's
  PC gate, and the CuMesh upstream-API packaging — all itemized in
  [`docs/known-issues-and-planned-work.md`](docs/known-issues-and-planned-work.md).

## Credits and license

Upstream: the Microsoft TRELLIS.2 team (MIT — this port is MIT too).
Community HIP builds this port stands on:
`bioritmovideo/trellis2-rocm-gfx1201` (gfx1201 build recipe) and
`egore/comfyui-trellis2-gguf-rocm` (HIP patch stack); PIA-SPACE-LAB
for the ungated conditioner mirror. Note: the texture path currently
validates against the community HIP build of nvdiffrast, whose NVIDIA
license limits derivative works — nothing in this bundle derives from
nvdiffrast code, and a clean-room replacement behind the same
interface is the planned shape (see `docs/AMD-PORT.md`).
