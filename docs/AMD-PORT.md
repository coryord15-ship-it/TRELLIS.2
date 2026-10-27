# TRELLIS.2 on AMD: the ROCm port, layer by layer

**Microsoft TRELLIS.2 (image → textured 3D) running end-to-end on an
AMD RX 9070 XT via ROCm: the PyTorch body unmodified, the six native
extensions on community HIP builds, and the components a recompile
can't fix replaced or corrected locally.**

Developed while porting TRELLIS.2 inference to an RX 9070 XT. Upstream
TRELLIS.2 is Linux + NVIDIA only (its README asks for an NVIDIA GPU with
24 GB and CUDA 12.4). This page documents how it runs on ROCm instead —
which parts needed no change, which parts the community had already
ported, and which parts were replaced locally. Provenance is stated per
layer; only the last layer is this port's own work.

## Status at a glance

| Layer / check | Status |
|---|---|
| PyTorch body | Unmodified on ROCm torch 2.11.0 (SDPA attention, FlexGEMM sparse convolution) |
| Six native extensions | Community HIP builds for gfx1201 (recipe credited in Layer 2) |
| CuMesh fill — R1 replacement | Proven: calibration zero failures (25 real chunks); corpus equivalence 2,355 chunks, 0 violations |
| Batch fill mode | Proven: 386,195 chunks in 1,024.1 s = 377.1 chunks/s (job 046) |
| Sectioned export | Proven (job 049, Oct 27, 2026): the extent-pin fallback (r35.9) fired once, on section 3, and the merged bounds verified full span on every axis. Export gate satisfied; publication proceeds on the author's confirmation |

## Tested environment

| Component | Version / value |
|---|---|
| Host | Windows + WSL2 Ubuntu 24.04 |
| CPU | Ryzen 9 9900X3D |
| GPU | RX 9070 XT 16 GB (gfx1201) |
| System RAM | ~32 GB (WSL capped at 24 GB, 8 GB swap, in the campaign runs) |
| ROCm | Baseline 7.2.1; campaign proof runs on 7.2.2 |
| PyTorch | 2.11.0 |

Full environment and build detail: `building-on-rocm.md`.

## In this bundle

- `generation-driver-changes.md` — every change to the generation
  driver, with its measured reason.
- `sectioned-export-findings.md` — the sectioned `to_glb` defects, the
  shipped r34 override, and the extent fix proven by job 049.
- `building-on-rocm.md` — bench, environment, assembly order, and the
  bench's standing gates.
- `known-issues-and-planned-work.md` — open bugs and planned work.
- The companion **CuMesh bundle** — the R1 fill replacement: design,
  proof, and failure catalog.

**Failures first:** what breaks on this stack lives in
`sectioned-export-findings.md`, the CuMesh bundle's
`failure-catalog.md`, and `known-issues-and-planned-work.md`.

## Layer 1 — the PyTorch body: no change

TRELLIS.2's Python is PyTorch throughout. ROCm's PyTorch build serves the
same `torch.cuda`-shaped API on AMD hardware, so the model code, the
pipelines, and the sampling logic run unmodified on ROCm torch (campaign
proof stack: torch 2.11.0, ROCm 7.2.2). Two backend swaps happen by
configuration rather than by forking model code:

- **Attention:** upstream's flash-attention path is CUDA-built. The port
  runs PyTorch's SDPA (`scaled_dot_product_attention`) on ROCm instead.
- **Sparse convolution:** routed through the FlexGEMM extension (below),
  which has a HIP build, instead of the CUDA-only sparse stack.

## Layer 2 — the six native extensions: community HIP builds

TRELLIS.2 builds six CUDA-source extensions: o_voxel, FlexGEMM, CuMesh,
nvdiffrast, nvdiffrec_render, and one more in its build tree. A working
end-to-end HIP build of all six for gfx1201 already existed in the
community, and the port stands on it:

- Recipe: `bioritmovideo/trellis2-rocm-gfx1201` — all six extensions
  compiled via HIP; GLB export with 2048² PBR textures working; the
  nvdiffrast renderer validated 110/110 on the recipe author's machine.
- HIP patch stack for the extensions: `egore/comfyui-trellis2-gguf-rocm`.
- Published benchmarks from the recipe author (Radeon AI PRO R9700, same
  gfx1201 silicon, 32 GB): 512 ≈ 56 s, 1024_cascade ≈ 81 s, **peak VRAM
  3.4 GB**. The stock pipeline's "24 GB minimum" is not a floor for this
  route — stages load selectively, and a 16 GB card has large headroom.
  (These are the recipe author's numbers, quoted for provenance; the
  campaign's own measured results are in the CuMesh bundle and in
  `generation-driver-changes.md`.)

Two substitutions ride along with the recipe:

- **Conditioner weights:** upstream's DINOv3 conditioner repo
  (`facebook/dinov3-vitl16`) is gated on Hugging Face; the recipe remaps
  it to an ungated mirror (PIA-SPACE-LAB), so first generation does not
  stop at a license gate.
- **fp32 GEMM defense:** gfx1201 has a known silent fp32 GEMM corruption
  defect (ROCm issue #6595 — large fp32 matmuls can return wrong
  results). The recipe mitigates with a chunked-GEMM patch and ships a
  reproducer; the port additionally forces rocBLAS over hipBLASLt
  (`TORCH_BLAS_PREFER_HIPBLASLT=0`) and leaves `expandable_segments`
  unset (it produced silent NaNs on gfx1201 in the port's spec profile).
  Output is scanned for NaN/Inf and GLB accessor bounds are checked —
  the corruption this guards against is silent, so the product is
  checked, not the demo.

## Layer 3 — what the community build does not fix (this port's changes)

Two CUDA-born components fail in ways a recompile does not cure, and one
operational property of the pipeline had to change. These are the changes
in this bundle and the CuMesh bundle:

1. **CuMesh `fill_holes`** — its upload path zero-fills every row ≥ 2²⁰
   on HIP, and one of its kernels constructs an invalid launch on a
   data-shaped input class. Replaced by **R1**, an independent HIP
   implementation of the same fill contract, proven by calibration +
   corpus equivalence (2,355 chunks: 2,352 strict-equal, 3 outcome-equal,
   0 violations). Full record: the CuMesh bundle.
2. **Per-chunk process toll in the fill seam** — the driver's crash-
   isolation design spawned one worker process per chunk at ~0.473 s of
   spawn + HIP init each, projecting ~40.6 h for a 386,195-chunk fill
   (measured, job 045). The batch worker mode (one process per chunk
   plan) brought the same fill to **1,024.1 s = 377.1 chunks/s**
   (job 046). See `generation-driver-changes.md`.
3. **Sectioned export defects** — upstream's sectioned `to_glb` cleanup
   bundle deletes large parts of real meshes on this output class
   (manifold-only component judging), and its budget simplify clips
   section extents. The r34 cleanup override is shipped (job 047); the
   extent fix (r35.9) is proven (job 049, Oct 27, 2026). The export
   gate is satisfied; publication proceeds on the author's
   confirmation. Full record:
   `sectioned-export-findings.md`.

## Delivery vehicle

WSL2 Ubuntu 24.04 on the Windows host, GPU via `/dev/dxg`
(`HSA_ENABLE_DXG_DETECTION=1`), extensions compiled for `gfx1201` only
(`GPU_ARCHS=gfx1201`, no fat binary). A Windows-native route (AMD's
PyTorch-on-Windows edition) exists for PyTorch itself, but there is no
published Windows-native AMD build of the six extensions, so WSL2 is the
route that exists. Details: `building-on-rocm.md`.

## Assembling it (quick path)

Summary of `building-on-rocm.md`'s assembly order — that page is the
recipe; this is the shape:

1. ROCm in the WSL distro; PyTorch + torchvision from the rocm wheel
   index.
2. Source trees: upstream TRELLIS.2 (snapshot `af44b45f…`), the
   community recipe, and its HIP patch stack.
3. The six native extensions via HIP (`GPU_ARCHS=gfx1201`).
4. Weights (`microsoft/TRELLIS.2-4B`); the conditioner loads via the
   recipe's ungated mirror remap.
5. The R1 fill worker (CuMesh bundle), gated by calibration +
   equivalence before the driver points at it; generation through
   the driver in `changes/`.

## Known limits of the port as it stands

- **nvdiffrast licensing.** The texture path currently runs the community
  HIP build of nvdiffrast for validation. nvdiffrast's license (NVIDIA
  Source Code License) limits derivative works to non-commercial use.
  The long-term shape is a clean-room implementation behind the same
  interface in a separate repository — the license's own definition
  excludes works that merely link to the interfaces — **not built yet**.
  Nothing in this bundle derives from nvdiffrast code.
- **One architecture proven.** Everything above is proven on gfx1201
  only. Other RDNA targets are untested, not claimed.
- **An unsupported second GPU must stay disabled.** gfx1010 (RX 5700 XT)
  was dropped from ROCm's supported list; a visible unsupported device
  in the same machine can trip ROCm's gfx1201 detection during runs.

## Credits

- Upstream: the **Microsoft TRELLIS.2** team, for the MIT release this
  port stands on.
- `bioritmovideo/trellis2-rocm-gfx1201` — the gfx1201 build recipe:
  all six extensions via HIP, the chunked-GEMM defense, and the
  published benchmarks quoted in Layer 2.
- `egore/comfyui-trellis2-gguf-rocm` — the HIP patch stack for the
  extensions.
- PIA-SPACE-LAB — the ungated conditioner mirror that keeps first
  generation from stopping at a license gate.

## License

MIT — the same license as upstream TRELLIS.2.
