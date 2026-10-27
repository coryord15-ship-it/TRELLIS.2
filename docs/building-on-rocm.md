# Building and running the port on ROCm

*Part of the TRELLIS.2 AMD bundle — start at `AMD-PORT.md`.*

Proven bench: Windows host + **WSL2 Ubuntu 24.04**, Ryzen 9 9900X3D,
**RX 9070 XT 16 GB (gfx1201)**, ~32 GB system RAM (WSL capped at 24 GB
with 8 GB swap in the campaign runs). Validated baseline stack:
ROCm 7.2.1; campaign proof runs: **ROCm 7.2.2, torch 2.11.0** (rocm
wheel index), hipcc HIP 7.2.53211.

## Environment

```bash
export GPU_ARCHS=gfx1201                 # compile RDNA 4 only, no fat binary
export HSA_ENABLE_DXG_DETECTION=1        # GPU visibility through WSL2 /dev/dxg
export TORCH_BLAS_PREFER_HIPBLASLT=0     # force rocBLAS (see GEMM defense)
# leave PYTORCH_CUDA_ALLOC_CONF's expandable_segments UNSET on gfx1201:
# it produced silent NaNs in this port's testing.
```

## Assembly order

1. ROCm install in the WSL distro (`graphics,rocm` usecase, no DKMS —
   the kernel side is the Windows Adrenalin driver; validated on the
   26.1.1-era driver and later).
2. PyTorch + torchvision from the official rocm wheel index matching
   the ROCm version.
3. Source trees: upstream TRELLIS.2 (snapshot `af44b45f…`), the
   community recipe (`bioritmovideo/trellis2-rocm-gfx1201`), and its
   HIP patch stack (`egore/comfyui-trellis2-gguf-rocm`).
4. Native extensions via HIP — all six: o_voxel, FlexGEMM, CuMesh,
   nvdiffrast, nvdiffrec_render, +1. CuMesh compiles but is not used
   for filling once R1 is in place (see the CuMesh bundle); the build
   is long and quiet (the recipe author's machine took ~70 min serial
   for CuMesh alone) — budget for it.
5. Weights: `microsoft/TRELLIS.2-4B` (16.3 GB). The DINOv3 conditioner
   loads from the recipe's ungated mirror remap at first generation,
   not from the gated upstream repo.
6. The R1 fill worker (CuMesh bundle) built with
   `--offload-arch=gfx1201`, probed, and gated before the driver
   points at it.
7. Generation runs through the driver (`changes/generate.py`) with the
   gate script (`changes/r1-stage.sh`) having exported the fill flags
   only after calibration + equivalence pass.

## gfx1201-specific defenses

- **Silent fp32 GEMM corruption (ROCm issue #6595):** large fp32
  matmuls on gfx1201 can return wrong results without any error. The
  recipe's chunked-GEMM patch is the first mitigation; the port runs
  the recipe's reproducer and logs its output either way, forces
  rocBLAS with `TORCH_BLAS_PREFER_HIPBLASLT=0`, and scans generated
  vertices for NaN/Inf plus GLB accessor bounds on every run. Judge
  the product, not the demo.
- **VRAM is not the constraint the upstream README implies:** the
  recipe's published peak is 3.4 GB (1024_cascade) on same-silicon
  32 GB hardware; the 16 GB card runs the full pipeline with headroom.
  The "24 GB minimum" belongs to the CUDA build's loading pattern.
- **Disable unsupported second GPUs during runs:** an RX 5700 XT
  (gfx1010) in the same machine is outside ROCm's supported list and
  a visible unsupported device can trip gfx1201 detection. There is
  no driver path that gives gfx1010 a job in this stack.
- **GPU residency gate:** a run is only believed when the device name
  reads RX 9070 XT *and* peak VRAM during generation is > 1 GB with
  generation time in the published envelope. A silent CPU fallback
  shows ~0 GB VRAM and ~10× time; the gate exists to catch exactly
  that.

## What "the port works" must prove (the bench's standing gates)

1. GPU residency, as above.
2. No corruption signature (NaN/Inf scans, finite GLB bounds).
3. Structural truth: the GLB parses, contains real geometry, embeds
   its PBR images.
4. Fill correctness: calibration 25/25 and corpus equivalence with
   0 violations, re-run in full whenever the fill worker binary
   changes (the campaign re-ran both for the batch-mode drop even
   though the fill math was untouched).
5. Human comparison of renders against the input and a trusted
   hosted reference run — same-seed equality across hosts is not
   expected; subject fidelity and absence of corruption artifacts
   is the bar.
