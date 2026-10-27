# Bundle manifest — TRELLIS.2 (AMD/ROCm port)

**Upstream repo:** microsoft/TRELLIS.2
**License:** MIT (per the upstream README: "This model and code are released
under the MIT License"), with dependencies under separate terms — its README
names nvdiffrast / nvdiffrec explicitly.
**Base snapshot:** `af44b45f…` — only the prefix is recorded in the local port
records; the full commit hash is not. See GAPS.md.
**Weights:** microsoft/TRELLIS.2-4B (16.3 GB, Hugging Face).
**Fork shape (planned):** fork under the author's name, AMD port on a clearly
named branch. This bundle is the staging copy, not a submission.

## What this bundle is — and is not

TRELLIS.2 upstream is Linux + NVIDIA CUDA only. Running it on an RX 9070 XT
(gfx1201) took three layers, and this bundle keeps them separate:

1. **ROCm PyTorch body** — upstream's Python runs as-is on ROCm torch
   (torch's CUDA-shaped API is served by ROCm). No changes in this bundle.
2. **Community HIP patch stacks** for the six native extensions —
   `bioritmovideo/trellis2-rocm-gfx1201` (recipe) built on HIP patches from
   `egore/comfyui-trellis2-gguf-rocm`. Not our code; credited in
   `docs/AMD-PORT.md`, not copied here.
3. **Our changes** — the generation driver and its gates (chunked hole fill
   with the R1 HIP worker, batch fill, sectioned export with bounds
   verification, the r34 cleanup override), listed below.

Note on file roles: `generate.py`, `build-inside.sh`, and `r1-stage.sh`
are the port's own driver/harness files. Upstream TRELLIS.2 has no
`generate.py` — the driver wraps upstream's pipeline entry points. They are
bundled because they are where the port's behavioral changes live; the
posting map's earlier position (pipeline glue does not post) vs the
Oct 27 staging order (include the repo changes) is flagged in GAPS.md.

## Changed / added files (as they would be committed)

| File | Purpose | Proven by |
|---|---|---|
| `changes/generate.py` (sha256 66ea98e9…) | The port's generation driver at r35.8: extraction → chunked fill (planner, census, fallback ladder) → R1 worker seam → batch fill (r35.7) → sectioned export with per-section + merged bounds verification → r34 cleanup override (r35.8). | Jobs 044, 046, 047 |
| `changes/build-inside.sh` (e29f6441…) | Build/run harness inside the WSL bench: environment + census gates, R1 stage gates, generation phases, skip switches with banners. Stamp `2026-10-27-r35.8-r34-bounds-small`. | Jobs 044–047 |
| `changes/r1-stage.sh` (8db8db80…) | The R1 proof gate script: CPU logic tests, HIP compile, load probe, fixture battery, calibration (25 chunks), corpus equivalence sample (2,355 chunks), and (r35.7) the batch gate — byte-identical fixtures through `--batch`, poisoned-manifest exit-10 discipline. Exports the fill flags only when every gate passes. | Jobs 044, 046, 047 |
| `changes/r34-on-r33.patch` (be9d5292…) | The r34 change as a standalone patch against the r33 driver (4 hunks, 451 lines): the sectioned-export cleanup override + fragment census. Included so the delta can be reviewed without the full driver file. | Staged Oct 26; shipped in r35.8, job 047 |

The R1 fill worker sources themselves are in the **CuMesh bundle** — R1 is
a CuMesh-seam component that this driver calls.

## Docs in this bundle

- `docs/AMD-PORT.md` — the port route: what assumes CUDA, what replaces it,
  layer by layer, with provenance (upstream / community stacks / ours).
- `docs/generation-driver-changes.md` — the driver's changes: chunked fill
  and its fallback ladder, the batch seam, bounds verification.
- `docs/sectioned-export-findings.md` — the sectioned-export defect family:
  manifold-only component judging (r34, shipped), the extent clip
  (autopsy-convicted; fix proven by job 049, Oct 27, 2026 — export gate
  satisfied), and the crossing-sheet structure
  observed in extraction output, with census numbers.
- `docs/known-issues-and-planned-work.md` — open bugs, refuted items,
  planned work, and parked items for this bundle, with the publication
  gate stated up front.
- `docs/building-on-rocm.md` — the bench: WSL2 stack, environment,
  extension builds, and the gfx1201-specific defenses.
