# Generation driver changes (generate.py)

*Part of the TRELLIS.2 AMD bundle — start at `AMD-PORT.md`.*

The driver is the port's generation code around upstream TRELLIS.2's
pipelines: extraction of a voxel mesh, hole filling, and GLB export.
Upstream's pipeline is not edited; the driver owns everything between
extraction output and a finished GLB, and it is where the port's
behavioral changes live. `changes/generate.py` is the driver at r35.8
(proof stack: jobs 044, 046, 047); `changes/r34-on-r33.patch` isolates
the r34 delta against the r33 driver.

## Chunked fill and the fallback ladder

Extraction output is a large mesh (campaign figures: 1,602,346 voxels,
34 chunks, 3,034,500 genuine faces in the subject arm's extraction).
The driver plans a chunked fill rather than feeding the whole mesh to one
fill call:

- **Planner budgets:** 131,072 faces and 262,144 verts per chunk; a
  real subject plan runs to 386,195 chunks (cascade arm) or 373,812
  chunks (control arm), 4,348 chunks for the canvas arm.
- **Census first:** a boundary census (boundary edges, boundary
  components, per-component perimeters) identifies candidate hole
  components and which chunk vertices are fillable interior.
- **The seam:** each chunk goes to the R1 fill worker as a standalone
  `.npz` (`verts` float32, `faces` int64, chunk-local indexing;
  `max_hole_perimeter = 3e-2` in raw mesh units).
- **The ladder:** whole chunk → two halves → unfilled fallback. A chunk
  that survives no rung is returned unfilled (documented in the run
  stats as a fallback chunk), never synthesized. Campaign runs report
  **0 fallback chunks** on every filled arm.
- **Parent vetting:** after the worker returns, the driver keeps only
  fill faces whose vertices are fillable interior; faces across cut
  boundaries are dropped by design (they belong to the kernel's lobe-wise
  fill that the parent's summed census rejects — see the CuMesh bundle).
  Job 046's subject arm kept 3,180,133 / 3,180,133 faces and
  1,627,189 / 1,627,189 verts genuine, with a 0.00% filter drop.

## Batch fill (r35.7)

The per-chunk subprocess seam costs ~0.473 s of spawn + HIP init per
chunk regardless of chunk size (measured, job 045: 2.64 chunks/s,
~40.6 h projection for the subject plan). The driver now optionally
pre-fills the whole chunk plan through one worker process:

- Activated only when both `R1_FILL_WORKER` and `R1_FILL_BATCH` are set;
  unset either and behavior is byte-for-byte the per-chunk path.
- The batch callback stages the plan as spool `.npz` files + manifest,
  runs one worker over the manifest, relays its progress lines as the
  same per-chunk log lines, and validates every output with the same
  append-only floor check as the single-shot seam.
- A batch failure costs only its in-flight chunk: that chunk takes the
  untouched single-shot ladder, the remainder re-batches in a fresh
  process, and three no-progress rounds abandon batching wholesale.
- Measured (job 046): subject arm **386,195 chunks in 1,024.1 s =
  377.1 chunks/s** wall (387.7 start→last-end), mean 2.8 ms per chunk —
  ~143× the per-chunk seam's pace. Job 047 reproduced 376.2 chunks/s
  (1,026.5 s) after the batch gate re-ran in full. Fill math is
  unchanged between the two modes; the batch gate byte-compares every
  harness fixture through `--batch` against its single-shot output
  before the batch flags are exported (see `changes/r1-stage.sh`).

## Bounds verification (both arms)

Two gates were added around export, and both are loud:

1. **Per-section gate** (sectioned export): each section's exported GLB
   is checked against the section input's own vertex bounds, mapped
   into GLB space. A section that fails is skipped from the merge,
   and the skip is logged (`[sections] section N FAILED the bounds
   gate - SKIPPED`).
2. **Merged gate:** the merged GLB is checked against the full-mesh
   intended extents. On failure the clipped file is kept as
   `*-bounds-fail.glb` with a `*.BOUNDS-FAIL.txt` stating the failing
   axis, exported vs intended extent, and tolerance — and the run
   does not claim a clean export.

These gates are what made the sectioned-export defects below visible
as numbers instead of silent output loss.

## Sectioned export

For large meshes the driver exports in horizontal bands (4 sections for
the subject cascade arm: cuts at face-count boundaries [0, 713933,
1574016, 2284194, 3180133] with a 63,602-face overlap margin), each
band through `to_glb`'s cleanup bundle, then merges. The r34 cleanup
override and its instrumentation live in this path — see
`sectioned-export-findings.md`.
