# Sectioned export: the defect family, in order of conviction

*Part of the TRELLIS.2 AMD bundle — start at `AMD-PORT.md`.*

The sectioned GLB export (bands of ≈10⁶ faces each through `to_glb`'s
cleanup bundle, then merged) lost geometry on real output in three
distinct ways. This page records each defect, the instrument that named
it, and its status. Numbers are from the campaign's runs (jobs 028, 046,
047, 049) and a follow-up evidence autopsy of job 047's artifacts, all on the
same subject mesh family.

## Defect 1 — manifold-only component judging deletes the mesh (r34, shipped)

**Symptom.** Job 028's subject export lost up to 96.1% of its upper
sections inside `to_glb`'s cleanup bundle while reporting success.

**Mechanism.** The bundle's `remove_small_connected_components(1e-5)`
judges connectivity *manifold-only* — the state the preceding repair op
leaves behind. Sections of generated meshes are held together in large
part by edges shared by 3+ faces; under manifold-only judging such a
section shatters into sub-threshold fragments and the op deletes them as
"small components". Job 028's census: section 3 read as **338,440
manifold-only components, largest fragment 4.27%**; section 4 as 40,533
components, largest 332 faces. Under ordinary any-edge connectivity the
same section 3 is 97.0% a single component. The op was deleting the
mesh, not dust in it.

**The change (r34, in `changes/r34-on-r33.patch`).** During each
section's `to_glb`, the cleanup bundle becomes an explicit op sequence
**without** the small-component op. The other three ops
(`remove_duplicate_faces`, `repair_non_manifold_edges`, `fill_holes`)
run stock, in the bundle's own order, at both cleanup passes. The 1e-5
threshold is unchanged — it was never the defect. Interception is a
context manager around the sectioned `to_glb` call, because the bundle
runs inside upstream's function on a CuMesh instance: route A patches
the CuMesh class; route B, if the compiled class refuses, swaps the
`cumesh` reference inside `o_voxel.postprocess` for a delegating shim.
Originals are restored on exit either way. A patchability probe runs
before the section loop; if neither route installs, the run fails loud
(exit 5) — there is no silent stock-chain fallback. The non-sectioned
stock path is untouched.

**Instrumentation shipped with it.** Before each section's export, a
`[r34-census]` line logs the section's manifold-fragment census
(component count, largest-fragment share, share of faces touching a
non-manifold edge — the defect's three predictor numbers); every op at
both passes logs before/after face counts under `[r34-ops]`, with the
removed op logged as SKIPPED with its threshold.

**Verification.** The census code was checked against job 028's actual
raw GLBs before shipping: all five sections reproduced the audit's
published manifold-only numbers exactly (sec3: 338,440 components,
largest 41,500 faces = 4.27%, non-manifold touch 84.71%; sec4: 40,533,
largest 332 = 0.33%, 93.24%). The cleanup-override harness passes route
A, route B, and the both-blocked fail-loud case.

**Job 047 result.** The override ran exactly as designed (probe route
A; the op SKIPPED at both passes in every section) — and the merged
export **still failed bounds, with numbers identical to job 046**
(axis 1: exported 0.9073 vs intended 0.9964, short 0.0890). The fix's
mechanism was refuted as the *sole* cause: the component op was eating
geometry, but it was not what trips the bounds gate. That is defect 2.

## Defect 2 — budget simplify clips a section's extent (convicted; fixed and proven, job 049)

Job 047's census showed section 3/4 failing its *own* per-section
bounds gate (axis 2: exported 0.1568 vs intended 0.1714, short 0.0146,
tol 0.0086) and being skipped; the merged output is then partial
(3 sections, 930,439 verts), which is what fails the merged gate by
exactly job 046's 0.0890. The autopsy of 047's artifacts convicted the
stage, by re-slicing the untouched raw mesh with the run's recipe and
reproducing all four `[r34-census]` lines exactly:

- The section *input* is full-size (extent 0.1714 — the gate's
  "intended" is the input's own bounds). No shortfall exists at the
  split. The loss appears only in the post-`to_glb` section GLB.
- The loss is **entirely one-sided**: the low side of the range is
  preserved to 2×10⁻⁵; the high side is cut by the full 0.0146. The
  lost geometry is **312 faces / 227 verts**, clean — 0.0% of them
  touch a non-manifold edge — a small protrusion at the band's
  extremity.
- Inside `to_glb`, the only vertex-removing stages are the two
  simplifications, and they bracket cleanup #1's `fill_holes`
  explosion on this section (902,997 → **2,510,806 faces**, +1.61M
  synthetic faces), after which the budget simplification cuts toward
  the section's decimation target (~301,722 faces; exported 293,269).
  A 312-face clean protrusion is exactly the class of geometry such
  a cut spends first. The log prints counts only inside `to_glb`
  (no per-stage bounds), so the record bounds the suspects to the
  simplify chain rather than naming one op.

**Status: FIXED + PROVEN.** The fix (r35.9 — an extent-pin through
the section simplify: a section that fails its bounds gate is
re-exported once with the simplify chain neutralized, and a section
that still fails stops the run loudly instead of merging partial)
was proven by job 049 (Oct 27, 2026), a run whose merged export
cleared bounds verification with every section present. The result
is recorded in the closing section of this page.

## Defect 3 — the extraction output itself is crossing sheets (observed property)

The same autopsy measured what kind of non-manifold structure
generated output actually carries, on job 047's mesh (3,180,133 faces,
bands graded by height):

| band | faces | manifold-only components | largest fragment | faces touching a non-manifold edge |
|---|---|---|---|---|
| 0 (bottom) | 777,535 | 3,097 | 34.74% | 3.2% |
| 1 | 987,287 | 18,177 | 61.27% | 6.5% |
| 2 | 837,382 | 69,186 | 51.74% | 17.1% |
| 3 (top) | 959,541 | 383,256 | 4.83% | **88.7%** |

Structure of the top band (959,541 faces, 1,100,776 unique edges):

- Edge incidence: 1 → 7.3% of edges; 2 → 56.0%; 3 → 4.6%; **4 → 32.1%;
  5+ → 0**. Maximum incidence anywhere in the mesh is exactly 4, and
  98.2% of all incidence-≥4 edges lie in the top band.
- Around a sampled incidence-4 edge, the four faces sit at a median
  minimum angular gap of **88.3°** — a near-perfect cross: **two
  perpendicular thin sheets sharing an edge**, repeated across the
  band (Cartesian character, steepest where the figure is thin).
- Degenerate faces: 0. Exact duplicate faces: 0. Duplicated vertex
  positions: 0 anywhere in the mesh (1,627,189 verts = 1,627,189
  unique positions) — welding is a provable no-op; a weld-then-census
  reproduces the raw census byte-identically.
- Under any-edge connectivity the band is 3,151 components with the
  largest at **97.37%** — one connected object. The 383,256
  "fragments" exist only under manifold-only judging.
- Attribution: the fill stage kept only 145,633 faces mesh-wide
  (4.6%), so fill fans cannot be the main carrier; the structure is
  overwhelmingly a property of the extraction output the fill
  received. Per-face provenance is not recorded in exported
  artifacts, so the conviction is stated at the extraction/fill-output
  level.

Consequences for downstream code, stated plainly:

- Manifold-only component judging is invalid on this output at any
  threshold (defect 1's root).
- `repair_non_manifold_edges` is a no-op against crossing sheets
  (0 faces changed in both of job 047's passes).
- `fill_holes` reads the crossing lattice's cells as holes — the
  +1.61M-face explosion behind defect 2.
- Any remedy must act on the extraction/fill output or on how
  downstream stages judge it; no cleanup-bundle change cures the
  structure itself. That work is open, not scheduled in this bundle.

One measurement trap, recorded so nobody re-walks it: exported section
GLBs store triangles as chart-separated clusters (xatlas splits
vertices — sec3's GLB holds 563,261 vertex records over 171,759 unique
positions), so a census run on a stored section GLB measures the
storage representation, not the mesh. It reads identically for passing
and failing sections.

## Result — job 049: the extent pin holds (Oct 27, 2026)

Job joe-20261027-049 (build stamp `2026-10-27-r35.9-extent-pin`)
ran the same subject and export shape as jobs 046/047: DONE, wall
2,486.9 s, exit code 0. The run reproduced defect 2 and then cured
it, in that order:

- Sections 0, 1, and 2 passed their per-section bounds gates on
  the first attempt. The r34 cleanup override
  (`remove_small_connected_components` skipped) fired in all four
  sections.
- Section 3's first (decimated) export failed its gate with job
  047's numbers exactly: axis 2 exported extent 0.1568 vs intended
  0.1714 (short 0.0146, tol 0.0086).
- The extent-pin fallback fired exactly once, on section 3:
  re-export with decimation_target=16777216 (2²⁴), which
  neutralizes the simplify chain. The re-export held the input
  extents on every axis — log line: "bounds gate PASS
  (extent-pin fallback)". Section 3 exported at 3,868,431 verts /
  2,580,957 faces.
- `[r35.9-extent]` per-stage extent tracing was present for all
  sections (no `extents=unavailable` degradation).
- Merged verification: "merged bounds verified against the raw
  cascade mesh - full span on every axis" — the first clean merged
  export of the campaign. Merged `subject-1024cascade.glb`:
  263,322,892 bytes.

Known cost, stated honestly: the fallback exports the failing
section at near-full density (2,580,957 faces vs the ~300k-face
decimation budget), so the merged GLB is large (251.12 MiB). That
is correctness-first behavior. The planned refinement is an
extent-preserving decimation — keep the extremes, decimate the
interior — recorded in `known-issues-and-planned-work.md`
(defect 2).

Artifact note: the `subject-1024cascade-bounds-fail.glb` present
in job 049's results folder is job 047's stale artifact
(83,271,152 bytes, dated Oct 27 14:39), swept in by the package's
collect step — not a new failure.
