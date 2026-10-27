# Known issues and planned work — TRELLIS.2 AMD/ROCm port

Everything in this file is open, refuted, planned, or parked — nothing
here is presented as working that has not passed a run. Items are
stated at exactly the strength the campaign record supports; numbers
are the record's own.

## Publication gate

**Export gate satisfied (job 049, Oct 27, 2026); publication
proceeds on the author's confirmation.** The merged export failed
bounds verification (defect 2, below) through job 047. The fix for
that defect (r35.9, the extent-pin fallback) was proven by job 049 —
a small one-arm confirmation test of the same shape as job 047:
section 3's first export reproduced the defect exactly, the
fallback re-export passed its gate, and the merged export cleared
bounds verification with every section present ("merged bounds
verified against the raw cascade mesh - full span on every axis").
Full numbers in defect 2's resolution, below.

## Open bugs

### 1. Budget simplify clips a section's extent (defect 2 — convicted)

- **What:** In the sectioned export, a section can pass the split
  full-size and still be exported short: the per-section bounds gate
  then skips it, and the merged output — missing that section — fails
  merged bounds verification.
- **Evidence (job 047, same subject mesh as 046):** section 3's own
  gate read axis 2: exported extent 0.1568 vs intended 0.1714 (short
  0.0146, tol 0.0086); the section was skipped; the partial merge
  (3 sections, 930,439 verts) failed the merged gate on axis 1:
  exported 0.9073 vs intended 0.9964 (short 0.0890) — identical to
  job 046's merged failure, because the same section is excluded
  every run.
- **Conviction (evidence autopsy of 047's artifacts):** the section
  input is full-size (0.1714 is the input's own bounds; no shortfall
  at the split). The loss appears only in the post-`to_glb` section
  GLB and is entirely one-sided — the low side of the range is
  preserved to 2×10⁻⁵, the high side cut by the full 0.0146. The
  lost geometry is 312 faces / 227 verts, clean (0.0% touch a
  non-manifold edge). Inside `to_glb` the only vertex-removing
  stages are the two simplifications; they bracket cleanup #1's
  `fill_holes` growth on this section (902,997 → 2,510,806 faces,
  +1.61M synthetic faces), after which the budget simplification cut
  toward the section's decimation target (~301,722 faces; exported
  293,269). The record bounds the suspects to the simplify chain; it
  does not name one op.
- **Status:** FIXED + PROVEN (job 049, Oct 27, 2026) — see the
  resolution below. (Stage convicted, single op not convicted —
  that part of the record stands; the fix does not depend on
  naming the single op.)
- **Attempt (r35.9):** pin the vertices that realize the
  section's extents through the section simplify, log the extent
  after every stage inside `to_glb` (so a future cut names its
  stage in the same run), and make a skipped section fail the run
  loudly instead of merging partial.
- **Resolution (job 049, Oct 27, 2026):** run DONE, wall
  2,486.9 s, exit code 0, build stamp
  `2026-10-27-r35.9-extent-pin`. Section 3's first (decimated)
  export failed its bounds gate, reproducing job 047's failure
  exactly: axis 2 exported extent 0.1568 vs intended 0.1714
  (short 0.0146, tol 0.0086). The extent-pin fallback fired
  exactly once (section 3 only): re-export with
  decimation_target=16777216 (2²⁴, simplify chain neutralized),
  which held the input extents on every axis — log line
  "bounds gate PASS (extent-pin fallback)". Section 3 exported
  at 3,868,431 verts / 2,580,957 faces. Sections 0, 1, and 2
  passed their gates on the first attempt; the r34 cleanup
  override fired in all four sections; `[r35.9-extent]`
  per-stage tracing was present for all sections (no
  `extents=unavailable` degradation). The merged verification
  line reads "merged bounds verified against the raw cascade
  mesh - full span on every axis" — the first clean merged
  export of the campaign; merged `subject-1024cascade.glb` is
  263,322,892 bytes. (The `subject-1024cascade-bounds-fail.glb`
  in job 049's results folder is job 047's stale artifact —
  83,271,152 bytes, Oct 27 14:39 — swept in by the package's
  collect step, not a new failure.)
- **Known cost / planned refinement:** the fallback exports the
  failing section at near-full density (2,580,957 faces vs the
  ~300k-face decimation budget), so the merged GLB is large
  (251.12 MiB). That is correctness-first behavior and is
  accepted as such; the planned refinement is an
  extent-preserving decimation — keep the extremes, decimate
  the interior — so a fallback export lands near budget size
  without re-opening the clip.

### 2. Torch version ceiling under WSL (upstream bug class)

- **What:** The stack is pinned to torch 2.11.0 because newer ROCm
  wheels abort at device enumeration inside WSL2: 2.12.1, 2.13.0,
  and 2.14.1 all fail a rocprofiler-sdk check (0 KFD agents vs 2 HSA
  agents — WSL has no `/sys/class/kfd`). 2.11.0 passes the gate
  (GPU visible, matmul correct). The ceiling is a WSL artifact, not
  an AMD one; the port author's 2.13.0 stack was validated on bare
  metal, where KFD exists.
- **Cost:** every torch fix and kernel improvement since 2.11 is
  unreachable until a wheel clears the gate.
- **Status:** open (upstream defect class; no local workaround
  exists — emptying `HSA_TOOLS_LIB` does not stop the abort).
- **Planned attempt:** retest each new ROCm-index wheel in a
  throwaway venv (device count does not abort, matmul correct, one
  512 generation inside the published time envelope). Pass/fail per
  wheel, no commitment implied.

### 3. Base snapshot hash is a prefix only

- **What:** The upstream base is recorded in the port records only
  as `af44b45f…`; the full commit hash appears in no source read.
- **Status:** open (record gap).
- **Planned attempt:** pin the full hash before publication, so the
  fork's delta has a named baseline.

## Observed properties with open remedies

### 4. Extraction output is perpendicular crossing sheets

- **What:** Generated meshes arrive at the export already carrying
  dense non-manifold structure: thin sheets crossing on shared
  edges at ~90°. It is a property of the extraction/fill output, not
  of any downstream stage — re-slicing the untouched raw mesh with
  the run's recipe reproduced job 047's per-section census exactly,
  and the fill stage kept only 145,633 faces mesh-wide (4.6% of
  3,180,133), so fill fans cannot be the main carrier.
- **Evidence (job 047's mesh, bands graded by height):** faces
  touching a non-manifold edge climb 3.2% → 6.5% → 17.1% → **88.7%**
  from bottom band to top. Top band (959,541 faces, 1,100,776 unique
  edges): edge incidence 1 → 7.3%, 2 → 56.0%, 3 → 4.6%, **4 → 32.1%,
  5+ → 0** — the mesh-wide maximum incidence is exactly 4, and 98.2%
  of all incidence-≥4 edges lie in the top band. Around a sampled
  incidence-4 edge, the four faces sit at a median minimum angular
  gap of 88.3° — a near-perfect cross. Degenerate faces: 0. Exact
  duplicate faces: 0. Duplicated vertex positions: 0 (1,627,189
  verts = 1,627,189 unique positions; welding is a provable no-op).
  Under any-edge connectivity the top band is 3,151 components with
  the largest at 97.37% — one connected object; its 383,256
  "fragments" exist only under manifold-only judging.
- **Consequences already measured:** manifold-only component
  judging is invalid on this output at any threshold (that was
  defect 1, fixed by r34 — see `sectioned-export-findings.md`);
  `repair_non_manifold_edges` is a no-op against crossing sheets
  (0 faces changed in both of job 047's passes); `fill_holes` reads
  the crossing lattice's cells as holes (the +1.61M-face growth
  behind defect 2).
- **Status:** open — observed property, no cure claimed.
- **Planned attempt:** none scheduled in this bundle. Any remedy
  must act on the extraction/fill output itself or on how
  downstream stages judge it; no cleanup-bundle change cures the
  structure. The HIP-native o_voxel replacement (item 11) is the
  venue where a structural remedy would land if pursued there.
  Expect the same shape on any subject whose top band is thin
  geometry — the gradient is a property of the generated output,
  not of one subject.

## Refuted

### 5. The r34 mechanism as the sole cause of the bounds failures

- **What was believed:** that `remove_small_connected_components`
  eating geometry (defect 1) was what tripped the bounds gates.
- **What the run showed:** job 047 ran the r34 override exactly as
  designed (probe route A; the op skipped at both cleanup passes in
  every section) and the merged export still failed bounds with
  numbers identical to job 046 (axis 1 short 0.0890). The component
  op was eating geometry, but it was not what trips the gate — that
  is defect 2 (item 1).
- **Status:** refuted as the sole cause. Full record in
  `sectioned-export-findings.md` (defects 1 and 2); not duplicated
  here.

## Planned work

### 6. V4 — parallel proof harness (built, unshipped, PC gate owed)

- **What:** The r1-stage proof gauntlet (logic tests, calibration,
  corpus equivalence) is near-serial and leaves most of a 12-core
  CPU idle. V4 shards the CPU test chunks across a per-core process
  pool of long-lived workers and merges verdicts by chunk index
  (deterministic). The referee math is untouched — the pool shards
  *around* the fill logic, never threads it.
- **Status:** planned — built and proof-checked off-machine; not yet
  shipped in any drop.
- **Gate before it may gate a run:** it must reproduce the serial
  harness's verdicts *exactly* on the frozen 2,355-chunk corpus
  sample — the same strict-equal / outcome-equal / violation counts
  (job 046's full serial run: 2,352 strict-equal, 3 outcome-equal,
  0 violations). That PC-side gate is still owed.

### 7. GEMM defenses — correctness bought, price unmeasured

- **What:** Generation runs with `TORCH_BLAS_PREFER_HIPBLASLT=0`
  (rocBLAS forced over hipBLASLt) plus a chunked fp32 GEMM patch
  that mitigates a real gfx1201 defect (ROCm issue #6595, silent
  wrong results on large fp32 matmuls; its reproducer runs as a
  build gate). `expandable_segments` is deliberately unset (silent
  NaNs on gfx1201 in the fork author's A/B).
- **Open part:** the performance price of this bundle has never been
  measured on this rig.
- **Planned attempt:** a paired dual test — one env flip, gated on
  the #6595 reproducer first (if hipBLASLt re-opens the fp32 wound
  it dies on correctness whatever the stopwatch says); a candidate
  refinement is chunking only where the bug lives (large fp32), if
  the reproducer can draw that line.

### 8. Attention runs on SDPA, never baselined

- **What:** Upstream's fast path is flash-attention, a CUDA package
  with no ROCm build in this stack, so the port inherited SDPA
  everywhere (`ATTN_BACKEND=sdpa`, `SPARSE_ATTN_BACKEND=sdpa`). The
  gap between SDPA-on-ROCm and a native kernel on gfx1201 has never
  been timed here; the tax is invisibility.
- **Planned attempt:** dual-test the Triton flash-attention backend
  (JIT-compiles for gfx1201 at runtime; the CK backend is Wave64-
  shaped for CDNA and fights RDNA's Wave32). AITER is not a
  candidate arm today: its operator set does not cover this
  pipeline's sparse-3D attention.

### 9. ROCm point-release drift

- **What:** The bench baseline is ROCm 7.2.1; the campaign proof
  runs used ROCm 7.2.2 / torch 2.11.0. AMD's GEMM tuning for
  gfx1201 is still improving per point release, so staying behind
  is a slow, silent tax — but a ROCm upgrade is the highest-blast-
  radius change at this layer (torch wheel index, extension
  rebuilds, and the torch ladder in item 2 all key off it).
- **Planned attempt:** a paired A/B (current vs then-current 7.x,
  full extension rebuild on the candidate arm), queued behind the
  cheaper trials; it trails the queue, never leads it.

### 10. Environment is performed, not declared

- **What:** The bench is built by an imperative script; the complete
  dependency closure is declared nowhere. (The campaign proved this
  the hard way: a from-scratch rebuild reproduced the same three
  missing transitive dependencies — `dash`, `urllib3`, `onnxruntime`
  — because the recipe never asked for them.) The ROCm installer
  .deb is also downloaded unhashed — the one unauthenticated
  executable in the pipe.
- **Planned attempt:** one hash-locked spec (pip-compiled closure
  with per-artifact hashes, plus the non-pip state: ROCm + .deb
  hash, Python, source-build commits, weight hashes); installs
  converge to it; the existing census becomes a spec verifier (a
  `trellis-env` doctor check) instead of a discovery tool.

### 11. Replacement queue for the CUDA-born components (R2–R5)

Under the port's AMD-first rule — any component that was CUDA-built
is replaced by an AMD-way implementation, gated by paired dual
tests — the queue after R1 (the fill, in the CuMesh bundle) is:

- **R2 — nvdiffrast contract replacement (planned).** The texture
  bake calls NVIDIA's `RasterizeCudaContext` API, currently served
  by a community HIP reimplementation. nvdiffrast's license limits
  derivative works to non-commercial use, so the deliverable is a
  **clean-room rasterizer behind our own API, in a new repo** — not
  a fork of nvdiffrast, and nothing of nvdiffrast is in this bundle.
  Gate: the textured bake end to end, with the r34 census riding as
  part of the gate.
- **R3 — o_voxel HIP-native (planned).** Replace the CUDA-source
  voxel extension (extraction + postprocess) with a HIP-native one.
  The r31–r34 findings transfer as the replacement's spec — the
  postprocess behaviors are already written down. Gate: extraction
  whole + postprocess gates + bounds census. This is also the venue
  where a structural remedy for item 4 would land.
- **R4 — nvdiffrec_render use-or-delete verdict (planned).** First
  step is call-site verification. If nothing calls it, the scrap is
  a deletion and the slot closes in one move; if something does, it
  becomes a replacement slot of its own, sized by what calls it.
- **R5 — pipeline Python by attrition (parked).** The substrate is
  replaced as the earlier slots land behind our own interfaces; a
  whole-pipeline rewrite is the limit case, sequenced last because
  its blast radius is total. No card until the earlier slots land.

### 12. Sparse-conv rival to flex_gemm — standing search

- **What:** `SPARSE_CONV_BACKEND=flex_gemm` was selected, not
  defaulted — it is upstream's own Triton-based backend and the
  most AMD-shaped choice in the stack. Its ROCm rivals are
  unbuilt, not beaten: **there is no second arm**, so no dual test
  exists to run.
- **Status:** parked as a standing search. If a ROCm-buildable rival
  (or a build-our-own sparse-GEMM scope) appears, it enters the
  replacement queue as its own slot.

### 13. Orchestrator migration (bash → Python) — deferred, trigger named

- **What:** `build-inside.sh` (in `changes/`) is a ~1,700-line bash
  program doing state management bash was never meant to do. The
  Python destination is agreed; the rewrite is deferred because
  every phase's behavior would have to be re-proven against the
  only orchestration that has ever produced a whole mesh on this
  stack, mid-campaign.
- **Status:** parked, with the trigger named: after the current
  campaign, or the next time a phase needs structural surgery,
  whichever comes first. Meanwhile new components are born
  Python-only (the census/verifier of item 10 already is), so the
  migration shrinks every month it waits.

## Parked

### 14. V-CEILING — quality-ceiling probe (filed intent)

- **What:** One ~70-hour-class burn at raised quality settings, run
  purely to learn where the pipeline's quality ceiling is — a probe,
  not production.
- **Status:** parked. Not built, not scheduled; the shipped v1 is
  the current priority, and a day-plus run gets a day-plus warning
  and a watched weekend slot if it is ever scheduled.
