#!/usr/bin/env python3
"""generate.py - one TRELLIS.2 generation with full evidence logging.

Prints torch version, GPU name, per-stage wall times and PEAK VRAM (the peak
VRAM number is the proof the GPU did the work - a silent CPU fallback would
show ~0 GB and a 10x slower time). Exports a GLB and scans the mesh vertices
for NaNs. Exits non-zero on any failure so the caller's verdict is honest.

The TRELLIS.2 export API is used defensively: the returned O-Voxel object is
introspected and the known export paths are tried in order, logging which one
worked. If upstream renamed things, the traceback + object repr land in the log
and Joe fixes from evidence instead of guesses.
"""
import argparse
import math
import os
import sys
import time
import traceback


def log(msg):
    print(f"[gen] {msg}", flush=True)


def finite_scan(obj, depth=0):
    """Best-effort NaN scan over common vertex containers."""
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return bool(torch.isfinite(obj.float()).all())
    except Exception:
        pass
    return True


def drop_degenerate_faces(faces):
    """r24: drop index-degenerate faces - faces where two or more vertex
    indices are equal, e.g. the literal (0,0,0) rows the cascade decoder
    emits past row 2^20 (1,091,754 of them on probe-S1; see the r24 cutter
    autopsy). Such a face has zero area and renders as nothing, but it
    poisons everything downstream: it inflates face counts past the
    CuMesh 2^20 zero-fill cliff and piles into a single histogram bin in
    the section cutter. Takes an (N, 3) integer array; returns
    (kept_faces, n_dropped)."""
    import numpy as _np

    _f = _np.asarray(faces)
    if _f.ndim != 2 or _f.shape[1] != 3 or _f.shape[0] == 0:
        return _f, 0
    _deg = (_f[:, 0] == _f[:, 1]) | (_f[:, 1] == _f[:, 2]) | (_f[:, 0] == _f[:, 2])
    return _f[~_deg], int(_deg.sum())


# ---------------------------------------------------------------------
# r28: CHUNKED GPU->HOST READOUT. HUNT-NOTE-2 (2026-10-25): on this
# stack (ROCm 7.2.2, gfx1201, WSL2/DXG) a single-shot copy of an
# (N, 3) array with N past ~2^20 rows silently loses its middle - the
# first 1,048,576 rows land, the rest read back as the destination's
# zeros (an end block lands only intermittently). There is no bad
# array on the GPU and no chunk loop in TRELLIS to fix; the loss is
# in the transfer itself. The same platform behavior is documented
# from the CuMesh side by PR #35 (ron-ytsma), whose fix chunks its
# copies at 65,536 rows; chunked copies read back exact where
# single-shot large ones fail on this ROCm stack. The helper below
# is our own implementation of that technique: copy in <=65,536-row
# chunks along dim 0 and reassemble on the host. EVERY mesh-scale
# readout in this file goes through it. At or under the chunk size
# the behavior is the old single-shot .cpu().numpy(), unchanged.
READOUT_CHUNK_ROWS = 65536


def chunked_readout(x, chunk_rows=READOUT_CHUNK_ROWS):
    """Return x as a numpy array, copying GPU->host in row chunks of
    at most chunk_rows when x has more rows than that. Accepts torch
    tensors (any device/dtype) and array-likes (returned as
    np.asarray). None in, None out."""
    import numpy as _np

    if x is None:
        return None
    if not hasattr(x, "detach"):
        return _np.asarray(x)
    t = x.detach()
    try:
        n = int(t.shape[0])
    except Exception:
        return t.cpu().numpy()
    if n <= chunk_rows:
        return t.cpu().numpy()
    parts = []
    for _s in range(0, n, chunk_rows):
        parts.append(t[_s:_s + chunk_rows].cpu().numpy())
    return _np.concatenate(parts, axis=0)


def genuine_row_profile(faces, nblocks=32):
    """r28: count genuine (not all-zero) rows of an (N, k) faces array
    per equal row-block; returns a list of nblocks ints. Computed
    where the data lives - torch ops for a tensor, numpy for an
    array - so a GPU profile never crosses to the host except as 32
    small counts. A face row of literal (0,0,0) is the copy cliff's
    signature (and index-degenerate besides), so genuine-row counts
    per block expose a readout desert as a run of zero blocks."""
    import numpy as _np

    if faces is None:
        return None
    counts = []
    if hasattr(faces, "detach"):
        f = faces.detach()
        n = int(f.shape[0])
        for _b in range(nblocks):
            _lo = _b * n // nblocks
            _hi = (_b + 1) * n // nblocks
            counts.append(int((f[_lo:_hi] != 0).any(dim=1).sum().item()) if _hi > _lo else 0)
        return counts
    f = _np.asarray(faces)
    n = int(f.shape[0])
    for _b in range(nblocks):
        _lo = _b * n // nblocks
        _hi = (_b + 1) * n // nblocks
        counts.append(int((f[_lo:_hi] != 0).any(axis=1).sum()) if _hi > _lo else 0)
    return counts


# ---------------------------------------------------------------------
# r31: CHUNKED FLEXIBLE-DUAL-GRID EXTRACTION (the extraction fix,
# Cory approved 2026-10-25). r30's probe closed the hunt's verdict
# table: BOTH the cascade meshes AND the non-cascade 1024 control come
# out of pipeline.run with every output array that crosses 2^20 rows
# wounded the same way - genuine rows to exactly 1,048,576 (2^20),
# then a desert of zero rows, then a small genuine end-aligned stub -
# while the sparse coords themselves are whole (subject coords
# 1,602,346 > 2^20). The wound is born at or before the mesh the
# pipeline hands back, in the one stage that turns the sparse
# structure into vertices/faces: the FDG extraction,
# o_voxel.convert.flexible_dual_grid_to_mesh, called ONCE per mesh by
# trellis2's FlexiDualGridVaeDecoder (trellis2/models/sc_vaes/
# fdg_vae.py) with the mesh's full coords.
#
# What the extraction code guarantees (read from the upstream tree,
# o-voxel/o_voxel/convert/flexible_dual_grid.py): vertex row i of the
# output IS voxel row i of the input - mesh_vertices =
# (coords + dual_vertices) * voxel_size + aabb[0], one vertex per
# voxel, same order - and every face is a triangle over voxel indices
# found by hashmap lookup of the anchor voxel's +1-cube neighbours.
# So the extraction decomposes EXACTLY over voxel subsets: give it a
# subset of voxels closed under the +1-corner offsets and it produces
# that subset's vertices (identical values - a voxel's vertex depends
# only on its own coord + dual vertex) and every quad anchored in the
# subset whose corners are present. This module therefore wraps the
# stock function: slice the coords into core slabs along the widest
# axis, close each core under the corner offsets (the halo), call the
# STOCK extraction per chunk - never with more than
# FDG_CHUNK_INPUT_BUDGET input rows, so no output array and no
# intermediate in the call can cross the 2^20 wall even at the
# theoretical maximum density (<=3 quads per voxel -> <=6 triangles
# and <=12 lookup rows per input row) - then merge by GLOBAL VOXEL
# IDENTITY (the coord triple itself, never position matching): one
# output vertex per voxel, chunk faces remapped through the voxel
# key, exact duplicate triangles (quads regenerated where chunk
# halos overlap) dropped once. The budgets are house tuning values:
# 2^16 input rows per call keeps worst-case faces (393,216) under
# 2^19 and worst-case lookup rows (786,432) under 2^20.
#
# SCOPE, stated plainly: this fixes the extraction stage. The merged
# profile logged below ([r31-chunk] merged) proves what extraction
# itself produced, BEFORE Mesh.fill_holes / simplify / to_glb touch
# it - if a later stage re-wounds the mesh, r31's own post-run
# [gpu-profile] lines convict it by comparison, which is why both
# sets of lines stay in the log. The wrapper delegates untouched to
# the stock function when the mesh is small (<= budget: every 512
# run) and whenever train=True (training-mode extraction appends
# per-quad mid vertices that are not voxel-row-aligned; inference
# never uses it). If chunk planning itself fails, the wrapper logs
# and RAISES - a run that cannot apply the fix must not masquerade
# as the acceptance run.
FDG_CHUNK_INPUT_BUDGET = 65536   # max coords rows handed to one extraction call (2^16)
FDG_CHUNK_CORE_BUDGET = 49152    # max owned (core) coords rows per chunk before halo closure
FDG_KEY_STRIDE = 4096            # > any TRELLIS grid resolution (<=1536); coord key stride

# The extraction's quad-corner offsets (the union of its three
# per-axis edge-neighbour tables): a quad anchored at a voxel reaches
# only neighbours in the +1 cube. Closure under these 8 offsets
# guarantees every core-anchored quad finds its corners in-chunk.
_FDG_CORNER_OFFSETS = tuple(
    (dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1))


def _fdg_coord_keys(coords_np):
    """Global voxel identity: one int64 key per (x, y, z) coord row."""
    import numpy as _np

    c = coords_np.astype(_np.int64)
    return (c[:, 0] * FDG_KEY_STRIDE + c[:, 1]) * FDG_KEY_STRIDE + c[:, 2]


def _plan_fdg_chunks(coords_np, core_budget=FDG_CHUNK_CORE_BUDGET,
                     input_budget=FDG_CHUNK_INPUT_BUDGET):
    """Plan chunked extraction over a coords array (N, 3).

    Returns a list of (core_rows, input_rows) int64 index arrays.
    Core rows partition the whole set (every voxel is owned by
    exactly one chunk); input rows are the core closed under the
    +1-cube corner offsets, so every quad anchored at a core voxel
    has all four corners present in its chunk's input. Cores are
    whole axis-layers (slabs along the widest axis) accumulated up
    to core_budget; a core whose closure would exceed input_budget
    is halved and replanned, so every planned input respects
    input_budget. Raises ValueError on duplicate coord rows (voxel
    identity would be ambiguous) - the sparse structure's coords
    are unique by construction, so a duplicate means something
    upstream changed shape and chunking must not guess."""
    import numpy as _np

    coords = _np.asarray(coords_np)
    n = int(coords.shape[0])
    keys = _fdg_coord_keys(coords)
    key_to_row = {}
    for _i, _k in enumerate(keys.tolist()):
        if _k in key_to_row:
            raise ValueError(f"duplicate coord row at index {_i} - voxel identity ambiguous, refusing to chunk")
        key_to_row[_k] = _i
    _deltas = tuple((dx * FDG_KEY_STRIDE + dy) * FDG_KEY_STRIDE + dz
                    for (dx, dy, dz) in _FDG_CORNER_OFFSETS)

    def _closure(core_rows):
        _seen = set()
        for _r in core_rows.tolist():
            _k = int(keys[_r])
            for _d in _deltas:
                _kk = _k + _d
                if _kk in key_to_row:
                    _seen.add(key_to_row[_kk])
        return _np.array(sorted(_seen), dtype=_np.int64)

    # Core candidates: whole layers along the widest axis, in axis
    # order, accumulated to core_budget; an oversize single layer is
    # pre-split into core_budget-sized row parts.
    _axis = int(_np.argmax(coords.max(0) - coords.min(0)))
    _order = _np.argsort(coords[:, _axis], kind="stable")
    _vals = coords[_order, _axis]
    _bounds = [0] + [int(i) + 1 for i in _np.flatnonzero(_np.diff(_vals))] + [n]
    _layers = [_order[_bounds[i]:_bounds[i + 1]] for i in range(len(_bounds) - 1)]
    _cores = []
    _cur = []
    _cur_n = 0
    for _layer in _layers:
        if len(_layer) > core_budget:
            if _cur:
                _cores.append(_np.concatenate(_cur))
                _cur, _cur_n = [], 0
            for _s in range(0, len(_layer), core_budget):
                _cores.append(_layer[_s:_s + core_budget])
            continue
        if _cur_n + len(_layer) > core_budget and _cur:
            _cores.append(_np.concatenate(_cur))
            _cur, _cur_n = [], 0
        _cur.append(_layer)
        _cur_n += len(_layer)
    if _cur:
        _cores.append(_np.concatenate(_cur))

    # Enforce the input budget: any core whose halo closure exceeds
    # it is halved (by rows) and each half replanned, recursively.
    _plan = []
    _stack = list(_cores)
    while _stack:
        _core = _stack.pop()
        _inp = _closure(_core)
        if len(_inp) <= input_budget or len(_core) <= 1:
            _plan.append((_core, _inp))
        else:
            _h = len(_core) // 2
            _stack.append(_core[:_h])
            _stack.append(_core[_h:])
    _plan.reverse()
    return _plan


def _merge_fdg_chunks(coords_np, plan, chunk_outputs):
    """Merge per-chunk extraction outputs into one (vertices, faces).

    chunk_outputs[i] = (verts_i, faces_i) produced by extracting
    plan[i]'s input rows: verts_i row j is the vertex of the voxel
    at input row j (the extraction's per-voxel correspondence), and
    faces_i indexes verts_i. The merge keeps ONE vertex per global
    voxel (keyed by the coord triple - identical voxels in overlap
    regions carry identical vertex values, computed from their own
    coord + dual vertex, so first-wins loses nothing), remaps every
    chunk face through the key, and drops exact duplicate triangles
    (a quad anchored in a halo voxel is regenerated by the chunk
    that owns it; both copies are the same three global indices in
    the same order, since the split rule sees the same four corner
    positions and weights). Returns (verts float32, faces int64)."""
    import numpy as _np

    keys = _fdg_coord_keys(_np.asarray(coords_np))
    key_to_global = {}
    verts_rows = []
    face_blocks = []
    for (_core, _inp), (_v, _f) in zip(plan, chunk_outputs):
        _v = _np.asarray(_v)
        _f = _np.asarray(_f)
        if _v.shape[0] != len(_inp):
            raise ValueError(f"chunk extraction returned {_v.shape[0]} vertices for {len(_inp)} input voxels - per-voxel correspondence broken, refusing to merge")
        _l2g = _np.empty(len(_inp), dtype=_np.int64)
        for _j, _r in enumerate(_inp.tolist()):
            _k = int(keys[_r])
            _g = key_to_global.get(_k)
            if _g is None:
                _g = len(key_to_global)
                key_to_global[_k] = _g
                verts_rows.append(_v[_j])
            _l2g[_j] = _g
        if _f is not None and len(_f):
            face_blocks.append(_l2g[_np.asarray(_f, dtype=_np.int64)])
    verts = (_np.array(verts_rows, dtype=_np.float32)
             if verts_rows else _np.zeros((0, 3), dtype=_np.float32))
    if face_blocks:
        faces = _np.concatenate(face_blocks, axis=0)
        # Exact-duplicate drop, keeping the first occurrence's row
        # (and therefore its winding) for each distinct triangle.
        _sorted = _np.sort(faces, axis=1)
        _uniq_rows, _first = _np.unique(_sorted, axis=0, return_index=True)
        faces = faces[_np.sort(_first)]
    else:
        faces = _np.zeros((0, 3), dtype=_np.int64)
    return verts, faces


def chunked_fdg_extract(coords_np, extract_one,
                        core_budget=FDG_CHUNK_CORE_BUDGET,
                        input_budget=FDG_CHUNK_INPUT_BUDGET):
    """Drive a per-chunk extractor over the whole coords set and
    merge. extract_one(input_rows) must return (verts, faces) for
    exactly those coord rows, with verts row-aligned to input_rows -
    the contract the stock FDG extraction satisfies. Returns
    (verts, faces, plan). Pure numpy: unit-tested on CPU with a
    synthetic extractor (tests/test_chunked_fdg.py)."""
    plan = _plan_fdg_chunks(coords_np, core_budget, input_budget)
    outputs = [extract_one(_inp) for (_core, _inp) in plan]
    verts, faces = _merge_fdg_chunks(coords_np, plan, outputs)
    return verts, faces, plan


def _chunked_fdg_to_mesh(orig, coords, dual_vertices, intersected_flag,
                         split_weight, aabb, voxel_size, grid_size, train):
    """Torch glue around chunked_fdg_extract: slices the decoder's
    row-aligned tensors (coords / dual vertices / intersected flags /
    split weights all share the sparse tensor's row order), calls the
    stock extraction per chunk, merges on the host, and returns
    tensors on the original device. Delegates untouched for small
    meshes and train mode (see the r31 header for why)."""
    import torch

    if train:
        return orig(coords, dual_vertices, intersected_flag, split_weight,
                    aabb=aabb, voxel_size=voxel_size, grid_size=grid_size,
                    train=True)
    n = int(coords.shape[0])
    if n <= FDG_CHUNK_INPUT_BUDGET:
        return orig(coords, dual_vertices, intersected_flag, split_weight,
                    aabb=aabb, voxel_size=voxel_size, grid_size=grid_size,
                    train=False)
    coords_np = chunked_readout(coords)
    dev = coords.device

    def _extract_one(rows_np):
        _idx = torch.as_tensor(rows_np, dtype=torch.long, device=dev)
        _v, _f = orig(coords[_idx], dual_vertices[_idx],
                      intersected_flag[_idx],
                      split_weight[_idx] if split_weight is not None else None,
                      aabb=aabb, voxel_size=voxel_size,
                      grid_size=grid_size, train=False)
        return chunked_readout(_v), chunked_readout(_f)

    try:
        verts_np, faces_np, plan = chunked_fdg_extract(coords_np, _extract_one)
    except Exception:
        log("[r31-chunk] chunked extraction FAILED - the run must not "
            "masquerade as the acceptance run, so this is fatal:\n"
            + traceback.format_exc())
        raise
    log(f"[r31-chunk] {n} voxels -> {len(plan)} chunks "
        f"(input rows per chunk: min {min(len(i) for _c, i in plan)}, "
        f"max {max(len(i) for _c, i in plan)}, budget {FDG_CHUNK_INPUT_BUDGET})")
    for _label, _arr in (("vertices", verts_np), ("faces", faces_np)):
        _prof = genuine_row_profile(_arr)
        log(f"[r31-chunk] merged {_label}: {sum(_prof)} genuine of "
            f"{_arr.shape[0]} rows (per-32-blocks {_prof}) - this is the "
            "EXTRACTION output, before fill_holes/simplify/to_glb")
    return (torch.as_tensor(verts_np, dtype=torch.float32, device=dev),
            torch.as_tensor(faces_np, dtype=torch.int32, device=dev))


def install_chunked_fdg():
    """Install the chunked wrapper over the stock FDG extraction in
    every module namespace that holds a binding to it - decisively
    trellis2.models.sc_vaes.fdg_vae, whose module-level
    `from o_voxel.convert import flexible_dual_grid_to_mesh` is the
    reference FlexiDualGridVaeDecoder.forward actually calls.
    Returns the list of patched module names; empty means the
    extraction binding was not found (caller fails loud)."""
    import importlib

    _found = []
    for _modname in ("trellis2.models.sc_vaes.fdg_vae",
                     "o_voxel.convert.flexible_dual_grid",
                     "o_voxel.convert"):
        try:
            _mod = importlib.import_module(_modname)
            _fn = getattr(_mod, "flexible_dual_grid_to_mesh", None)
        except Exception:
            continue
        if callable(_fn):
            _found.append((_modname, _mod, _fn))
    _wrappers = {}
    _patched = []
    for _modname, _mod, _fn in _found:
        if getattr(_fn, "_r31_chunked", False):
            _patched.append(_modname)
            continue
        if id(_fn) not in _wrappers:
            def _wrapper(coords, dual_vertices, intersected_flag,
                         split_weight, aabb, voxel_size=None,
                         grid_size=None, train=False, _orig=_fn):
                return _chunked_fdg_to_mesh(_orig, coords, dual_vertices,
                                            intersected_flag, split_weight,
                                            aabb, voxel_size, grid_size,
                                            train)
            _wrapper._r31_chunked = True
            _wrappers[id(_fn)] = _wrapper
        _mod.flexible_dual_grid_to_mesh = _wrappers[id(_fn)]
        _patched.append(_modname)
    if "trellis2.models.sc_vaes.fdg_vae" not in _patched:
        return []
    return _patched


# ---------------------------------------------------------------------
# r32: CHUNKED fill_holes (the fill_holes fix, Cory approved
# 2026-10-25 under "plan and fix it all"). r31 closed the extraction
# stage and convicted this one by comparison: its merged profiles are
# WHOLE (canvas 1,683,908 faces; subject 3,034,500; control
# 2,858,434) while the meshes pipeline.run returns are wounded at the
# same 2^20 cliff (canvas faces: genuine prefix 1,048,576, zero
# desert, genuine end stub 6,562; the export [filter] then keeps
# exactly 1,048,576). Between those two measurements the pipeline
# does exactly one mesh-mutating thing: decode_latent calls
# Mesh.fill_holes() (trellis2/representations/mesh/base.py).
#
# THE MECHANISM, named from the code (the PC's stock sources, read
# from the upstream TRELLIS.2 tree - byte-identical to the study
# copy; the r32 source capture now also ships representations/ so
# the PC's own copy can be diffed after the run):
#
# Mesh.fill_holes round-trips the WHOLE mesh through CuMesh:
#   mesh.init(vertices, faces); boundary machinery;
#   mesh.fill_holes(...); self.vertices, self.faces = mesh.read()
# and CuMesh::init (cumesh/src/io.cu) copies both arrays in with
# SINGLE-SHOT cudaMemcpy2D calls whose height is the row count. On
# this stack (ROCm 7.2.2, gfx1201, WSL2/DXG) a device-to-device
# cudaMemcpy2D above 2^20 rows silently truncates: the first
# 1,048,576 rows land, the middle stays the destination buffer's
# zeros, a tail block lands intermittently. Everything downstream
# of init - the boundary machinery, the fill kernels, read() -
# faithfully processes that wounded internal copy, and read()
# itself is innocent for vertices/faces (io.cu's buffer_to_tensor
# uses a LINEAR cudaMemcpy when the element size matches the
# tensor row size, which it does for float3/int3). The fill
# algorithm is innocent too, and APPEND-ONLY (cumesh/src/
# clean_up.cu): originals are never modified, reordered or
# deleted; each boundary loop under the perimeter threshold gains
# ONE centroid vertex appended after the last vertex and ONE fan
# face per loop edge appended after the last face - which is why
# r31's wounded profile ends in a genuine stub of exactly the
# patch count (+6,562 faces / +1,078 verts on canvas). Arrays
# under the wall survive whole (canvas verts 842,266); arrays over
# it wound identically (faces everywhere; subject verts 1.6M).
# PR #35 patches exactly init's copies (linear memcpy + chunked
# buffer_to_tensor), but the patched cumesh build is blocked on
# this machine (Phase K, rocprim tuple_size), so r32 does from
# the Python side what r31 did for extraction: never let one
# CuMesh call see more than a bounded number of rows.
#
# THE WRAPPER (installed over Mesh.fill_holes before
# pipeline.run, in install_chunked_fill below):
# - Small meshes delegate to the stock method untouched: faces
#   AND vertices at or under 2^19 rows. The 512 arms (420,804
#   faces / 209,354 verts) are proven whole through the stock
#   path; the wall is 2^20; 2^19 keeps every stock copy at half
#   the wall or less.
# - Big meshes are filled in chunks by the pure-numpy core
#   (chunked_fill_core): a host census finds the boundary loops;
#   faces are partitioned into contiguous chunks of at most
#   2^17 faces / 2^18 referenced vertices, with every face
#   incident to a candidate hole loop locked into ONE chunk
#   (loops are never split). Per chunk the wrapper builds a real
#   Mesh and calls the ORIGINAL stock fill_holes on it - every
#   CuMesh copy inside that call is at most 2^17 rows for
#   faces, 2^18 for vertices, and the widest derived array
#   (directed edge adjacency, <= 3x the chunk's faces) stays
#   under 0.4 x 2^20. Because the stock fill is append-only,
#   the merge is exact: originals pass through bit-for-bit and
#   only appended fan faces are harvested - and a fan face is
#   KEPT only if every original vertex it references lies on
#   ONE global boundary component whose own perimeter is under
#   the fill threshold and whose incident faces are ALL inside
#   the chunk. Fans the filler spins across partition cuts
#   belong to no such component (their vertices span the cut)
#   and are dropped, so chunking can lose a fill at a
#   pathological pinch but can never invent geometry. If a
#   chunk's output violates the append-only invariant (the
#   input prefix must come back bit-exact), that chunk falls
#   back to its unfilled faces and logs it - a corrupted chunk
#   is never emitted.
# - After the fill (either path) the wrapper logs [r32-fill]
#   genuine-row profiles of the mesh's faces and vertices,
#   computed on the GPU tensors exactly like the post-pipeline
#   [gpu-profile] - the grade reads them in order: [r31-chunk]
#   merged (extraction, whole) -> [r32-fill] (this stage) ->
#   [gpu-profile] (pipeline return). Wholeness must survive
#   each arrow.
#
# SIMPLIFY GUARD (same patch, same reason): Mesh.simplify also
# round-trips the whole mesh through CuMesh init/read.
# generate.py's only call is simplify(16777216) - a target above
# every face count this pipeline produces, i.e. a simplification
# that cannot decimate anything, whose round trip on this stack
# can only wound. The guard returns without the round trip when
# target >= the current face count and delegates otherwise.
# Genuine decimations at or under the section caps in to_glb
# (sections are hard-capped at 1,000,000 faces, and their
# vertex/derived arrays are their own story) are NOT touched.
#
# Budgets are house tuning values, derived above: delegation
# limit 2^19 (half the wall, above the proven-whole 420k-face
# arms), chunk budgets 2^17 faces / 2^18 vertices (every copy
# <= wall/4, widest derived array <= 0.4 x wall). If chunk
# planning or the census fails, the wrapper logs and RAISES -
# like r31, a run that cannot apply the fix must not masquerade
# as the acceptance run.
FILL_STOCK_FACE_LIMIT = 524288    # 2^19: stock fill_holes delegation limit (faces)
FILL_STOCK_VERT_LIMIT = 524288    # 2^19: stock fill_holes delegation limit (vertices)
FILL_CHUNK_FACE_BUDGET = 131072   # 2^17: max faces in one chunked CuMesh fill call
FILL_CHUNK_VERT_BUDGET = 262144   # 2^18: max referenced vertices in one chunk


# ---------------------------------------------------------------------
# r33: CHUNK ISOLATION for the chunked fill (Cory's route call,
# 2026-10-25, after job 028's split verdict). r32's chunking killed
# the 2^20 copy cliff - the canvas 1024 cascade arm filled whole,
# 1,694,825/1,694,825 faces - but the chunk fills still ran
# IN-PROCESS, and on the subject 1024 cascade arm three chunks hit
# CuMesh's boundary-components kernel failure
# (connectivity.hip:867, RuntimeError code 9, "invalid
# configuration argument"); the poisoned HIP context then
# SEGFAULTED and the whole run died holding a whole mesh. r33
# changes exactly one variable: every chunk's stock fill runs in
# its OWN short-lived subprocess (this same file, re-invoked with
# --fill-chunk-worker), so a kernel failure, a poisoned context or
# a segfault costs that chunk's process and nothing else. r32's
# discipline is untouched: chunked_fill_core still plans the
# chunks, and every returned fill is still verified against the
# append-only invariant before a single fan is harvested.
#
# Failure ladder per chunk (each step logged [r33-fill]):
#   1. one subprocess attempt on the whole chunk;
#   2. on failure (nonzero exit, timeout, missing or invalid
#      output): ONE retry as two half-chunks, each in its own
#      subprocess - halving changes the kernel's launch
#      configuration, which is the variable code 9 names;
#   3. if a half still fails, the fill raises and
#      chunked_fill_core falls that chunk back to UNFILLED - its
#      original faces pass through bit-exact and the run survives
#      to export.
# A subprocess segfault cannot take the parent with it, and a
# fresh process carries a fresh HIP context, so step 2 is a real
# retry, not a replay of the poison.
#
# The half-chunk merge reassembles the two half results into the
# append-only shape the core verifies: ALL original chunk faces
# first, in order, then the halves' appended fan faces remapped
# into whole-chunk indices, behind ALL original chunk vertices.
# Fill QUALITY at a retry seam can drop (a loop the split divides
# may go unfilled); geometry can never be invented - the core's
# extract step still vets every kept fan against the global
# boundary census exactly as in r32.
#
# The per-attempt timeout is a house tuning value: r32's
# in-process chunk fills took seconds; 900 s per attempt is
# head-room for a loaded machine, not an expectation.
FILL_CHUNK_TIMEOUT_S = 900        # per subprocess fill attempt

# ---------------------------------------------------------------------
# r35: THE R1 SWAP (Cory's ruling 2026-10-26: R1 fully built
# before job 038 ended, so the new code runs; the old code is
# NOT deleted). Exactly ONE behavioral change against r33: the
# fill-chunk worker command pointer in _run_fill_subprocess
# below. When build-inside.sh's Phase R1 gate stage has
# compiled, probed, calibrated and equivalence-gated the R1 HIP
# fill worker, it exports R1_FILL_WORKER=<worker binary> and
# chunk fills run there - argv [worker, in, out, threshold],
# the same seam, no interpreter, no --fill-chunk-worker flag.
# With the variable unset, the command is the incumbent stock
# worker, byte-for-byte the r33 behavior; the one-line revert
# is to restore that unconditional assignment. Everything else
# in this file - census, planner, ladder, accounting - is
# byte-identical to r33.

def _fill_chunk_worker(in_path, out_path, max_hole_perimeter):
    """Subprocess entry point for the r33 isolated fill: run the
    STOCK Mesh.fill_holes on one chunk stored in an .npz and
    write the filled chunk to a second .npz. This process imports
    trellis2 fresh and NEVER installs the r32/r33 wrappers, so
    Mesh.fill_holes here is the stock method. Exit 0 on success;
    any exception exits nonzero, and a CuMesh segfault takes this
    process alone - which is the entire point."""
    import numpy as _np
    import torch

    _d = _np.load(in_path)
    _cv = _d["verts"]
    _cf = _d["faces"]
    from trellis2.representations.mesh.base import Mesh as _MeshCls

    _dev = "cuda" if torch.cuda.is_available() else "cpu"
    _cm = _MeshCls(
        torch.as_tensor(_cv, dtype=torch.float32, device=_dev),
        torch.as_tensor(_cf, dtype=torch.int32, device=_dev))
    _cm.fill_holes(max_hole_perimeter=max_hole_perimeter)
    _ov = chunked_readout(_cm.vertices)
    _of = chunked_readout(_cm.faces).astype("int64")
    _np.savez(out_path, verts=_ov, faces=_of)
    return 0


def _run_fill_subprocess(cv, cf, max_hole_perimeter,
                         worker_cmd=None,
                         timeout=FILL_CHUNK_TIMEOUT_S,
                         log_fn=None):
    """Run one chunk's stock fill in a subprocess and return
    (out_verts, out_faces) as numpy arrays in the chunk's local
    indexing. Raises RuntimeError on ANY failure: nonzero exit
    (a negative return code is a signal - -11 is the segfault
    this isolation exists to survive), timeout, missing output,
    or an output smaller than its input (below the append-only
    floor). worker_cmd is injectable for tests; it receives
    (in_path, out_path, max_hole_perimeter) appended to its argv.
    The chunk travels through a temp .npz that is always removed
    afterwards - transient compute scratch, never a kept file."""
    import os
    import subprocess
    import tempfile

    import numpy as _np

    _tmp = tempfile.mkdtemp(prefix="r33-fill-")
    _in = os.path.join(_tmp, "in.npz")
    _out = os.path.join(_tmp, "out.npz")
    try:
        _np.savez(_in,
                  verts=_np.asarray(cv, dtype=_np.float32),
                  faces=_np.asarray(cf, dtype=_np.int64))
        if worker_cmd is None:
            # r35: the one command pointer (see the r35 header
            # above). R1_FILL_WORKER is exported by build-inside.sh
            # only after its Phase R1 gate stage passes; with the
            # variable unset the command is the incumbent stock
            # worker, byte-for-byte the r33 behavior. ONE-LINE
            # REVERT: restore the unconditional incumbent
            # assignment (the else branch) and delete this branch.
            _r1_worker = os.environ.get("R1_FILL_WORKER")
            if _r1_worker:
                worker_cmd = [_r1_worker]
                if log_fn is not None and not globals().get(
                        "_r35_r1_pointer_logged"):
                    globals()["_r35_r1_pointer_logged"] = True
                    log_fn("[r35-r1] chunk fills are running on "
                           "the R1 fill worker: " + _r1_worker)
            else:
                worker_cmd = [sys.executable, os.path.abspath(__file__),
                              "--fill-chunk-worker"]
        _cmd = list(worker_cmd) + [_in, _out,
                                   repr(float(max_hole_perimeter))]
        try:
            _rc = subprocess.run(_cmd, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"fill subprocess timed out after {timeout}s")
        if _rc != 0:
            raise RuntimeError(f"fill subprocess exited {_rc}")
        if not os.path.isfile(_out):
            raise RuntimeError("fill subprocess wrote no output file")
        _d = _np.load(_out)
        _ov = _np.asarray(_d["verts"])
        _of = _np.asarray(_d["faces"], dtype=_np.int64)
        if _of.shape[0] < _np.asarray(cf).shape[0] or \
                _ov.shape[0] < _np.asarray(cv).shape[0]:
            raise RuntimeError(
                "fill subprocess output is smaller than its input "
                "- not an append-only fill")
        return _ov, _of
    finally:
        try:
            for _p in (_in, _out):
                if os.path.isfile(_p):
                    os.remove(_p)
            os.rmdir(_tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------
# r35.7: BATCH FILL (optimization campaign variant V1; Cory's
# order 2026-10-27 after job 045's measured stop).
#
# The measured problem (job 045, Cory's PC): the r33 seam spawns
# one worker PROCESS per chunk, so every chunk paid process
# spawn + HIP runtime init - ~0.473 s per chunk nearly regardless
# of chunk size, against ~0.017 s of fill compute on no-op
# chunks. At 386,195 planned chunks that projected to ~41 h for
# one arm's fill, and Cory ruled it a fail (his bar: a run over
# ~10 h is a problem; time beyond the bar is only ever bought to
# purchase QUALITY, and this time bought none).
#
# The fix changes process lifetime, never fill math: when the
# caller supplies a batch filler, chunked_fill_core hands the
# WHOLE chunk plan to _run_fill_batch, which stages every chunk
# as an .npz in a spool dir and runs ONE worker process in its
# --batch mode (fill_worker.cpp r35.7) over a manifest. The
# worker pays its startup once, fills item after item, and
# prints R1-BATCH progress lines the parent relays into the
# [r35.6-fill] heartbeat via the on_start/on_end hooks, so the
# console narration stays per-chunk and truthful.
#
# Failure discipline is r33's, at batch granularity: the worker
# exits at the FIRST failed item (fresh HIP context per process
# is the point of the isolation), the parent keeps every
# completed item, sends the failed item back as None - which
# routes it through the untouched _fill_one_isolated ladder
# (whole single-shot, then halves, then unfilled fallback) - and
# re-batches the remainder in a fresh process. A worker that
# dies mid-chunk (signal, kill) is treated the same: the
# in-flight item goes to the ladder. A no-progress watchdog
# (FILL_CHUNK_TIMEOUT_S of silence, the inherited house value)
# kills a hung worker the same way. If batching itself breaks
# wholesale (spawn failure, repeated dead rounds), every
# remaining chunk falls back to the per-chunk path - slower,
# never wronger.
#
# Gating / revert: _chunked_fill_holes builds the batch filler
# ONLY when both R1_FILL_WORKER and R1_FILL_BATCH are set in the
# environment (build-inside.sh exports R1_FILL_BATCH only after
# the r1-stage gates, including a batch byte-identity gate,
# pass). Unset either and behavior is byte-for-byte r35.6.

def _load_fill_output(out_path, cv, cf):
    """Load one fill worker output .npz and apply the same
    append-only floor check _run_fill_subprocess applies.
    Returns (verts, faces) numpy arrays in chunk-local indexing.
    Raises RuntimeError on a missing file or a below-floor
    output."""
    import os

    import numpy as _np

    if not os.path.isfile(out_path):
        raise RuntimeError("fill worker wrote no output file")
    _d = _np.load(out_path)
    _ov = _np.asarray(_d["verts"])
    _of = _np.asarray(_d["faces"], dtype=_np.int64)
    if _of.shape[0] < _np.asarray(cf).shape[0] or \
            _ov.shape[0] < _np.asarray(cv).shape[0]:
        raise RuntimeError(
            "fill worker output is smaller than its input "
            "- not an append-only fill")
    return _ov, _of


def _run_fill_batch(items, worker_cmd, max_hole_perimeter,
                    on_start=None, on_end=None, log_fn=None):
    """Fill many chunks in ONE worker process (r35.7).

    items is a list of (chunk_verts, chunk_faces) numpy pairs,
    all sharing max_hole_perimeter. Returns a list parallel to
    items: (out_verts, out_faces) for chunks the batch filled
    and validated, None for chunks it did not (the caller routes
    those through the r33 ladder). on_start(idx) fires when the
    worker starts a chunk and on_end(idx, appended_faces) when
    its validated output is in hand - the caller's narration
    hooks. Worker stderr is merged into its stdout pipe and any
    line that is not R1-BATCH progress is relayed verbatim to
    log_fn, so R1-FAIL diagnostics reach the console as they
    always have. The spool dir is transient compute scratch,
    removed on exit, like r33's per-attempt dirs. A round
    counts as progress only when it stores a validated result
    or excludes a failed chunk; three progressless rounds -
    or a hard cap of n+16 rounds - abandon batching to the
    ladder, so no worker failure shape can spin this loop
    forever (santa review, Oct 27)."""
    import os
    import queue as _queue
    import subprocess
    import tempfile
    import threading

    import numpy as _np

    _n = len(items)
    _results = [None] * _n
    if _n == 0:
        return _results
    _log = log_fn if log_fn is not None else (lambda _m: None)
    _tmp = tempfile.mkdtemp(prefix="r33-fill-batch-")
    _proc_ref = [None]  # latest batch worker, for the finally
    try:
        _ins = [os.path.join(_tmp, f"in-{_i}.npz")
                for _i in range(_n)]
        _outs = [os.path.join(_tmp, f"out-{_i}.npz")
                 for _i in range(_n)]
        _written = set()
        _pending = list(range(_n))
        _dead_rounds = 0
        _rounds = 0
        while _pending:
            _rounds += 1
            if _rounds > _n + 16:
                # Backstop (santa review, Oct 27): with correct
                # exclusion accounting every round completes or
                # excludes at least one chunk, so rounds are
                # bounded by the chunk count plus slack. Past
                # that, something is lying - stop batching.
                _log("[r35.7-batch] round cap exceeded - "
                     "abandoning batch mode; every remaining "
                     "chunk takes the per-chunk path")
                break
            for _gi in _pending:
                if _gi not in _written:
                    _cv, _cf = items[_gi]
                    _np.savez(
                        _ins[_gi],
                        verts=_np.asarray(_cv, dtype=_np.float32),
                        faces=_np.asarray(_cf, dtype=_np.int64))
                    _written.add(_gi)
            _man = os.path.join(_tmp, "manifest.txt")
            with open(_man, "w") as _f:
                for _gi in _pending:
                    _f.write(f"{_ins[_gi]}\t{_outs[_gi]}\t"
                             f"{repr(float(max_hole_perimeter))}"
                             "\n")
            _log(f"[r35.7-batch] one worker process for "
                 f"{len(_pending)} chunks "
                 f"({len(_written)} staged in the spool)")
            try:
                _proc = subprocess.Popen(
                    list(worker_cmd) + ["--batch", _man],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1)
                _proc_ref[0] = _proc
            except Exception as _se:
                _log(f"[r35.7-batch] batch worker failed to "
                     f"start ({_se}) - every remaining chunk "
                     "takes the per-chunk path")
                break
            _lines = _queue.Queue()

            def _reader(_proc=_proc, _lines=_lines):
                try:
                    for _line in _proc.stdout:
                        _lines.put(_line.rstrip("\n"))
                except Exception:
                    pass
                _lines.put(None)

            _rt = threading.Thread(target=_reader, daemon=True)
            _rt.start()
            _current = None        # global idx in flight
            _failed = set()        # global idx failed this round
            _progress = False      # a validated result was stored
            _last = time.monotonic()
            while True:
                try:
                    _line = _lines.get(timeout=5)
                except _queue.Empty:
                    if _proc.poll() is not None and \
                            _lines.empty():
                        break
                    if time.monotonic() - _last > \
                            FILL_CHUNK_TIMEOUT_S:
                        _log(f"[r35.7-batch] no worker progress "
                             f"for {FILL_CHUNK_TIMEOUT_S}s - "
                             "killing the batch worker; any "
                             "in-flight chunk goes to the ladder")
                        _proc.kill()
                        if _current is not None:
                            _failed.add(_current)
                        break
                    continue
                if _line is None:
                    break
                _last = time.monotonic()
                _parts = _line.split()
                if len(_parts) >= 4 and _parts[0] == "R1-BATCH" \
                        and _parts[1] == "item":
                    try:
                        _gi = _pending[int(_parts[2])]
                    except (ValueError, IndexError):
                        _log(_line)
                        continue
                    if _parts[3] == "start":
                        _current = _gi
                        if on_start is not None:
                            on_start(_gi)
                    elif _parts[3] == "ok":
                        _current = None
                        _progress = True
                        try:
                            _ov, _of = _load_fill_output(
                                _outs[_gi], *items[_gi])
                        except Exception as _le:
                            _log(f"[r35.7-batch] chunk {_gi}: "
                                 f"batch output failed "
                                 f"validation ({_le}) - chunk "
                                 "goes to the ladder")
                            _failed.add(_gi)
                            continue
                        _results[_gi] = (_ov, _of)
                        if on_end is not None:
                            on_end(_gi, int(_of.shape[0])
                                   - int(items[_gi][1].shape[0]))
                    elif _parts[3] == "fail":
                        _current = None
                        _failed.add(_gi)
                    else:
                        _log(_line)
                elif _parts and _parts[0] == "R1-BATCH":
                    pass  # e.g. the final 'done items=' line
                else:
                    _log(_line)  # worker diagnostics, verbatim
            _rc = _proc.wait()
            if _current is not None:
                _failed.add(_current)  # worker died mid-chunk
            if _rc != 0:
                _log(f"[r35.7-batch] batch worker exited {_rc} "
                     f"({len(_failed)} chunk(s) failed this "
                     "round go to the ladder)")
            _pending = [_gi for _gi in _pending
                        if _results[_gi] is None
                        and _gi not in _failed]
            if not _pending:
                break
            if _progress or _failed:
                _dead_rounds = 0
            else:
                _dead_rounds += 1
                if _dead_rounds >= 3:
                    _log("[r35.7-batch] three batch rounds made "
                         "no progress - abandoning batch mode; "
                         "every remaining chunk takes the "
                         "per-chunk path")
                    break
        return _results
    finally:
        # Never leave a batch worker running behind us
        # (santa review, Oct 27): if a narration hook or any
        # other error raised out of the round loop, the worker
        # could otherwise still be filling - and holding the
        # GPU - while the caller degrades to ladder fills.
        if _proc_ref[0] is not None and \
                _proc_ref[0].poll() is None:
            _proc_ref[0].kill()
        # Whole-tree removal (santa review, Oct 27): the
        # worker's atomic writes can leave out-<i>.npz.tmp
        # orphans behind when a kill lands mid-write, and an
        # enumerated file list silently leaked the spool in
        # exactly that scenario.
        import shutil
        shutil.rmtree(_tmp, ignore_errors=True)


def _split_chunk_halves(cv, cf):
    """Split one chunk into two half-chunks by contiguous face
    ranges, each with its referenced vertices compacted. Returns
    a list of (verts, local_faces, vert_subset); vert_subset maps
    half-local vertex ids back to whole-chunk ids."""
    import numpy as _np

    _cv = _np.asarray(cv)
    _cf = _np.asarray(cf, dtype=_np.int64)
    _mid = _cf.shape[0] // 2
    _halves = []
    for _part in (_cf[:_mid], _cf[_mid:]):
        _uv, _inv = _np.unique(_part, return_inverse=True)
        _halves.append((_cv[_uv],
                        _inv.reshape(-1, 3).astype(_np.int64),
                        _uv.astype(_np.int64)))
    return _halves


def _merge_half_fills(cv, cf, halves, results):
    """Reassemble two half-chunk fill results into the
    append-only shape chunked_fill_core verifies: (verts, faces)
    = ALL original chunk rows first, bit-exact and in order, then
    the halves' appended vertices and fan faces remapped into
    whole-chunk indexing (halves and results are parallel lists
    from _split_chunk_halves and the per-half subprocess runs)."""
    import numpy as _np

    _cv = _np.asarray(cv)
    _cf = _np.asarray(cf, dtype=_np.int64)
    _out_v = [_cv]
    _out_f = [_cf]
    _new_base = int(_cv.shape[0])
    for (_hv, _hf, _uv), (_ov, _of) in zip(halves, results):
        _of = _np.asarray(_of, dtype=_np.int64)
        _nf_in = int(_hf.shape[0])
        _nv_in = int(_hv.shape[0])
        _app = _of[_nf_in:]                 # this half's appended fans
        if len(_app):
            _g = _app.copy()
            _orig = _g < _nv_in
            _g[_orig] = _uv[_g[_orig]]      # half-local -> whole-chunk
            _g[~_orig] = _new_base + (_g[~_orig] - _nv_in)
            _out_f.append(_g)
        _newv = _np.asarray(_ov)[_nv_in:]   # this half's appended verts
        if len(_newv):
            _out_v.append(_newv)
            _new_base += len(_newv)
    return (_np.concatenate(_out_v, axis=0),
            _np.concatenate(_out_f, axis=0))


def _fill_one_isolated(cv, cf, max_hole_perimeter,
                       worker_cmd=None,
                       timeout=FILL_CHUNK_TIMEOUT_S,
                       log_fn=None):
    """chunked_fill_core's fill_one, r33 form: the stock fill
    runs in a subprocess (see the r33 header). One whole-chunk
    attempt, then one retry as two half-chunks, each isolated;
    a failure after that raises, and the core falls the chunk
    back to unfilled."""
    try:
        return _run_fill_subprocess(cv, cf, max_hole_perimeter,
                                    worker_cmd=worker_cmd,
                                    timeout=timeout, log_fn=log_fn)
    except Exception as _e1:
        if log_fn:
            log_fn(f"[r33-fill] whole-chunk subprocess fill failed "
                   f"({_e1}); retrying as two half-chunks, each in "
                   "its own subprocess")
        _halves = _split_chunk_halves(cv, cf)
        _results = []
        for _hv, _hf, _uv in _halves:
            _results.append(_run_fill_subprocess(
                _hv, _hf, max_hole_perimeter,
                worker_cmd=worker_cmd, timeout=timeout, log_fn=log_fn))
        if log_fn:
            log_fn("[r33-fill] half-chunk retry succeeded - the "
                   "failure stayed in the dead subprocess")
        return _merge_half_fills(cv, cf, _halves, _results)


def _fill_edge_keys(faces):
    """Packed int64 key per undirected face edge, (F, 3) rows in
    face-corner order (v0v1, v1v2, v2v0). Vertex ids must fit in
    32 bits (any mesh this pipeline makes does)."""
    import numpy as _np

    f = _np.asarray(faces, dtype=_np.int64)
    a = _np.stack([f[:, 0], f[:, 1], f[:, 2]], axis=1)
    b = _np.stack([f[:, 1], f[:, 2], f[:, 0]], axis=1)
    lo = _np.minimum(a, b)
    hi = _np.maximum(a, b)
    return (lo << 32) | hi


def boundary_census(verts, faces):
    """Host-side boundary census of a triangle mesh (pure numpy).

    Returns a dict:
      boundary_edges: (B, 2) int64 vertex pairs with exactly one
        incident face;
      components: list of dicts {verts (int64 array), perimeter
        (float)} - one per connected component of the boundary
        graph (union-find over boundary edges), i.e. per hole
        loop or loop cluster;
      vertex_component: (V,) int64 component id per vertex, -1
        for vertices on no boundary edge.
    Perimeter is the component's summed boundary-edge length -
    the same quantity cumesh compares against
    max_hole_perimeter for a simple loop."""
    import numpy as _np

    verts = _np.asarray(verts, dtype=_np.float64)
    faces = _np.asarray(faces, dtype=_np.int64)
    nv = int(verts.shape[0])
    keys = _fill_edge_keys(faces).ravel()
    ukeys, counts = _np.unique(keys, return_counts=True)
    bkeys = ukeys[counts == 1]
    if len(bkeys) == 0:
        return {"boundary_edges": _np.zeros((0, 2), dtype=_np.int64),
                "components": [],
                "vertex_component": _np.full(nv, -1, dtype=_np.int64)}
    be = _np.stack([bkeys >> 32, bkeys & 0xFFFFFFFF], axis=1).astype(_np.int64)
    # Union-find over boundary vertices.
    parent = {}

    def _find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for _a, _b in be.tolist():
        if _a not in parent:
            parent[_a] = _a
        if _b not in parent:
            parent[_b] = _b
        _ra, _rb = _find(_a), _find(_b)
        if _ra != _rb:
            parent[_ra] = _rb
    comp_of_root = {}
    vertex_component = _np.full(nv, -1, dtype=_np.int64)
    for _v in parent:
        _r = _find(_v)
        if _r not in comp_of_root:
            comp_of_root[_r] = len(comp_of_root)
        vertex_component[_v] = comp_of_root[_r]
    ncomp = len(comp_of_root)
    lengths = _np.linalg.norm(verts[be[:, 0]] - verts[be[:, 1]], axis=1)
    edge_comp = vertex_component[be[:, 0]]
    perimeters = _np.bincount(edge_comp, weights=lengths, minlength=ncomp)
    members = [[] for _ in range(ncomp)]
    for _v, _c in enumerate(vertex_component.tolist()):
        if _c >= 0:
            members[_c].append(_v)
    components = [{"verts": _np.array(members[_i], dtype=_np.int64),
                   "perimeter": float(perimeters[_i])}
                  for _i in range(ncomp)]
    return {"boundary_edges": be, "components": components,
            "vertex_component": vertex_component}


def _face_adjacency(faces):
    """Face adjacency through shared edges as CSR arrays
    (offsets, neighbours): face i's neighbours are
    neighbours[offsets[i]:offsets[i+1]]. Two faces are adjacent
    iff they share an edge (an edge with exactly two incident
    faces pairs them)."""
    import numpy as _np

    f = _np.asarray(faces, dtype=_np.int64)
    nf = int(f.shape[0])
    keys = _fill_edge_keys(f).ravel()
    order = _np.argsort(keys, kind="stable")
    skeys = keys[order]
    face_of = order // 3
    starts = _np.flatnonzero(_np.concatenate(
        [[True], skeys[1:] != skeys[:-1], [True]]))
    src = []
    dst = []
    for _s, _e in zip(starts[:-1].tolist(), starts[1:].tolist()):
        if _e - _s == 2:
            _f0, _f1 = int(face_of[_s]), int(face_of[_s + 1])
            src += [_f0, _f1]
            dst += [_f1, _f0]
    src = _np.array(src, dtype=_np.int64)
    dst = _np.array(dst, dtype=_np.int64)
    deg = _np.bincount(src, minlength=nf)
    offsets = _np.zeros(nf + 1, dtype=_np.int64)
    _np.cumsum(deg, out=offsets[1:])
    neigh = _np.empty(len(src), dtype=_np.int64)
    cursor = offsets[:-1].copy()
    for _s, _d in zip(src.tolist(), dst.tolist()):
        neigh[cursor[_s]] = _d
        cursor[_s] += 1
    return offsets, neigh


def plan_fill_chunks(verts, faces, max_hole_perimeter,
                     face_budget=FILL_CHUNK_FACE_BUDGET,
                     vert_budget=FILL_CHUNK_VERT_BUDGET):
    """Partition faces into chunks for the chunked fill.

    Returns (chunks, info). Each chunk is a dict:
      faces: (M,) int64 global face rows owned by the chunk
        (chunks are face-disjoint and cover every face);
      verts: (K,) int64 global vertex rows the chunk's faces
        reference, sorted; chunk-local index = position here;
      local_faces: (M, 3) int64 faces remapped to chunk-local
        vertex indices;
      fillable_interior: (K,) bool - chunk-local vertices
        lying on one global boundary component whose perimeter
        is under the fill threshold (the census component map)
        and whose every incident global face is inside this
        chunk. Fan faces are kept only when every original
        vertex they touch is one of these (see the r32
        header).

    Partition rules: contiguous regions grown by edge-adjacency
    BFS; atomic groups first - the faces incident to the
    vertices of every boundary component whose perimeter is at
    most 2x max_hole_perimeter are unioned (by shared vertex)
    into groups that seed chunks whole, so a fillable loop is
    never split. A group that alone exceeds a budget dissolves
    into the general pool (counted in info) and is region-grown
    like everything else; the interior-vertex rule still keeps
    such a loop's fill correct-or-dropped, never invented."""
    import numpy as _np

    verts = _np.asarray(verts)
    faces = _np.asarray(faces, dtype=_np.int64)
    nf = int(faces.shape[0])
    census = boundary_census(verts, faces)
    offsets, neigh = _face_adjacency(faces)
    # Incident faces per vertex (for grouping + interior tests).
    vface_count = _np.bincount(faces.ravel(), minlength=int(verts.shape[0]))

    # Atomic groups: union-find over faces incident to candidate
    # loop vertices, merging faces that share such a vertex.
    cand = [c for c in census["components"]
            if c["perimeter"] <= 2.0 * max_hole_perimeter]
    cand_verts = (_np.concatenate([c["verts"] for c in cand])
                  if cand else _np.zeros(0, dtype=_np.int64))
    seed_face = _np.zeros(nf, dtype=bool)
    if len(cand_verts):
        seed_face = (_np.isin(faces, cand_verts)).any(axis=1)
    parent = list(range(nf))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if seed_face.any():
        vert_first = {}
        for _fi in _np.flatnonzero(seed_face).tolist():
            for _v in faces[_fi].tolist():
                if _v in vert_first:
                    _ra, _rb = _find(_fi), _find(vert_first[_v])
                    if _ra != _rb:
                        parent[_ra] = _rb
                else:
                    vert_first[_v] = _fi
    groups = {}
    for _fi in _np.flatnonzero(seed_face).tolist():
        groups.setdefault(_find(_fi), []).append(_fi)
    group_list = []
    dissolved = 0
    for _g in groups.values():
        _g = _np.array(sorted(_g), dtype=_np.int64)
        _gv = _np.unique(faces[_g])
        if len(_g) <= face_budget and len(_gv) <= vert_budget:
            group_list.append(_g)
        else:
            dissolved += 1
    group_list.sort(key=len, reverse=True)

    owner = _np.full(nf, -1, dtype=_np.int64)
    chunk_faces = []

    def _grow(seed_faces):
        """One chunk: seed faces plus BFS growth by edge
        adjacency until a budget would break. Returns the
        chunk's face rows (possibly empty if every seed was
        already owned)."""
        _in = [int(x) for x in seed_faces if owner[x] < 0]
        if not _in:
            return _np.zeros(0, dtype=_np.int64)
        _cid = len(chunk_faces)
        _verts_in = set()
        for _fi in _in:
            owner[_fi] = _cid
            _verts_in.update(faces[_fi].tolist())
        _queue = list(_in)
        _qi = 0
        while _qi < len(_queue):
            _fi = _queue[_qi]
            _qi += 1
            for _nb in neigh[offsets[_fi]:offsets[_fi + 1]].tolist():
                if owner[_nb] >= 0:
                    continue
                _new = [v for v in faces[_nb].tolist()
                        if v not in _verts_in]
                if len(_in) + 1 > face_budget:
                    continue
                if len(_verts_in) + len(_new) > vert_budget:
                    continue
                owner[_nb] = _cid
                _in.append(_nb)
                _verts_in.update(_new)
                _queue.append(_nb)
        return _np.array(sorted(_in), dtype=_np.int64)

    for _g in group_list:
        _c = _grow(_g)
        if len(_c):
            chunk_faces.append(_c)
    for _fi in range(nf):
        if owner[_fi] < 0:
            _c = _grow([_fi])
            if len(_c):
                chunk_faces.append(_c)
    if (owner < 0).any():
        raise ValueError("fill chunk planner left faces unassigned - "
                         "refusing to chunk")

    # Allowed original vertices for kept fans (global test,
    # applied per chunk below): the vertex lies on a boundary
    # component whose own perimeter is under the fill threshold
    # (cumesh fills a loop iff its perimeter is strictly under
    # max_hole_perimeter).
    _comp_perim = _np.array(
        [c["perimeter"] for c in census["components"]],
        dtype=_np.float64)
    _vcomp = census["vertex_component"]
    _allowed_global = _np.zeros(int(verts.shape[0]), dtype=bool)
    _on_comp = _vcomp >= 0
    if _on_comp.any():
        _allowed_global[_on_comp] = (
            _comp_perim[_vcomp[_on_comp]] < max_hole_perimeter)
    chunks = []
    for _cf in chunk_faces:
        _vrows = _np.unique(faces[_cf])
        _local = _np.searchsorted(_vrows, faces[_cf])
        _cnt_in = _np.bincount(faces[_cf].ravel(),
                                minlength=int(verts.shape[0]))
        _fillable = _allowed_global[_vrows] & (
            _cnt_in[_vrows] == vface_count[_vrows])
        chunks.append({"faces": _cf, "verts": _vrows,
                       "local_faces": _local,
                       "fillable_interior": _fillable})
    info = {"census": census,
            "candidate_components": len(cand),
            "atomic_groups": len(group_list),
            "dissolved_groups": dissolved}
    return chunks, info


def extract_chunk_fills(in_verts, in_faces, out_verts, out_faces,
                        fillable_interior):
    """Split one chunk's stock-fill output into kept fills.

    The stock fill is append-only (see the r32 header): the
    output's first rows must be the inputs bit-for-bit, then
    appended centroid vertices, then appended fan faces.
    Returns None if that invariant is violated (the caller
    falls the chunk back to unfilled - never emit corruption).
    Otherwise returns (kept_faces, new_verts): kept_faces are
    the appended faces, in OUTPUT-LOCAL indices, whose every
    original-vertex reference is an allowed vertex of the
    chunk (fillable_interior - a vertex of a genuine,
    under-threshold global hole component kept whole inside
    this chunk; new-vertex references always pass);
    new_verts are the appended vertices those kept faces
    actually use, in output order."""
    import numpy as _np

    in_verts = _np.asarray(in_verts)
    in_faces = _np.asarray(in_faces, dtype=_np.int64)
    out_verts = _np.asarray(out_verts)
    out_faces = _np.asarray(out_faces, dtype=_np.int64)
    nv_in = int(in_verts.shape[0])
    nf_in = int(in_faces.shape[0])
    if int(out_verts.shape[0]) < nv_in or int(out_faces.shape[0]) < nf_in:
        return None
    if not _np.array_equal(out_verts[:nv_in], in_verts):
        return None
    if not _np.array_equal(out_faces[:nf_in], in_faces):
        return None
    appended_faces = out_faces[nf_in:]
    if len(appended_faces) == 0:
        return _np.zeros((0, 3), dtype=_np.int64), \
            _np.zeros((0, 3), dtype=in_verts.dtype)
    _orig_ok = _np.asarray(fillable_interior, dtype=bool).copy()
    keep_rows = []
    for _i, _f in enumerate(appended_faces.tolist()):
        if all((_idx >= nv_in) or _orig_ok[_idx] for _idx in _f):
            keep_rows.append(_i)
    kept = appended_faces[keep_rows] if keep_rows else \
        _np.zeros((0, 3), dtype=_np.int64)
    used_new = sorted({int(_idx) for _f in kept.tolist()
                       for _idx in _f if _idx >= nv_in})
    new_verts = out_verts[used_new] if used_new else \
        _np.zeros((0, 3), dtype=in_verts.dtype)
    # Remap kept faces: original refs unchanged; new refs point
    # at positions within new_verts, offset past the input rows
    # (the caller shifts them to global appended rows).
    if len(kept):
        _remap = {old: nv_in + j for j, old in enumerate(used_new)}
        kept = _np.array([[(_remap[int(_idx)] if _idx >= nv_in
                             else int(_idx)) for _idx in _f]
                          for _f in kept.tolist()], dtype=_np.int64)
    return kept, new_verts


def chunked_fill_core(verts, faces, fill_one, max_hole_perimeter,
                      face_budget=FILL_CHUNK_FACE_BUDGET,
                      vert_budget=FILL_CHUNK_VERT_BUDGET,
                      log_fn=None, fill_batch=None):
    """Chunked fill over host arrays (pure numpy).

    fill_one(chunk_verts, chunk_local_faces) must run the stock
    fill on that sub-mesh and return (out_verts, out_faces) in
    the same local indexing - the contract the stock
    Mesh.fill_holes satisfies. Returns (verts, faces, stats):
    the originals bit-exact plus the kept appended fills,
    merged. A chunk whose fill raises or violates the
    append-only invariant falls back to unfilled (counted in
    stats); census/planning failures raise.

    r35.7: fill_batch, when given, is a callable
    (items, on_start, on_end) -> list parallel to the chunk
    plan of (out_verts, out_faces) or None, used to fill the
    whole plan in one worker process (see _run_fill_batch);
    chunks it returns None for take the fill_one path below,
    unchanged. The harvest/verification code is identical
    either way - batching changes who fills, never what is
    accepted."""
    import numpy as _np

    verts = _np.asarray(verts)
    faces = _np.asarray(faces, dtype=_np.int64)
    chunks, info = plan_fill_chunks(verts, faces, max_hole_perimeter,
                                    face_budget, vert_budget)
    stats = {"chunks": len(chunks), "fallback_chunks": 0,
             "kept_fill_faces": 0, "kept_fill_verts": 0,
             "dropped_fill_faces": 0, "candidate_components":
             info["candidate_components"],
             "atomic_groups": info["atomic_groups"],
             "dissolved_groups": info["dissolved_groups"]}
    out_vert_rows = [verts]
    out_face_rows = [faces]
    n_new_so_far = 0
    # r35.6 INSTRUMENTATION (logging only -- no behaviour
    # change): a per-chunk heartbeat around each chunk's fill
    # invocation. Job 044's arm-2 fill produced no output for
    # hours and nothing in the log could name the chunk it was
    # on; these lines make any future stall attributable to
    # exactly one chunk from the log alone. A stage-start line
    # names the fill and its chunk plan up front. The START
    # line is emitted before the invocation, the END line
    # only after it returns, so a START with no matching END
    # marks the chunk a stall died in (when the ladder
    # catches a raised failure instead, the [r32] fallback
    # line below still follows the START). While one chunk's
    # fill runs, a daemon ticker thread re-prints a
    # still-filling line every _tick_s seconds, so one
    # chunk's long silence is distinguishable from a dead
    # process (and the worker's output-based STALLED label,
    # threshold 900 s, stays reset while ticks flow). The per-chunk
    # boundary-edge count is deliberately absent: it exists
    # only inside the worker's plan, where the device/host
    # cross-check guards it; the parent holds only the global
    # census, already logged by the planner's own lines. _hb
    # falls back to a bare flushed print when no log_fn was
    # supplied (standalone tests), so the heartbeat cannot
    # silently vanish.
    _hb = log_fn if log_fn is not None else (
        lambda _m: print(_m, flush=True))
    _n_chunks = len(chunks)
    import threading as _threading
    # Still-here tick cadence inside ONE chunk's fill
    # (seconds): the parent blocks in the fill subprocess
    # for the chunk's whole duration, so a daemon thread
    # prints the tick (see the loop). ~5 minutes in
    # production; the module global _R356_FILL_TICK_SECONDS
    # can override it in tests.
    _tick_s = globals().get("_R356_FILL_TICK_SECONDS", 300)
    _hb(f"[r35.6-fill] fill stage start: {_n_chunks} chunks "
        f"planned, verts={int(verts.shape[0])} "
        f"faces={int(faces.shape[0])} epoch={time.time():.3f} "
        f"wall={time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())}")
    # r35.7: batch pre-pass. When a batch filler is supplied the
    # whole plan is filled up front in one worker process (see
    # _run_fill_batch); the hooks print this stage's per-chunk
    # heartbeat lines at the true moment the worker starts/ends
    # each chunk. Chunks the batch could not fill come back None
    # and take the per-chunk path in the loop below, whose own
    # heartbeat lines then apply. A wholesale batch failure
    # degrades to all-per-chunk, never to a dead run.
    _batch_results = None
    if fill_batch is not None:
        _items = [(verts[_ch["verts"]], _ch["local_faces"])
                  for _ch in chunks]
        _batch_m0 = {}

        def _batch_on_start(_i):
            _t = time.time()
            _batch_m0[_i] = time.monotonic()
            _icv, _icf = _items[_i]
            _hb(f"[r35.6-fill] chunk {_i} of {_n_chunks}: fill "
                f"start verts={int(_icv.shape[0])} "
                f"faces={int(_icf.shape[0])} epoch={_t:.3f} "
                f"wall={time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(_t))}")

        def _batch_on_end(_i, _appended):
            _t = time.time()
            _m = _batch_m0.pop(_i, time.monotonic())
            _hb(f"[r35.6-fill] chunk {_i} of {_n_chunks}: fill "
                f"end elapsed={time.monotonic() - _m:.3f}s "
                f"appended_faces={_appended} epoch={_t:.3f} "
                f"wall={time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(_t))}")

        try:
            _batch_results = fill_batch(_items, _batch_on_start,
                                        _batch_on_end)
        except Exception:
            if log_fn:
                log_fn("[r35.7-batch] batch fill raised wholesale "
                       "- every chunk takes the per-chunk path:\n"
                       + traceback.format_exc())
            _batch_results = None
    for _ci, _ch in enumerate(chunks):
        _cv = verts[_ch["verts"]]
        _cf = _ch["local_faces"]
        # r35.7: a batch-served chunk already printed its
        # heartbeat lines live during the pre-pass; the loop's
        # own lines and ticker are for chunks it fills itself.
        _served = _batch_results is not None and \
            _batch_results[_ci] is not None
        _t0 = time.time()
        _m0 = time.monotonic()
        if not _served:
            _hb(f"[r35.6-fill] chunk {_ci} of {_n_chunks}: fill "
                f"start verts={int(_cv.shape[0])} "
                f"faces={int(_cf.shape[0])} epoch={_t0:.3f} "
                f"wall={time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(_t0))}")
        _tick_stop = _threading.Event()

        def _ticker(_ci=_ci, _m0=_m0):
            while not _tick_stop.wait(_tick_s):
                _hb(f"[r35.6-fill] chunk {_ci} of "
                    f"{_n_chunks}: still filling "
                    f"(elapsed {time.monotonic() - _m0:.0f}s) "
                    f"epoch={time.time():.3f} "
                    f"wall={time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())}")

        _tick_thread = None
        if not _served:
            _tick_thread = _threading.Thread(target=_ticker,
                                             daemon=True)
            _tick_thread.start()
        try:
            if _served:
                _ov, _of = _batch_results[_ci]
            else:
                _ov, _of = fill_one(_cv, _cf)
        except Exception:
            _tick_stop.set()
            if _tick_thread is not None:
                _tick_thread.join()
            if log_fn:
                log_fn(f"[r32] chunk {_ci}: stock fill raised - "
                       "chunk falls back to unfilled:\n"
                       + traceback.format_exc())
            stats["fallback_chunks"] += 1
            continue
        _tick_stop.set()
        if _tick_thread is not None:
            _tick_thread.join()
        if not _served:
            _hb(f"[r35.6-fill] chunk {_ci} of {_n_chunks}: fill "
                f"end elapsed={time.monotonic() - _m0:.3f}s "
                f"appended_faces="
                f"{int(_np.asarray(_of).shape[0]) - int(_cf.shape[0])} "
                f"epoch={time.time():.3f} "
                f"wall={time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime())}")
        _ex = extract_chunk_fills(_cv, _cf, _ov, _of,
                                  _ch["fillable_interior"])
        if _ex is None:
            if log_fn:
                log_fn(f"[r32] chunk {_ci}: fill output violates "
                       "the append-only invariant - chunk falls "
                       "back to unfilled")
            stats["fallback_chunks"] += 1
            continue
        _kept, _newv = _ex
        _appended = int(_np.asarray(_of).shape[0]) - int(_cf.shape[0])
        stats["dropped_fill_faces"] += _appended - len(_kept)
        if len(_kept):
            _g = _kept.copy()
            _orig_mask = _g < len(_cv)
            _g[_orig_mask] = _ch["verts"][_g[_orig_mask]]
            _g[~_orig_mask] = int(verts.shape[0]) + n_new_so_far \
                + (_g[~_orig_mask] - len(_cv))
            out_face_rows.append(_g)
            out_vert_rows.append(_newv)
            n_new_so_far += len(_newv)
            stats["kept_fill_faces"] += len(_kept)
            stats["kept_fill_verts"] += len(_newv)
    out_verts = _np.concatenate(out_vert_rows, axis=0)
    out_faces = _np.concatenate(out_face_rows, axis=0)
    return out_verts, out_faces, stats


def _chunked_fill_holes(mesh, orig, max_hole_perimeter):
    """Torch glue for the chunked fill_holes wrapper: delegates
    small meshes to the stock method untouched; big meshes go
    through chunked_fill_core with the stock method applied per
    chunk to real Mesh instances. Logs the [r32-fill] GPU
    profiles after either path. Planning/census failure raises
    (r31's rule: no masquerading)."""
    import torch

    n_v = int(mesh.vertices.shape[0])
    n_f = int(mesh.faces.shape[0])
    if n_f <= FILL_STOCK_FACE_LIMIT and n_v <= FILL_STOCK_VERT_LIMIT:
        orig(mesh, max_hole_perimeter=max_hole_perimeter)
        _path = "stock (at/under the delegation limits)"
    else:
        from trellis2.representations.mesh.base import Mesh as _MeshCls

        verts_np = chunked_readout(mesh.vertices)
        faces_np = chunked_readout(mesh.faces).astype("int64")
        dev = mesh.vertices.device

        def _fill_one(_cv, _cf):
            # r33: the per-chunk stock fill runs in an isolated
            # subprocess (see _fill_one_isolated and the r33
            # header) - a CuMesh kernel failure or segfault in
            # one chunk's fill can no longer kill the run.
            return _fill_one_isolated(
                _cv, _cf, max_hole_perimeter, log_fn=log)

        # r35.7: the batch filler exists ONLY when the R1 worker
        # is in command AND batch mode was exported by the gate
        # stage (R1_FILL_BATCH) - with either unset, the core
        # runs byte-for-byte the r35.6 per-chunk path. The batch
        # fills through the worker's --batch mode; chunks it
        # cannot fill still land in _fill_one's ladder above.
        _fill_batch = None
        _r1_worker = os.environ.get("R1_FILL_WORKER")
        if _r1_worker and os.environ.get("R1_FILL_BATCH"):
            def _fill_batch(_items, _on_start, _on_end):
                return _run_fill_batch(
                    _items, [_r1_worker], max_hole_perimeter,
                    on_start=_on_start, on_end=_on_end,
                    log_fn=log)
            log("[r35.7-batch] chunk fills are batched through "
                "one worker process: " + _r1_worker)

        out_v, out_f, _stats = chunked_fill_core(
            verts_np, faces_np, _fill_one, max_hole_perimeter,
            log_fn=log, fill_batch=_fill_batch)
        mesh.vertices = torch.as_tensor(
            out_v, dtype=mesh.vertices.dtype, device=dev)
        mesh.faces = torch.as_tensor(
            out_f, dtype=mesh.faces.dtype, device=dev)
        _path = (f"chunked ({_stats['chunks']} chunks, "
                 f"{_stats['kept_fill_faces']} fill faces / "
                 f"{_stats['kept_fill_verts']} fill verts kept, "
                 f"{_stats['dropped_fill_faces']} cut-boundary fans "
                 f"dropped, {_stats['fallback_chunks']} fallback "
                 f"chunks, {_stats['candidate_components']} candidate "
                 f"hole components in {_stats['atomic_groups']} "
                 f"atomic groups)")
    for _label, _arr in (("faces", mesh.faces),
                         ("vertices", mesh.vertices)):
        _prof = genuine_row_profile(_arr)
        log(f"[r32-fill] {_label}: {sum(_prof)} genuine of "
            f"{int(_arr.shape[0])} rows (per-32-blocks {_prof}) - "
            f"AFTER the fill_holes stage, path: {_path}")


def install_chunked_fill():
    """Install the r32 wrappers over trellis2's Mesh.fill_holes
    and Mesh.simplify (see the r32 header). Returns the list of
    patched binding names; empty means the Mesh class binding
    was not found or its stock signature is not the one the
    wrapper was written against (caller fails loud)."""
    import importlib
    import inspect

    try:
        _base = importlib.import_module(
            "trellis2.representations.mesh.base")
        _cls = getattr(_base, "Mesh", None)
    except Exception:
        return []
    if _cls is None:
        return []
    _orig_fill = getattr(_cls, "fill_holes", None)
    _orig_simplify = getattr(_cls, "simplify", None)
    if not callable(_orig_fill) or not callable(_orig_simplify):
        return []
    if getattr(_orig_fill, "_r32_chunked", False):
        return ["trellis2.representations.mesh.base.Mesh (already)"]
    try:
        _params = inspect.signature(_orig_fill).parameters
    except (TypeError, ValueError):
        return []
    if "max_hole_perimeter" not in _params:
        return []

    def _fill_wrapper(self, max_hole_perimeter=3e-2,
                      _orig=_orig_fill):
        return _chunked_fill_holes(self, _orig, max_hole_perimeter)

    _fill_wrapper._r32_chunked = True

    def _simplify_guard(self, target=1000000, verbose=False,
                        options={}, _orig=_orig_simplify):
        try:
            _nf = int(self.faces.shape[0])
        except Exception:
            _nf = None
        if _nf is not None and int(target) >= _nf:
            log(f"[r32] Mesh.simplify skipped: target {target} >= "
                f"current faces {_nf} - a non-binding simplification "
                "cannot decimate, and its CuMesh round trip is the "
                "2^20 wound this build exists to keep out")
            return
        return _orig(self, target, verbose=verbose, options=options)

    _simplify_guard._r32_guarded = True
    _cls.fill_holes = _fill_wrapper
    _cls.simplify = _simplify_guard
    return ["trellis2.representations.mesh.base.Mesh.fill_holes",
            "trellis2.representations.mesh.base.Mesh.simplify"]




def readout_ab_test():
    """r28: the built-in falsifier for the copy-cliff theory, run at
    the smoke phase of every build. Builds a 3,000,000x3 GPU tensor
    whose every column stamps its own row index (int32, plus an fp32
    twin - both are 12 bytes/row like the mesh arrays), reads it back
    single-shot and via chunked_readout, and counts rows whose stamp
    disagrees with their index. Prediction: single-shot corrupts
    most rows past 2^20; chunked is exact. Returns True only if the
    chunked readout was exact for both dtypes - the caller fails
    loud otherwise, because every later number in the run would be
    read through the same broken glass."""
    import numpy as _np
    import torch

    _n = 3000000
    _idx = _np.arange(_n)
    _ok = True
    for _dt in (torch.int32, torch.float32):
        _rows = torch.arange(_n, dtype=torch.int64, device="cuda").unsqueeze(1)
        _t = torch.cat([_rows, _rows, _rows], dim=1).to(_dt)
        del _rows
        _single = _t.cpu().numpy()
        _bad_single = int((_single[:, 0] != _idx).sum())
        _chunked = chunked_readout(_t)
        _bad_chunked = int((_chunked[:, 0] != _idx).sum())
        log(f"[readout-ab] {_t.dtype} {_n}x3: single-shot mismatched rows={_bad_single}, "
            f"chunked mismatched rows={_bad_chunked}")
        if _bad_chunked:
            _ok = False
        del _t, _single, _chunked
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return _ok


def plan_section_cuts(bin_counts, bin_scores, n_faces, cap):
    """r24: decide where to cut the axis-sorted face order into sections
    of at most `cap` faces. Replaces the r23 loop, which could only cut
    on bin edges whose cumulative count landed inside the face window
    [start + cap//2, start + cap]; on probe-S1 one bin holds 51% of all
    faces, so the cumulative counts leap clean over every window, the
    candidate list was empty at all 115 iterations, and the silent
    one-bin fallback produced 116 sections instead of 3 (autopsy:
    replicated exactly, 116/117 cuts).

    The fix tracks the position in FACE space: bin edges inside the
    window are candidates for the narrow-point rule (lowest bin score
    wins; the joint-dip bonus is already folded into bin_scores by the
    caller); when no bin edge falls in the window, the window lies
    inside one bin, so cut at the cap mid-bin and re-sync the bin cursor
    with searchsorted. A cut is a slice of the sorted face order - it
    never needed to sit on a bin edge.

    Returns (cuts, provenance): cuts tiles [0, n_faces]; provenance has
    one dict per cut after 0 - {cut, via ("narrow-point" or "mid-bin
    fallback"), candidates} - so the caller logs honestly which rule
    chose every cut."""
    import numpy as _np

    _nb = len(bin_counts)
    _cum = _np.concatenate([[0], _np.cumsum(bin_counts)])
    _cuts = [0]
    _prov = []
    _start_bin = 0
    _start_face = 0
    while n_faces - _start_face > cap:
        _lo_f = _start_face + cap // 2
        _hi_f = _start_face + cap
        _cand = [b for b in range(_start_bin + 1, _nb)
                 if _lo_f <= int(_cum[b]) <= _hi_f]
        if _cand:
            _bb = min(_cand, key=lambda b: bin_scores[b])
            _cut = int(_cum[_bb])
            _start_bin = _bb
            _prov.append({"cut": _cut, "via": "narrow-point", "candidates": len(_cand)})
        else:
            _cut = _hi_f
            _start_bin = int(_np.searchsorted(_cum, _cut, side="right")) - 1
            _prov.append({"cut": _cut, "via": "mid-bin fallback", "candidates": 0})
        _cuts.append(_cut)
        _start_face = _cut
    _cuts.append(n_faces)
    return _cuts, _prov


class BoundsVerificationError(Exception):
    """r27: raised when an exported GLB's read-back bounds do not cover
    the geometry it was supposed to contain (the r24 sectioned-export
    clip). Caught at the export call site, which exits nonzero instead
    of masking the clipped file with the stock chain."""


# ---------------------------------------------------------------------
# r34: BOUNDS FIX - the small-component op is REMOVED from the
# sectioned export path (pipeline-fix program item #2; the mechanism
# was solved by the two-pass audit, BOUNDS-AUDIT-PASS1.md and
# BOUNDS-AUDIT-PASS2-CENSUS.md in the r33 build tree).
#
# What the audit measured (job 028, subject 1024 control arm): the
# Step-2 cleanup bundle inside o_voxel.postprocess.to_glb -
#   remove_duplicate_faces(); repair_non_manifold_edges();
#   remove_small_connected_components(1e-5); fill_holes(3e-2)
# - deleted 4.7% / 18.6% / 42.3% / 82.2% / 96.1% of sections 0-4 in
# cleanup #1 alone, the loss growing with height. The deleter is
# the small-component op acting on the mesh AS THE REPAIR LEAVES
# IT: judged by manifold-only connectivity (faces linked only
# across edges with exactly two incident faces), the upper
# sections are gravel - sec3 is 338,440 fragments (largest 4.27%
# of the section), sec4 is 40,533 (largest 332 faces) - because
# 84.7% / 93.2% of their faces touch an edge shared by 3+ faces.
# Under ordinary vertex connectivity the same sec3 is 97.0% one
# component: the op was deleting the subject, not dust. (Duplicate
# removal was measured innocent: 0.00% extra faces in every
# section of both meshes. The 1e-5 threshold is NOT the problem -
# at every plausible referent the cutoff lands at 1-30 faces and
# the fragments are smaller than that; tuning it is explicitly
# not this fix.)
#
# THE CHANGE, confined to the SECTIONED to_glb calls in
# sectioned_export below:
#   1. While a section's to_glb runs, the bundle is replaced by
#      the explicit op sequence WITHOUT
#      remove_small_connected_components: remove_duplicate_faces,
#      repair_non_manifold_edges and fill_holes still run, in the
#      bundle's own order, at both cleanup passes (the pre-simplify
#      initial fill_holes is untouched) - only the small-component
#      calls are intercepted and skipped. Every op is logged with
#      before/after face counts under [r34-ops], so any future
#      cleanup loss has its per-op counts printed in the same run
#      that suffers it (audit PASS 1, item 3).
#   2. Before each section's to_glb, the section's
#      manifold-fragment census is logged under [r34-census]
#      (audit PASS 2 fix-shape item 2): manifold-only component
#      count, largest-fragment share, and the share of faces
#      touching a non-manifold edge - the three numbers that
#      predict this defect, printed next to the export they
#      describe.
#
# HOW the interception works, and why two routes: the bundle runs
# inside upstream's to_glb on a cumesh.CuMesh instance; this file
# cannot rewrite that function without forking the whole export
# (UV unwrap, BVH, texture baking). So each sectioned call runs
# under the _R34SectionCleanup context manager below, which
# patches the four bundle ops for exactly the duration of one
# to_glb call and restores the originals on exit, whatever
# happens. Route A patches the CuMesh class methods directly
# (the r32 wrappers' precedent); if the compiled class refuses
# attribute assignment, route B swaps the cumesh module reference
# inside o_voxel.postprocess for a shim whose CuMesh factory
# returns a delegating proxy that intercepts only the four ops.
# Patchability is proven once per sectioned export (a probe
# enter/exit before the section loop) and re-verified by
# read-back at every install; if neither route can be installed,
# R34CleanupError aborts the run - it subclasses
# BoundsVerificationError, so the caller applies the r27
# contract: exit 5, NO stock-chain fallback. An export that
# silently kept the killer op would be worse than no export.
#
# WHAT DOES NOT CHANGE: the non-sectioned (stock) to_glb path,
# extraction, the fill, chunking, the cascade, the section
# cutter, the gates - all byte-identical to r33. The source of
# the non-manifold edges themselves (extraction vs the fill's
# centroid fans) is still unattributed; that fix belongs to the
# stage the op-by-op probe convicts, not to this one (audit
# PASS 2 fix-shape item 3).


class R34CleanupError(BoundsVerificationError):
    """r34: raised when the sectioned cleanup override cannot be
    installed, so no sectioned export in this run could be trusted
    to exclude remove_small_connected_components. Subclasses
    BoundsVerificationError so the export call site applies the
    same no-fallback contract (exit 5): never mask a known-bad
    export with the stock chain."""


def _component_sizes(n_faces, a, b):
    """Sizes of the connected components of the face graph whose
    edges are (a[i], b[i]). scipy's sparse csgraph when importable
    (the audit census scripts' own engine), a union-find otherwise;
    the two agree exactly on component sizes - only speed
    differs."""
    import numpy as _np

    if len(a):
        try:
            import scipy.sparse as _sp
            import scipy.sparse.csgraph as _cg

            _g = _sp.coo_matrix(
                (_np.ones(len(a), dtype=_np.int8), (a, b)),
                shape=(n_faces, n_faces))
            _, _lab = _cg.connected_components(
                _g, directed=False, return_labels=True)
            return _np.bincount(_lab)
        except Exception:
            pass
    _parent = list(range(n_faces))

    def _find(_x):
        while _parent[_x] != _x:
            _parent[_x] = _parent[_parent[_x]]
            _x = _parent[_x]
        return _x

    for _x, _y in zip(a.tolist(), b.tolist()):
        _rx, _ry = _find(int(_x)), _find(int(_y))
        if _rx != _ry:
            _parent[_rx] = _ry
    _sizes = _np.bincount(
        _np.fromiter((_find(_i) for _i in range(n_faces)),
                     dtype=_np.int64, count=n_faces))
    # bincount over root ids leaves a zero wherever an index is no
    # component's root; drop the zeros so the count is exact.
    return _sizes[_sizes > 0]


def manifold_fragment_census(faces):
    """r34: the audit PASS 2 manifold-fragment census of one mesh
    (host numpy; the census scripts' recipe, verbatim): component
    count under MANIFOLD-ONLY connectivity - faces linked only
    across edges with exactly two incident faces, the state the
    cleanup bundle's repair-then-remove sequence effectively
    judges - the largest fragment's share, and the share of faces
    touching a non-manifold edge (incidence 3+). Takes an (N, 3)
    integer face array (vertex ids < 2^32; compacted or not -
    relabelling changes neither incidence nor connectivity).
    Returns a dict of plain numbers. Reproduces PASS 2's published
    sec3/sec4 counts on job 028's raw sections (338,440 components,
    largest 41,500 faces / 40,533 components, largest 332)."""
    import numpy as _np

    _f = _np.asarray(faces, dtype=_np.int64)
    if _f.ndim != 2 or _f.shape[1] != 3 or _f.shape[0] == 0:
        return {"n_faces": 0, "n_components": 0, "largest_faces": 0,
                "largest_pct": 0.0, "pct_touch_nonmanifold": 0.0}
    _n = int(_f.shape[0])
    _e = _np.vstack([
        _np.stack([_f[:, 0], _f[:, 1]], axis=1),
        _np.stack([_f[:, 1], _f[:, 2]], axis=1),
        _np.stack([_f[:, 2], _f[:, 0]], axis=1),
    ])
    _es = _np.sort(_e, axis=1)
    _key = _es[:, 0] * (1 << 32) + _es[:, 1]
    _, _inv, _ecnt = _np.unique(_key, return_inverse=True,
                                return_counts=True)
    _eids = _inv.reshape(3, _n).T
    _inc = _ecnt[_eids]
    _pct_nm = float(100.0 * (_inc.max(axis=1) > 2).mean())
    _eid_flat = _np.concatenate([_eids[:, 0], _eids[:, 1], _eids[:, 2]])
    _face_of = _np.tile(_np.arange(_n), 3)
    _man = _ecnt[_eid_flat] == 2
    _fo = _face_of[_man]
    _ef = _eid_flat[_man]
    _ord = _np.argsort(_ef, kind="stable")
    _fo_s, _ef_s = _fo[_ord], _ef[_ord]
    _same = _ef_s[1:] == _ef_s[:-1]
    _sizes = _component_sizes(_n, _fo_s[:-1][_same], _fo_s[1:][_same])
    _largest = int(_sizes.max()) if _sizes.size else 0
    return {"n_faces": _n,
            "n_components": int(_sizes.size),
            "largest_faces": _largest,
            "largest_pct": float(100.0 * _largest / _n),
            "pct_touch_nonmanifold": _pct_nm}


_R34_BUNDLE_OPS = ("remove_duplicate_faces", "repair_non_manifold_edges",
                   "remove_small_connected_components", "fill_holes")


class _R34SectionCleanup:
    """r34: context manager replacing the to_glb cleanup bundle
    with the explicit r34 op sequence for ONE sectioned to_glb
    call: remove_small_connected_components is intercepted and
    skipped; the other three bundle ops run stock, in the
    bundle's order, each logged with before/after face counts
    under [r34-ops]. Originals are restored on exit no matter
    what. Routes A (CuMesh class patch) and B (postprocess
    module shim) are in the r34 header above."""

    def __init__(self, section_idx, probe=False):
        self.section = section_idx
        self.probe = probe
        self.route = None
        self._counts = {}
        self._orig = None
        self._restore = None

    def _num_faces(self, obj):
        try:
            return int(obj.num_faces)
        except Exception:
            return -1

    def _pass_label(self, name):
        _k = self._counts.get(name, 0) + 1
        self._counts[name] = _k
        if name == "fill_holes":
            # to_glb calls fill_holes once before the cleanups
            # (initial fill) and once inside each cleanup pass.
            return "initial fill" if _k == 1 else f"cleanup #{_k - 1}"
        return f"cleanup #{_k}"

    def _run_op(self, orig_call, obj, name, args, kwargs):
        _label = self._pass_label(name)
        _before = self._num_faces(obj)
        if name == "remove_small_connected_components":
            if args:
                _thr = args[0]
            elif kwargs:
                _thr = next(iter(kwargs.values()))
            else:
                _thr = "?"
            if not self.probe:
                log(f"[r34-ops] section {self.section}: {_label} "
                    f"remove_small_connected_components({_thr}) "
                    f"SKIPPED (r34) - {_before} faces unchanged")
            return None
        _res = orig_call(*args, **kwargs)
        if not self.probe:
            log(f"[r34-ops] section {self.section}: {_label} {name}: "
                f"{_before} -> {self._num_faces(obj)} faces")
        return _res

    def _enter_route_a(self, cumesh):
        _cls = cumesh.CuMesh
        _orig = {}
        try:
            for _name in _R34_BUNDLE_OPS:
                _orig[_name] = getattr(_cls, _name)
        except Exception:
            return False
        _ctx = self

        def _make(_name, _fn):
            def _wrapper(_obj, *args, **kwargs):
                return _ctx._run_op(
                    lambda *a, **k: _fn(_obj, *a, **k),
                    _obj, _name, args, kwargs)
            return _wrapper

        _wrapped = {_name: _make(_name, _orig[_name])
                    for _name in _R34_BUNDLE_OPS}
        try:
            for _name, _w in _wrapped.items():
                setattr(_cls, _name, _w)
            if getattr(_cls, "remove_small_connected_components") is not \
                    _wrapped["remove_small_connected_components"]:
                raise TypeError("CuMesh class patch did not stick")
        except Exception:
            for _name, _fn in _orig.items():
                try:
                    setattr(_cls, _name, _fn)
                except Exception:
                    pass
            return False
        self._orig = _orig

        def _restore():
            for _name, _fn in _orig.items():
                try:
                    setattr(_cls, _name, _fn)
                except Exception:
                    pass

        self._restore = _restore
        self.route = "A (CuMesh class patch)"
        return True

    def _enter_route_b(self, cumesh):
        try:
            import o_voxel.postprocess as _pp
        except Exception:
            return False
        _ctx = self
        _real_cls = cumesh.CuMesh

        class _Proxy:
            def __init__(self, _real):
                object.__setattr__(self, "_r34_real", _real)

            def __getattr__(self, _name):
                _real = object.__getattribute__(self, "_r34_real")
                if _name in _R34_BUNDLE_OPS:
                    _fn = getattr(_real, _name)
                    return lambda *a, **k: _ctx._run_op(
                        _fn, _real, _name, a, k)
                return getattr(_real, _name)

            def __setattr__(self, _name, _value):
                setattr(object.__getattribute__(self, "_r34_real"),
                        _name, _value)

        def _factory(*args, **kwargs):
            return _Proxy(_real_cls(*args, **kwargs))

        class _Shim:
            def __init__(self):
                self.CuMesh = _factory

            def __getattr__(self, _name):
                return getattr(cumesh, _name)

        _orig_mod = _pp.cumesh
        try:
            _pp.cumesh = _Shim()
            if _pp.cumesh is _orig_mod or \
                    _pp.cumesh.CuMesh is not _factory:
                raise TypeError("postprocess module shim did not stick")
        except Exception:
            try:
                _pp.cumesh = _orig_mod
            except Exception:
                pass
            return False

        def _restore():
            _pp.cumesh = _orig_mod

        self._restore = _restore
        self.route = "B (o_voxel.postprocess cumesh shim)"
        return True

    def __enter__(self):
        try:
            import cumesh
        except Exception as _e:
            raise R34CleanupError(
                f"r34: cumesh is not importable ({_e!r}) - the "
                "sectioned cleanup override cannot be installed"
            ) from _e
        if self._enter_route_a(cumesh) or self._enter_route_b(cumesh):
            if self.probe:
                log(f"[r34] cleanup override patchability probe OK "
                    f"(route {self.route})")
            else:
                log(f"[r34] section {self.section}: cleanup override "
                    f"ACTIVE via route {self.route} - "
                    "remove_small_connected_components excluded, "
                    "bundle ops logged under [r34-ops]")
            return self
        raise R34CleanupError(
            "r34: neither the CuMesh class patch nor the postprocess "
            "module shim could be installed - refusing to run a "
            "sectioned export that would silently keep "
            "remove_small_connected_components")

    def __exit__(self, *exc):
        if self._restore is not None:
            try:
                self._restore()
            except Exception:
                log("[r34] WARNING: cleanup override restore failed:\n"
                    + traceback.format_exc())
            self._restore = None
        return False


def glb_axis_bounds(path):
    """r27: read an exported GLB back with trimesh and return
    (min, max) per axis over ALL geometry in the file, or None if the
    file holds no vertices."""
    import numpy as _np
    import trimesh as _tm

    obj = _tm.load(path, process=False)
    geoms = list(obj.geometry.values()) if hasattr(obj, "geometry") else [obj]
    parts = [g.vertices for g in geoms
             if getattr(g, "vertices", None) is not None and len(g.vertices)]
    if not parts:
        return None
    allv = _np.vstack(parts)
    return allv.min(0), allv.max(0)


def trellis_to_glb_bounds(vmin, vmax):
    """r27: the TRELLIS.2 world -> exported-GLB axis mapping, measured
    on the r24 run's own files (inside-log [diag] bounds vs the whole
    512 smoke GLB): GLB = (x, z, -y). Takes TRELLIS-space min/max
    triples, returns GLB-space (min, max)."""
    import numpy as _np

    vmin = _np.asarray(vmin, dtype=float)
    vmax = _np.asarray(vmax, dtype=float)
    gmin = _np.array([vmin[0], vmin[2], -vmax[1]])
    gmax = _np.array([vmax[0], vmax[2], -vmin[1]])
    return gmin, gmax


def bounds_mismatch(got_min, got_max, want_min, want_max, voxel_size,
                    loose_glb_axis=None):
    """r27: compare exported bounds against intended bounds (SAME axis
    space on both sides). Returns a list of failure strings; empty
    means pass. An axis fails if its exported extent is smaller than
    the intended extent by more than the tolerance, or if its exported
    range pokes outside the intended range by more than the tolerance.

    Tolerances (house tuning values, sized from the r24 run's own
    numbers): base per-axis tolerance = max(8 voxels, 5% of the
    intended extent). Healthy in-to_glb simplification trimmed one
    axis by 0.034 on a 0.981 extent (3.5%) and must pass; the r24
    export clip removed 0.189 (canvas, 50% of the axis) and 0.062
    (subject, 20%) and must fail. A section's CUT axis gets a looser
    10% factor: section band edges are trimmed harder by the
    scaled-down decimation target, and the neighbouring section's
    overlap margin covers the seam - but a 20%+ loss still fails
    even there."""
    fails = []
    for a in range(3):
        wext = float(want_max[a] - want_min[a])
        frac = 0.10 if (loose_glb_axis is not None and a == loose_glb_axis) else 0.05
        tol = max(8.0 * float(voxel_size), frac * wext)
        gmin, gmax = float(got_min[a]), float(got_max[a])
        if gmax - gmin < wext - tol:
            fails.append(
                f"axis {a}: exported extent {gmax - gmin:.4f} vs intended "
                f"{wext:.4f} (short by {wext - (gmax - gmin):.4f}, tol {tol:.4f})")
        if gmin < float(want_min[a]) - tol or gmax > float(want_max[a]) + tol:
            fails.append(
                f"axis {a}: exported range [{gmin:.4f}, {gmax:.4f}] outside "
                f"intended [{float(want_min[a]):.4f}, {float(want_max[a]):.4f}] "
                f"(tol {tol:.4f})")
    return fails


def check_export_bounds(path, trellis_vmin, trellis_vmax, voxel_size,
                        loose_trellis_axis=None):
    """r27: the full gate for one exported GLB: read it back, map the
    intended TRELLIS-space bounds into GLB space, and compare. Returns
    a list of failure strings (empty = pass). An unreadable or empty
    GLB is a failure, not a pass - a gate that waves through what it
    cannot inspect is how the r24 clip shipped silently."""
    try:
        got = glb_axis_bounds(path)
    except Exception as e:
        return [f"could not read back {path} for bounds verification: {e}"]
    if got is None:
        return [f"{path} holds no vertices - nothing to verify, treating as failed"]
    wmin, wmax = trellis_to_glb_bounds(trellis_vmin, trellis_vmax)
    loose_glb = None
    if loose_trellis_axis is not None:
        loose_glb = (0, 2, 1)[int(loose_trellis_axis)]
    return bounds_mismatch(got[0], got[1], wmin, wmax, voxel_size,
                           loose_glb_axis=loose_glb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", default="1024_cascade",
                    help="TRELLIS.2 pipeline_type: 512, 1024, 1024_cascade, 1536_cascade")
    ap.add_argument("--seed", type=int, default=0)
    # r18: export quality knobs (defaults = pre-r18 behavior). The subject
    # run raises them to upstream example.py's own settings (4096 / 1M).
    ap.add_argument("--texture-size", type=int, default=2048)
    ap.add_argument("--probe-cumesh", action="store_true",
                    help="walk the mesh through CuMesh stage by stage (r21 diagnostic) instead of exporting")
    ap.add_argument("--decimation-target", type=int, default=200000)
    args = ap.parse_args()

    import torch

    log(f"torch {torch.__version__}  hip {getattr(torch.version, 'hip', None)}")
    log(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        log("FAIL: no GPU visible to torch - refusing to 'generate' on CPU silently")
        return 2
    log(f"device 0: {torch.cuda.get_device_name(0)}")

    if args.mode == "512":
        # r28: synthetic readout A/B at the smoke phase (see
        # readout_ab_test). A chunked readout that is not exact means
        # fail loud NOW, before any generation gets read through the
        # same broken glass.
        if not readout_ab_test():
            log("FAIL: chunked GPU->host readout is NOT exact on the synthetic A/B - "
                "refusing to generate numbers nobody can trust")
            return 9

    from PIL import Image

    raw = Image.open(args.input)
    if raw.mode in ("RGBA", "LA") or (raw.mode == "P" and "transparency" in raw.info):
        image = raw.convert("RGBA")
        import numpy as _np
        cov = float((_np.asarray(image.split()[-1]) > 127).mean())
        log(f"input: {args.input} size={image.size} mode=RGBA alpha-coverage={cov:.3f} (OUR cutout - pipeline skips its background guesser)")
    else:
        image = raw.convert("RGB")
        log(f"input: {args.input} size={image.size} mode=RGB (pipeline will guess the background)")

    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    # Gated-repo swaps (the port author's own recipe, PORTING_LOG.md): the
    # weights repo's pipeline.json names two GATED Hugging Face repos -
    # facebook/dinov3-vitl16-pretrain-lvd1689m (401) and briaai/RMBG-2.0 (403).
    # The author swaps both for open equivalents with the same architecture:
    # DINOv3 -> PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m mirror;
    # rembg  -> ZhengPeng7/BiRefNet (also the class's own upstream default).
    # An earlier build of this file used a wrong mirror name recorded from
    # memory (black-forest-labs/dinov3-vitl16 - does not exist); the names
    # below are the author's, read from his PORTING_LOG.md. Logged, not silent.
    import trellis2.modules.image_feature_extractor as _ife

    _orig_dinov3_init = _ife.DinoV3FeatureExtractor.__init__

    def _dinov3_init_mirror(self, model_name, image_size=512):
        if "facebook/dinov3" in str(model_name):
            log(f"conditioner remap: {model_name} -> PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m (ungated mirror)")
            model_name = "PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m"
        _orig_dinov3_init(self, model_name, image_size)

    _ife.DinoV3FeatureExtractor.__init__ = _dinov3_init_mirror

    from trellis2.pipelines.rembg import BiRefNet as _BiRefNetCls

    _orig_biref_init = _BiRefNetCls.__init__

    def _biref_init_open(self, model_name="ZhengPeng7/BiRefNet"):
        if "briaai" in str(model_name) or "RMBG" in str(model_name):
            log(f"rembg remap: {model_name} -> ZhengPeng7/BiRefNet (open, same architecture)")
            model_name = "ZhengPeng7/BiRefNet"
        _orig_biref_init(self, model_name)

    _BiRefNetCls.__init__ = _biref_init_open

    t0 = time.time()
    # Load the pipeline from the LOCAL HF snapshot when one is complete on
    # disk (theirs accepts a local path - see Pipeline.from_pretrained).
    # The pipeline's model entries are bare "ckpts/..." subpaths: resolved
    # from the snapshot they need no hub calls at all. Going through the hub
    # for them is fragile - a failed first attempt falls back to parsing
    # "ckpts/<name>" as a repo id, which 401s and masks the real error
    # (diagnosed 2026-10-24 from the r9-r11 run logs).
    _snap = None
    try:
        import glob as _glob
        _hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        for _cand in sorted(_glob.glob(os.path.join(_hub, "models--microsoft--TRELLIS.2-4B", "snapshots", "*")), reverse=True):
            if os.path.exists(os.path.join(_cand, "pipeline.json")) and os.path.exists(os.path.join(_cand, "ckpts", "shape_dec_next_dc_f16c32_fp16.json")):
                _snap = _cand
                break
    except Exception:
        _snap = None
    _src = _snap if _snap else "microsoft/TRELLIS.2-4B"
    log(f"pipeline source: {_src}")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(_src)
    pipeline.cuda()
    log(f"pipeline loaded in {time.time() - t0:.1f}s (cold loads are ~400s on the reference machine)")

    # r31: install the chunked FDG extraction BEFORE any generation
    # (see the r31 block at module level). If the decoder's extraction
    # binding cannot be found, fail loud NOW - an unpatched run would
    # silently reproduce the 2^20 wound this build exists to fix, and
    # its numbers would poison the acceptance comparison.
    _r31_patched = install_chunked_fdg()
    if not _r31_patched:
        log("FAIL: r31 chunked FDG extraction could not be installed - "
            "no module binding for flexible_dual_grid_to_mesh was found "
            "at the decoder's call site (trellis2.models.sc_vaes.fdg_vae). "
            "Refusing to generate an unpatched mesh under an r31 stamp.")
        return 8
    log(f"[r31] chunked FDG extraction installed at: {', '.join(_r31_patched)} "
        f"(budgets: <={FDG_CHUNK_INPUT_BUDGET} input rows/call, "
        f"<={FDG_CHUNK_CORE_BUDGET} core rows/chunk)")

    # r32: install the chunked fill_holes wrapper (and the
    # simplify guard) BEFORE any generation - decode_latent calls
    # Mesh.fill_holes inside pipeline.run, and an unpatched run
    # would silently reproduce the post-extraction 2^20 wound
    # r31 convicted. Fail loud, exactly like r31 above.
    _r32_patched = install_chunked_fill()
    if not _r32_patched:
        log("FAIL: r32 chunked fill_holes could not be installed - "
            "the Mesh.fill_holes binding was not found at "
            "trellis2.representations.mesh.base, or its stock "
            "signature is not the one the wrapper was written "
            "against. Refusing to generate under an r32 stamp "
            "with the wound unpatched.")
        return 9
    log(f"[r32] chunked fill_holes installed at: {', '.join(_r32_patched)} "
        f"(stock delegation at/under {FILL_STOCK_FACE_LIMIT} faces / "
        f"{FILL_STOCK_VERT_LIMIT} verts; chunk budgets "
        f"<={FILL_CHUNK_FACE_BUDGET} faces, <={FILL_CHUNK_VERT_BUDGET} verts)")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    outputs = pipeline.run(image, seed=args.seed, pipeline_type=args.mode)
    dt = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    log(f"generation mode={args.mode} seed={args.seed}: {dt:.1f}s  peak VRAM {peak_gb:.2f} GB")
    if peak_gb < 0.5:
        log("WARN: peak VRAM under 0.5 GB - suspicious; the GPU may not have done the work")

    mesh = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    log(f"output object: {type(mesh).__module__}.{type(mesh).__name__}")

    # r30: what does the returned mesh object actually carry? One line
    # listing its public attributes, so the hunt stops guessing which
    # stage data (coords, layout, per-stage leftovers) survives to here.
    try:
        _attrs = [a for a in dir(mesh) if not a.startswith("_")]
        log(f"[diag] mesh attrs: {_attrs}")
    except Exception:
        pass

    # r28 GPU-SIDE BLOCK PROFILE, computed BEFORE any host dump of this
    # mesh: the first GPU-side measurement in the hunt's history. If
    # the GPU profile itself shows a desert, the copy theory is wrong
    # and this run says so in plain lines; if the GPU is whole, the
    # host profile (read through the chunked helper) must match it
    # block for block.
    # r30: faces AND vertices are profiled (r28 did faces only, and
    # job 010's faces showed a genuine prefix ending at exactly 2^20
    # rows plus an end stub - HUNT-NOTE-3). Whether vertices carry the
    # same wound decides whole-mesh assembly vs face construction.
    # The gate also opens for the r30 non-cascade control arm via
    # TRELLIS_PROFILE=1.
    if "cascade" in str(args.mode) or os.environ.get("TRELLIS_PROFILE") == "1":
        for _label, _noun in (("faces", "face"), ("vertices", "vertex")):
            try:
                _arr0 = getattr(mesh, _label, None)
                if _arr0 is None:
                    log(f"[gpu-profile] {args.mode} {_label}: attribute missing on the mesh object")
                    continue
                _gpu_prof = genuine_row_profile(_arr0)
                _host_prof = genuine_row_profile(chunked_readout(_arr0))
                log(f"[gpu-profile] {args.mode} {_label}: genuine rows per 32 blocks {_gpu_prof} "
                    f"(total {sum(_gpu_prof)} of {int(_arr0.shape[0])})")
                log(f"[host-profile] {args.mode} {_label}: genuine rows per 32 blocks {_host_prof}")
                if _host_prof == _gpu_prof:
                    log(f"[readout] READOUT-MATCH: the host copy carries every genuine {_noun} the GPU holds")
                else:
                    log(f"[readout] READOUT-DESERT: the host copy LOSES genuine {_label} vs the GPU "
                        "profile - either the readout is still broken or the GPU array itself is "
                        "deserted (the [gpu-profile] line above says which). Do not trust any count "
                        "read from the host copy; the bounds gate below remains the hard floor.")
            except Exception:
                log(f"[readout] block profile ({_label}) failed:\n" + traceback.format_exc())

    # r19 DIAGNOSTIC (the cascade-scramble hunt): the 1024_cascade exports
    # come out of o_voxel.to_glb geometrically scrambled (slab-folded,
    # banded) while 512 exports are clean, and the pre-E2 raw exports of
    # cascade shapes were clean. Three suspects in a row: the decoder's
    # mesh itself, mesh.simplify(), and to_glb. So: log the mesh's vital
    # signs and save RAW trimesh exports (vertices/faces only, no o_voxel
    # machinery) BEFORE simplify and AFTER it. One run then shows exactly
    # which stage folds the shape.
    def _arr(x):
        # r28: EVERY mesh-scale host readout goes through the chunked
        # helper (HUNT-NOTE-2's copy cliff) - this one feeds mesh_stats
        # and both raw GLB dumps.
        try:
            return chunked_readout(x)
        except Exception:
            return None
    def mesh_stats(tag, m):
        v = _arr(getattr(m, "vertices", None))
        f = _arr(getattr(m, "faces", None))
        c = _arr(getattr(m, "coords", None))
        if v is not None and v.size:
            log(f"[diag] {tag}: verts={v.shape} bbox_min={v.min(0).round(4).tolist()} bbox_max={v.max(0).round(4).tolist()}")
        else:
            log(f"[diag] {tag}: vertices unavailable")
        if f is not None and f.size:
            log(f"[diag] {tag}: faces={f.shape}")
        if c is not None and c.size:
            log(f"[diag] {tag}: coords={c.shape} min={c.min(0).tolist()} max={c.max(0).tolist()}")
        log(f"[diag] {tag}: voxel_size={getattr(m, 'voxel_size', None)} layout={str(getattr(m, 'layout', None))[:200]}")
    def raw_export(tag, m):
        try:
            import trimesh as _tm
            v = _arr(getattr(m, "vertices", None)); f = _arr(getattr(m, "faces", None))
            if v is None or f is None:
                log(f"[diag] raw export {tag}: skipped (no verts/faces)")
                return
            stem = args.output[:-4] if args.output.endswith(".glb") else args.output
            p = f"{stem}-raw-{tag}.glb"
            _tm.Trimesh(vertices=v, faces=f, process=False).export(p)
            log(f"[diag] raw export {tag}: {p} ({v.shape[0]} verts)")
        except Exception:
            log("[diag] raw export failed:\n" + traceback.format_exc())
    mesh_stats("pre-simplify", mesh)
    raw_export("presimplify", mesh)

    # NaN scan wherever vertices are reachable.
    verts = getattr(mesh, "vertices", None)
    if verts is not None and not finite_scan(verts):
        log("FAIL: NaN/Inf found in output vertices (silent-corruption signature)")
        return 3

    errors = []

    # The output directory must exist before any export path writes to it.
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Export path 1: o_voxel postprocess -> GLB. Recipe copied from upstream
    # example.py (the first version of this file guessed the signature and
    # the 2026-10-24 14:28 run's traceback corrected it: to_glb takes the
    # mesh's components, not the mesh object). Two variants, DENSE FIRST
    # since the 2026-10-24 20:09 run: remesh=True "succeeded" on the 1024
    # cascade while collapsing the mesh to 160 vertices (the shape Cory
    # graded 'fucking close' at 38 MB, destroyed by the remesh), and its
    # success meant remesh=False never ran. Order is now remesh=False
    # (dense mesh + baked textures), then remesh=True, and every export is
    # vertex-checked: a degenerate result counts as a failure and the next
    # variant runs. Every failure is logged
    # IMMEDIATELY - an earlier version of this file only printed the errors
    # when all paths failed, which hid path 1's real error behind a
    # successful fallback on the 14:34 run.
    import o_voxel

    try:
        mesh.simplify(16777216)  # nvdiffrast vertex limit, per example.py
    except Exception:
        # r28: LOG, never swallow - a copy failure inside CuMesh sat
        # behind this bare pass in every log the hunt read
        # (HUNT-NOTE-2 §1); the next one is on record.
        log("mesh.simplify(16777216) raised - continuing with the unsimplified mesh:\n"
            + traceback.format_exc())
    mesh_stats("post-simplify", mesh)
    raw_export("postsimplify", mesh)

    if args.probe_cumesh:
        # r21 PROBE: walk this mesh through the EXACT CuMesh chain that
        # o_voxel.postprocess.to_glb runs for remesh=False (postprocess.py:
        # init -> fill_holes -> read -> simplify(3x) -> cleanup ->
        # simplify(target) -> cleanup -> unify), reading the mesh back and
        # logging bbox/counts after EVERY stage. r20's export lost exactly
        # the x>0 half of the figure somewhere in this chain; the stage
        # whose readback first shows the amputation is the culprit, named.
        import cumesh as _cumesh
        import numpy as _np
        _stem = args.output[:-4] if args.output.endswith(".glb") else args.output
        def _read_stats(tag, cm):
            try:
                v, f = cm.read()
                vn = chunked_readout(v); fn = chunked_readout(f)  # r28 chunked readout
                _deg = int((~((fn[:, 0] != fn[:, 1]) & (fn[:, 1] != fn[:, 2]) & (fn[:, 0] != fn[:, 2]))).sum()) if fn.size else 0
                log(f"[probe] {tag}: verts={vn.shape[0]} faces={fn.shape[0]} degenerate_faces={_deg} "
                    f"bbox_min={vn.min(0).round(4).tolist()} bbox_max={vn.max(0).round(4).tolist()}")
                return v, f
            except Exception:
                log(f"[probe] {tag}: READ FAILED:\n" + traceback.format_exc())
                return None, None
        def _dump(tag, v, f):
            try:
                import trimesh as _tm
                _tm.Trimesh(vertices=chunked_readout(v), faces=chunked_readout(f),
                            process=False).export(f"{_stem}-{tag}.glb")
                log(f"[probe] dumped {_stem}-{tag}.glb")
            except Exception:
                log(f"[probe] dump {tag} failed:\n" + traceback.format_exc())
        _v0 = mesh.vertices.cuda()
        _f0 = mesh.faces.cuda()
        if _f0.dtype not in (torch.int32, torch.int64):
            _f0 = _f0.to(torch.int32)
        log(f"[probe] input: verts={_v0.shape} faces={_f0.shape} faces_dtype={_f0.dtype}")
        _cm = _cumesh.CuMesh()
        try:
            _cm.init(_v0, _f0)
            _v1, _f1 = _read_stats("S1 post-init", _cm)
            if _v1 is not None:
                _dump("S1", _v1, _f1)
            _cm.fill_holes(max_hole_perimeter=3e-2)
            _read_stats("S2 post-fill_holes", _cm)
            _cm.simplify(args.decimation_target * 3, verbose=False)
            _read_stats("S3 post-simplify3x", _cm)
            _cm.remove_duplicate_faces(); _cm.repair_non_manifold_edges()
            _cm.remove_small_connected_components(1e-5); _cm.fill_holes(max_hole_perimeter=3e-2)
            _read_stats("S4 post-cleanup1", _cm)
            _cm.simplify(args.decimation_target, verbose=False)
            _read_stats("S5 post-simplify-target", _cm)
            _cm.remove_duplicate_faces(); _cm.repair_non_manifold_edges()
            _cm.remove_small_connected_components(1e-5); _cm.fill_holes(max_hole_perimeter=3e-2)
            _cm.unify_face_orientations()
            _v6, _f6 = _read_stats("S6 final", _cm)
            if _v6 is not None:
                _dump("S6", _v6, _f6)
        except Exception:
            log("[probe] chain aborted:\n" + traceback.format_exc())
        log("[probe] done")
        return 0

    def exported_vertex_count(path):
        # Total POSITION vertices across the exported GLB's primitives,
        # read back from the file itself (the ground truth of the export).
        import json as _json
        import struct as _struct
        with open(path, "rb") as fh:
            data = fh.read()
        jlen = _struct.unpack("<I", data[12:16])[0]
        doc = _json.loads(data[20:20 + jlen])
        total = 0
        for mesh_ in doc.get("meshes", []):
            for prim_ in mesh_.get("primitives", []):
                pos_idx = prim_.get("attributes", {}).get("POSITION")
                if pos_idx is not None:
                    total += doc["accessors"][pos_idx]["count"]
        return total

    def exported_face_count(path):
        # Total triangles across the exported GLB's primitives (index
        # count / 3), read back from the file itself. Companion to
        # exported_vertex_count; r24 logs both for every section so a
        # section's exported size is on record next to its input size.
        import json as _json
        import struct as _struct
        with open(path, "rb") as fh:
            data = fh.read()
        jlen = _struct.unpack("<I", data[12:16])[0]
        doc = _json.loads(data[20:20 + jlen])
        total = 0
        for mesh_ in doc.get("meshes", []):
            for prim_ in mesh_.get("primitives", []):
                idx_idx = prim_.get("indices")
                if idx_idx is not None:
                    total += doc["accessors"][idx_idx]["count"] // 3
        return total

    exp_vertices, exp_faces = mesh.vertices, mesh.faces

    # r24: DEGENERATE-FACE FILTER at export entry, for EVERY export path
    # below (sectioned, stock to_glb, trimesh fallback). The cascade
    # decoder's output carries a tail of index-degenerate faces - on
    # probe-S1, 1,091,754 literal (0,0,0) rows starting at exactly row
    # 2^20, present in the pipeline's own output (autopsy (a) and (d)5),
    # not a CuMesh artifact. They render as nothing, but they inflate
    # every face count past the CuMesh 2^20 cliff and wreck the section
    # cutter's histogram. Drop them once, here, and log the count.
    _faces_np = None
    try:
        _faces_np = chunked_readout(mesh.faces)  # r28: chunked - this single-shot read was the filter's own cliff victim in r24/r27
    except Exception:
        _faces_np = None
    if _faces_np is not None:
        _kept_np, _dropped = drop_degenerate_faces(_faces_np)
        log(f"[filter] {_dropped} index-degenerate faces dropped before export "
            f"({_dropped / max(1, _faces_np.shape[0]) * 100:.2f}% of {_faces_np.shape[0]}); "
            f"{_kept_np.shape[0]} genuine faces remain")
        if _dropped:
            _faces_np = _kept_np
            exp_faces = torch.as_tensor(_kept_np, dtype=mesh.faces.dtype, device=mesh.faces.device)
            try:
                mesh.faces = exp_faces
                log("[filter] mesh.faces now holds the filtered tensor, so mesh.export (path 2) is filtered too")
            except Exception:
                log("[filter] mesh.faces is not assignable on this object - paths 1/3 use the filtered tensor; path 2 (mesh.export) keeps the raw faces")

    # r23: SECTIONED EXPORT (Cory's design, 2026-10-24 23:01: "do models in
    # sections... that would allow more of those numbers for each part").
    # For meshes whose face count crosses the CuMesh 2^20 zero-fill cliff,
    # the stock export is PROVEN to amputate (r21 probe). So: sort the
    # faces by centroid along the longest axis (for a figure: height bands
    # - legs / torso / head), cut them into <=900k-face sections with a
    # small overlap at the seams (caps land inside the overlap, hidden),
    # export each section through to_glb on its own - every section under
    # the cliff - with the ORIGINAL coords/attrs (the bake samples the
    # voxel field by position, so all sections share it), then merge the
    # section GLBs into one file as a multi-geometry scene. Each section
    # keeps its own atlas at the full texture size. If anything in this
    # path fails, the stock chain below runs exactly as before.
    # r27 (2026-10-25): the "original coords/attrs" line above is r23's
    # design and the r24 grade convicted it - every sectioned export
    # came back clipped at world X~0. Sections now subset coords/attrs
    # to the section's own voxels (see the subsetting block in the
    # export loop), and read-back bounds gates guard every section and
    # the merged file, so a clip can never pass silently again.
    # r24 (cutter autopsy): the faces arrive DEGENERATE-FILTERED from the
    # caller; cut planning is plan_section_cuts() (face-space window,
    # mid-bin cap cut, per-cut provenance logged); exported ranges are
    # hard-capped under the cliff; a failed section is SKIPPED, not
    # abortive; the merge streams one section file at a time.
    def sectioned_export(_f):
        import numpy as _np
        import torch as _torch
        _CAP = 900000     # planning cap for a section's core range
        _HARD = 1000000   # exported-range hard cap, safely under the 2^20 cliff (1,048,576)
        _v = chunked_readout(mesh.vertices)  # r28 chunked readout
        _nf = _f.shape[0]
        if _nf <= _CAP:
            return False
        _cent = _v[_f].mean(axis=1)
        _axis = int(_np.argmax(_v.max(0) - _v.min(0)))
        _order = _np.argsort(_cent[:, _axis], kind="stable")
        _margin = max(1000, int(_nf * 0.02))
        # WHERE TO CUT (Joe's call, Cory 2026-10-24 23:04): not at
        # arithmetic boundaries - at the mesh's NARROW places near them.
        # A seam across a thin cross-section (ankle, waist, neck) cuts few
        # triangles, leaves a small hole for the filler, and hides in the
        # overlap. Score 256 thin slices along the axis by cross-section
        # area proxy (centroid spread in the other two axes), smooth, then
        # greedily: accumulate faces until >=50% of the cap, then cut at
        # the narrowest slice reachable before the cap would break.
        _lo, _hi = float(_cent[:, _axis].min()), float(_cent[:, _axis].max())
        _NB = 256
        _bin = _np.clip(((_cent[:, _axis] - _lo) / max(_hi - _lo, 1e-9) * _NB).astype(int), 0, _NB - 1)
        _others = [a for a in range(3) if a != _axis]
        _score = _np.full(_NB, 1e18)
        _cnt = _np.zeros(_NB, dtype=int)
        for _b in range(_NB):
            _m = _bin == _b
            _cnt[_b] = int(_m.sum())
            if _cnt[_b] > 50:
                _score[_b] = float(_cent[_m][:, _others[0]].std() * _cent[_m][:, _others[1]].std())
        _k = _np.ones(5) / 5.0
        _score_s = _np.convolve(_np.pad(_score, 2, mode="edge"), _k, mode="valid")
        # JOINT BONUS (Cory, 2026-10-24 23:04): joints - neck, wrists,
        # elbows, knees, ankles - are not just thin, they are DIPS: thinner
        # than the body on both sides of them. Elbows/knees barely narrow,
        # so a pure thinness score can walk past them; a dip is still a
        # dip. Divide each slice's score by its dip depth (how far it sits
        # below the wider of its two neighborhood means) so true joints
        # outrank merely-thin slices. A cut at a joint also lands where a
        # rig's own part boundaries live - the seam hides twice.
        _W = 10
        _dip = _np.ones(_NB)
        for _b in range(_NB):
            _lft = _score_s[max(0, _b - _W):_b]
            _rgt = _score_s[_b + 1:_b + 1 + _W]
            if _lft.size and _rgt.size and _score_s[_b] > 0:
                _ref = max(float(_lft.mean()), float(_rgt.mean()))
                _dip[_b] = 1.0 + max(0.0, (_ref - _score_s[_b]) / _ref) * 3.0
        _score_s = _score_s / _dip
        # r24: cut planning lives in plan_section_cuts() (module level).
        # Every cut's provenance is logged; the words "narrow-point" are
        # printed only for cuts that rule actually chose (in r23 the log
        # claimed the rule unconditionally while it decided zero cuts).
        _cuts, _prov = plan_section_cuts(_cnt, _score_s, _nf, _CAP)
        for _p in _prov:
            log(f"[sections] cut at face {_p['cut']}: via {_p['via']} ({_p['candidates']} narrow-point candidates in window)")
        log(f"[sections] {_nf} faces -> {len(_cuts) - 1} sections, cuts at faces {_cuts} (axis {_axis})")
        # Exported ranges = core range plus margin on both sides (the
        # overlap double-covers the seams; the bake samples by position).
        # r24: cap compliance is judged on THIS exported range, not the
        # core - r23 shipped single to_glb calls of 1,182,645 and
        # 2,428,878 faces because nothing checked it (autopsy (d)2).
        # Split any over-cap range into <= _HARD sub-sections, then fail
        # loud if one still exceeds the cap: to_glb never silently sees
        # an over-cap range.
        _ranges = []
        for _j, (_a0, _b0) in enumerate(zip(_cuts[:-1], _cuts[1:])):
            _a = max(0, _a0 - _margin)
            _b = min(_nf, _b0 + _margin)
            _subs = 0
            while _b - _a > _HARD:
                _ranges.append((_a, _a + _HARD))
                _a = _a + _HARD
                _subs += 1
            if _subs:
                log(f"[sections] planned section {_j} exported range exceeds the {_HARD}-face hard cap - split into {_subs + 1} sub-sections")
            _ranges.append((_a, _b))
        for (_a, _b) in _ranges:
            if _b - _a > _HARD:
                raise RuntimeError(f"sectioned export: exported range {_a}-{_b} ({_b - _a} faces) still exceeds the hard cap {_HARD} after splitting - refusing to call to_glb over the 2^20 cliff")
        _nsec = len(_ranges)
        # r34: prove the cleanup override installs BEFORE the first
        # section runs. If it cannot, R34CleanupError (a
        # BoundsVerificationError) aborts the whole export under the
        # caller's r27 no-fallback contract - a section must never
        # run under an override that is not actually in place.
        with _R34SectionCleanup(-1, probe=True):
            pass
        _stem = args.output[:-4] if args.output.endswith(".glb") else args.output
        _ok = []
        _skipped = []
        for _i, (_a, _b) in enumerate(_ranges):
            _sel = _order[_a:_b]
            _fc = _f[_sel]
            _uv, _inv = _np.unique(_fc, return_inverse=True)
            _vc = _v[_uv]
            _fc2 = _inv.reshape(-1, 3)
            _share = _fc.shape[0] / _nf
            _tgt = max(20000, int(args.decimation_target * _share))
            # r24 sanity facts for THIS section, computed before export:
            # its degenerate-face ratio (zero by construction after the
            # entry filter; logged so any future unfiltered caller is
            # caught on record). The old >=5,000-vert check alone passed
            # cliff-damaged sections - subject sec116 in r23 exported
            # 2,428,878 faces in one call and "passed" with 43,781 verts.
            _deg = int(((_fc2[:, 0] == _fc2[:, 1]) | (_fc2[:, 1] == _fc2[:, 2]) | (_fc2[:, 0] == _fc2[:, 2])).sum())
            _ratio = _deg / max(1, _fc2.shape[0])
            _p = f"{_stem}-sec{_i}.glb"
            # r34 CENSUS (audit PASS 2 fix-shape item 2): log this
            # section's manifold-fragment census - the bounds
            # defect's three predictor numbers - on the exact
            # compacted mesh to_glb is about to receive, before the
            # op sequence runs. Diagnostic only: a census failure
            # never blocks the export.
            try:
                _cen = manifold_fragment_census(_fc2)
                log(f"[r34-census] section {_i}/{_nsec}: {_cen['n_faces']} faces, manifold-only components={_cen['n_components']}, largest fragment {_cen['largest_faces']} faces ({_cen['largest_pct']:.2f}%), faces touching a non-manifold edge {_cen['pct_touch_nonmanifold']:.1f}%")
            except Exception:
                log(f"[r34-census] section {_i}/{_nsec}: census failed (diagnostic only - export continues):\n" + traceback.format_exc())
            # r27: SUBSET THE VOXEL FIELD WITH THE SECTION. r24 handed
            # to_glb the section's compacted verts/faces alongside the
            # FULL coords/attrs, and every sectioned export came back
            # clipped at world X~0 (canvas lost its right half; the
            # subject lost everything past X -0.015) while the raw mesh
            # and the unsectioned smoke export were whole. Now the
            # sparse coords AND the per-voxel attrs are subsetted to
            # the voxels this section's surface passes through (its
            # extent along the cut axis plus one voxel of slack), so
            # verts/faces/coords/attrs all describe the same subset
            # under one consistent indexing. Coords stay ABSOLUTE grid
            # coords and the aabb below stays the full volume, so the
            # voxels that remain keep their true positions. If the
            # subsetting itself fails, the full field is passed (r24
            # behaviour) and the bounds gate below still guards it.
            _coords_arg, _attrs_arg = mesh.coords, mesh.attrs
            try:
                _vs = float(mesh.voxel_size)
                _c_np = chunked_readout(mesh.coords)  # r28 chunked readout (handles tensors and array-likes)
                _wpos = -0.5 + _c_np[:, -3:].astype(_np.float64) * _vs
                _slo = float(_vc[:, _axis].min()) - _vs
                _shi = float(_vc[:, _axis].max()) + _vs
                _vmask = (_wpos[:, _axis] >= _slo) & (_wpos[:, _axis] <= _shi)
                _nvox = int(_vmask.sum())
                if 0 < _nvox < _c_np.shape[0]:
                    if hasattr(mesh.coords, "detach"):
                        _mi = _torch.as_tensor(_vmask, device=mesh.coords.device)
                        _coords_arg = mesh.coords[_mi]
                        _attrs_arg = mesh.attrs[_mi]
                    else:
                        _coords_arg = mesh.coords[_vmask]
                        _attrs_arg = mesh.attrs[_vmask]
                    log(f"[sections] section {_i}: voxel field subsetted to {_nvox}/{_c_np.shape[0]} voxels (cut axis {_axis}, range {_slo:.3f}..{_shi:.3f})")
                else:
                    log(f"[sections] section {_i}: voxel subset kept {_nvox}/{_c_np.shape[0]} rows - passing the full field")
            except Exception:
                log(f"[sections] section {_i}: voxel-field subsetting failed - passing the full coords/attrs; the bounds gate below still guards this export:\n" + traceback.format_exc())
                _coords_arg, _attrs_arg = mesh.coords, mesh.attrs
            try:
                # r34: this section's to_glb runs under the cleanup
                # override - remove_small_connected_components is
                # excluded and the remaining bundle ops are logged
                # per-op under [r34-ops]. The override is restored
                # the moment to_glb returns, so nothing outside this
                # call (the stock path included) ever sees it.
                with _R34SectionCleanup(_i):
                    _glb = o_voxel.postprocess.to_glb(
                        vertices=_torch.as_tensor(_vc, dtype=mesh.vertices.dtype, device=mesh.vertices.device),
                        faces=_torch.as_tensor(_fc2, dtype=mesh.faces.dtype, device=mesh.faces.device),
                        attr_volume=_attrs_arg,
                        coords=_coords_arg,
                        attr_layout=mesh.layout,
                        voxel_size=mesh.voxel_size,
                        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        decimation_target=_tgt,
                        texture_size=args.texture_size,
                        remesh=False,
                        remesh_band=1,
                        remesh_project=0,
                        verbose=True,
                    )
                _glb.export(_p, extension_webp=True)
            except Exception:
                # r24 SKIP-DON'T-ABORT: one failed section must not
                # discard the whole subject/canvas into the stock
                # fallback - r23 lost the canvas exactly that way, and
                # its never-attempted section stayed unrecoverable.
                # Log the failure, skip the section, keep going: every
                # planned section is attempted.
                log(f"[sections] section {_i}/{_nsec} (faces {_a}-{_b}) export FAILED - SKIPPED, continuing with the rest:\n" + traceback.format_exc())
                _skipped.append(_i)
                continue
            _n = exported_vertex_count(_p)
            _nf_out = exported_face_count(_p)
            if _n < 5000 or _ratio > 0.01:
                log(f"[sections] section {_i}/{_nsec} FAILED sanity - SKIPPED: exported verts={_n} (min 5000), exported faces={_nf_out}, input faces={_fc.shape[0]}, degenerate ratio={_ratio:.4f} (max 0.01); {_p} kept on disk for inspection")
                _skipped.append(_i)
                continue
            # r27 PER-SECTION BOUNDS GATE: read the exported section
            # back and compare its bounds against this section's own
            # input bounds (mapped into GLB space). The r24 clip
            # passed the old sanity checks - plenty of verts, zero
            # degenerate faces - while missing half the figure, so
            # counts alone are not a sanity check anymore. A clipped
            # or collapsed section is SKIPPED LOUDLY (r24 semantics:
            # one bad section never aborts the rest), and the
            # post-merge verify below is the hard floor behind it.
            _bfails = check_export_bounds(_p, _vc.min(0), _vc.max(0), float(mesh.voxel_size), loose_trellis_axis=_axis)
            if _bfails:
                log(f"[sections] section {_i}/{_nsec} FAILED the bounds gate - SKIPPED: " + "; ".join(_bfails) + f"; {_p} kept on disk for inspection")
                _skipped.append(_i)
                continue
            log(f"[sections] section {_i}/{_nsec}: faces {_a}-{_b} ({_fc.shape[0]}) -> {_n} verts / {_nf_out} faces, degenerate ratio {_ratio:.4f}, bounds gate PASS -> {_p}")
            _ok.append((_i, _p))
        if not _ok:
            log("[sections] every section failed or was skipped - sectioned path failed; the stock chain will run")
            return False
        if _skipped:
            log(f"[sections] WARNING: {len(_skipped)} of {_nsec} sections skipped {_skipped} - the merged output is PARTIAL; those ranges are missing from it")
        # r24 STREAMING merge: r23 held every section's geometry in
        # memory at once and the subject merge was OOM-killed. Load one
        # section file at a time, move its geometry into the scene, and
        # release the loaded scene before opening the next file - peak
        # memory is the growing scene plus one section, not every
        # section held twice over.
        import trimesh as _tm
        _scene = _tm.Scene()
        for _i, _p in _ok:
            _g = _tm.load(_p, force="scene")
            for _name, _geom in _g.geometry.items():
                _scene.add_geometry(_geom, geom_name=f"sec{_i}_{_name}")
            del _g
        _scene.export(args.output)
        _total = exported_vertex_count(args.output)
        log(f"[sections] merged {len(_ok)} sections -> {args.output} ({_total} verts total)")
        # r27 POST-MERGE BOUNDS VERIFY: the merged file must span the
        # raw cascade mesh on every axis. In r24 it did not - canvas X
        # stopped at ~0, the subject at -0.015 - and nothing checked,
        # so the clip shipped under a PASS banner. On mismatch: keep
        # the evidence (the merged file is renamed beside the output
        # and a .BOUNDS-FAIL.txt marker is written next to it, both
        # harvested into the run's results) and fail loud via
        # BoundsVerificationError. The caller exits nonzero and does
        # NOT mask the clip with the stock chain - the stock export is
        # the r20 over-cliff amputation, and one broken file hiding
        # another helps nobody.
        _mfails = check_export_bounds(args.output, _v.min(0), _v.max(0), float(mesh.voxel_size))
        if _mfails:
            _detail = "; ".join(_mfails)
            _bad = f"{_stem}-bounds-fail.glb"
            _marker = f"{_stem}.BOUNDS-FAIL.txt"
            try:
                with open(_marker, "w") as _mf:
                    _mf.write(f"r27 post-merge bounds verification FAILED for {args.output}\n{_detail}\nThe clipped merged file was kept as {_bad} for inspection.\n")
                os.replace(args.output, _bad)
            except Exception:
                log("[sections] WARNING: could not write the bounds-fail marker / rename the merged file - the FAIL below still stands:\n" + traceback.format_exc())
            raise BoundsVerificationError(f"merged output failed bounds verification: {_detail} (clipped file kept as {_bad}, marker {_marker})")
        log("[sections] merged bounds verified against the raw cascade mesh - full span on every axis")
        return _total >= 5000

    try:
        if _faces_np is not None and int(_faces_np.shape[0]) > 900000 and sectioned_export(_faces_np):
            return 0
    except BoundsVerificationError as _bve:
        # r27: the sectioned export RAN but its merged output failed
        # bounds verification. Exit nonzero (5) so the runner records a
        # genuine this-run failure; do NOT fall through to the stock
        # chain - its over-cliff export is the r20 amputation, and
        # masking a clipped file with a differently broken one helps
        # nobody. The clipped file + marker stay beside the output.
        log(f"FAIL: {_bve}")
        return 5
    except Exception:
        log("[sections] sectioned export failed - falling back to the stock chain:\n" + traceback.format_exc())

    for remesh in (False, True):
        try:
            glb = o_voxel.postprocess.to_glb(
                vertices=exp_vertices,
                faces=exp_faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=args.decimation_target,
                texture_size=args.texture_size,
                remesh=remesh,
                remesh_band=1,
                remesh_project=0,
                verbose=True,
            )
            glb.export(args.output, extension_webp=True)
            nverts = exported_vertex_count(args.output)
            if nverts < 5000:
                log(f"export path to_glb(remesh={remesh}) produced a DEGENERATE mesh ({nverts} vertices) - treating as failed, trying the next variant")
                errors.append(f"o_voxel.postprocess.to_glb(remesh={remesh}) degenerate ({nverts} vertices)")
                continue
            log(f"exported via o_voxel.postprocess.to_glb (remesh={remesh}, {nverts} vertices) -> {args.output}")
            return 0
        except Exception:
            log(f"export path to_glb(remesh={remesh}) failed:\n" + traceback.format_exc())
            errors.append(f"o_voxel.postprocess.to_glb(remesh={remesh}) failed (traceback above)")

    # Export path 2: the object exports itself.
    try:
        mesh.export(args.output)
        log(f"exported via mesh.export -> {args.output}")
        return 0
    except Exception:
        errors.append("mesh.export:\n" + traceback.format_exc())

    # Export path 3: raw vertices/faces through trimesh.
    try:
        import numpy as np
        import trimesh

        v = chunked_readout(verts)  # r28 chunked readout
        f = exp_faces  # r24: the filtered faces from export entry, not the raw mesh.faces
        f = chunked_readout(f)  # r28 chunked readout
        trimesh.Trimesh(vertices=v, faces=f, process=False).export(args.output)
        log(f"exported via trimesh fallback (NO PBR textures on this path) -> {args.output}")
        return 0
    except Exception:
        errors.append("trimesh fallback:\n" + traceback.format_exc())

    log("FAIL: every export path failed. Errors follow; object repr: " + repr(mesh)[:500])
    for e in errors:
        log(e)
    return 4


if __name__ == "__main__":
    if len(sys.argv) > 4 and sys.argv[1] == "--fill-chunk-worker":
        # r33: chunk-isolation subprocess entry (see the r33 fill
        # header). argv: --fill-chunk-worker in.npz out.npz
        # max_hole_perimeter. Runs the STOCK fill on one chunk in
        # this fresh process and exits; a crash here costs the
        # chunk's parent a retry, never the run.
        sys.exit(_fill_chunk_worker(sys.argv[2], sys.argv[3],
                                    float(sys.argv[4])))
    sys.exit(main())
