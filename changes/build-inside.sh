#!/usr/bin/env bash
# build-inside.sh - runs INSIDE the dedicated WSL2 Ubuntu-24.04 distro, as root.
# Called by TRELLIS-AMD-RUN.ps1. Arg 1 = package folder as a WSL path (/mnt/x/...).
#
# Exit codes (the .ps1 turns these into the split verdict):
#   0  all phases passed        40 native-extension build/import failed
#   10 GPU not visible to ROCm  60 smoke generation failed
#   20 ROCm install failed      70 canvas generation failed
#   30 torch install/check failed
#
# Every phase is marker-gated: re-running skips finished phases and picks up
# where it stopped. Safe to Ctrl-C / close the window at any point.

# Build stamp: printed into every log so the log itself proves which copy of
# this file ran (stale-copy swaps cost three diagnosis cycles on 2026-10-24).
# r26: adds the optional local HF token hook (hf-token.local, below).
# (r25 was the runner .ps1 WSL-reset update - a different file.)
# r27: clears stale verdict files at build start (below) and adds the
# K2 thrust/tuple.h include fix; generate.py carries the matching
# export-clip fix set (voxel-field subsetting + bounds gates).
# r28: generate.py reads every mesh-scale GPU array back in <=65,536-row
# chunks (HUNT-NOTE-2: the 2^20 "clip" was a single-shot GPU->host copy
# cliff, not a face writer) and proves it every run - [readout-ab] at
# smoke, [gpu-profile]/[host-profile] on the cascade meshes - while the
# bounds gates stay as the proof instrument. TRELLIS-AMD-RUN.bat: the
# closing pause now fires only at a real console (under the PC Worker,
# stdout is piped and the old pause blocked forever after the .ps1 had
# already exited cleanly).
# r35: THE R1 SWAP - generate.py CHANGES in this drop (its first
# change since r33): exactly one behavioral change, the
# fill-chunk worker command pointer in _run_fill_subprocess.
# When build-inside.sh exports R1_FILL_WORKER - only after the
# new Phase R1 gate stage below compiles, probes, calibrates
# and equivalence-gates the R1 HIP fill worker on this stack -
# chunk fills run on our own AMD-native worker (inside/r1-src,
# built on-PC at gate time) instead of the incumbent CuMesh
# subprocess; with the variable unset the command is
# byte-for-byte r33's. The gate stage (inside/r1-stage.sh) sits
# after the post-prep census gate and before Phase H: live WSL
# limits logged under [r35-r1] (recorded, never gated - the
# .wslconfig behind them is written before WSL boots, Cory's
# pre-step, not drop business) -> compile (hipcc, gfx1201) ->
# load probe -> calibration (incumbent determinism, the crafted
# B>0/M=0 code-9 shape, the threshold-edge loop) -> the frozen
# corpus equivalence sample (2,355 subject chunks,
# sample-subject.json, predeclared in this drop). Any gate
# failure exits 65 and generation does NOT run; the incumbent
# is never deleted and the previous generate.py stays beside
# the new one as generate.py.prev (kept by the update
# mechanism). The census/deps ladder is byte-identical to
# r33.7 - the closure-heal rides along unchanged.
# r33.7: deps-heal rung v2 - WHOLE-CLOSURE step - generate.py is
# still byte-identical to r33 and is NOT part of this update;
# get-pip.py stays in place from the r33.3 drop. Job 036 split its
# verdict: the r33.6 stamp applied, the rebuild sentinel-skipped
# (fresh venv intact, nothing renamed or deleted), Phase C was
# clean, the census read the predicted 68/58/3/7 - and the deps-
# heal rung HEALED 2 of 3: open3d via the dash install, requests
# via the urllib3 that arrived with it (both heals PERSIST in the
# venv). rembg alone stayed broken: attempt 1 installed
# onnxruntime, attempt 2 pymatting, and the next re-probe named
# skimage - rembg's closure is a CHAIN (onnxruntime -> pymatting
# -> scikit-image -> ...) and the 2-attempt cap ran out mid-
# chain; the guard exit-30'd correctly. The design lesson:
# hand-chasing named modules reinvents pip's resolver one link
# at a time - the resolver IS the closure engine. r33.7 adds ONE
# whole-closure step to the dependency ladder in BOTH rungs
# (Phase C census + post-prep gate), reached FAST: when a
# package's re-probe names a SECOND distinct missing module
# (the chain signature), or the package sits in the extras-gated
# table (rembg -> [cpu]), the chase stops and a single plain
# pip install "pkg[extras]" lets the resolver install the whole
# declared closure (plain, NOT force-reinstall: the package's
# own files are whole, its closure is incomplete; plain install
# fills missing members and leaves satisfied/pinned
# distributions alone). Pin safety: the closure pass's
# "Installing collected packages" line is logged under
# [r33.7 closure-heal] and the torch probe re-runs after every
# closure pass - a failed probe exits loud naming the closure
# pass as the suspect, so a numpy/torch drift is visible in the
# log, never silent. The rest of the ladder is unchanged:
# file-damage --no-deps heal first, named-module installs as
# the cheap dep-class attempts (cap 2), whole-closure for
# chains, force-reinstall-with-deps as the last resort,
# still-broken exits loud.
# r33.6: census DEPS-HEAL rung + post-prep census gate - generate.py
# is still byte-identical to r33 and is NOT part of this update;
# get-pip.py stays in place from the r33.3 drop. Job 035 split its
# verdict: the r33.5 REBUILD PASSED (sentinel-skip proved 034 fired
# it; old venv preserved as venv-broken-20261026; fresh venv probes
# clean; GPU gate passed) - then the census on the fresh venv found
# 68 probed, 58 pass, 3 FAIL, 7 excluded, and the run exit-30'd.
# The 3 FAILs are a NEW failure class - missing DEPENDENCIES, not
# damaged files: open3d imports die on absent dash, requests on
# absent urllib3, rembg on its absent onnxruntime backend. The
# r33.4 heal force-reinstalls --no-deps, which restores a package's
# OWN files and by construction can never ADD a distribution that
# is not installed; the re-probe failed 0/3 with the identical
# errors and the guard correctly refused to continue. Where did
# closure-less packages come from on a "fresh" venv? Not Phase C:
# its fresh-install path installs ONLY the pinned torch pair plus
# trimesh/huggingface_hub/pillow, all WITH dependency resolution,
# and none of the three is on any install line in this script.
# The census sits in Phase C - BEFORE the phases that populate
# the broad set. On 034's run it measured the small fresh venv
# (whole) and passed; 034 then ran on into Phase E, whose egore
# installer (an external script, run with the venv active) lands
# the broad TRELLIS.2/GGUF set - and that set arrived WITHOUT its
# full closure: dash and urllib3 ARE hard requirements of
# open3d/requests, so their absence proves dependency resolution
# was off for those installs (the standard --no-deps protection
# for a hand-pinned torch stack), while onnxruntime is not a base
# requirement of rembg AT ALL - it ships only in rembg's [cpu]/
# [gpu] extras, so no plain rembg install ever brings it. Job 035
# inherited that venv and its census was the first measurement
# after the installers ran. DIAGNOSIS-035.md has the full chain
# with log-line evidence. This drop answers the mechanism, twice:
# (1) the census heal learns the second failure class - a package
# still failing after the --no-deps heal gets up to 2 dependency-
# class attempts (install the distribution its import error
# names, WITH deps; if the named distribution is installed but
# its module is damaged, force-reinstall THAT at its version; if
# no single module is named, force-reinstall the package itself
# WITH deps so pip resolves its whole declared closure), then a
# re-probe; anything still broken exits 30 loud, unchanged;
# (2) the ROOT fix: a second census gate now stands AFTER Phase H
# prep - after EVERY installer in the run has run - so closure
# holes an installer creates are caught and healed in the SAME
# run instead of ambushing the next one. Phase C's own install
# line is untouched: 035 proved its products whole.
# r33.5: full venv rebuild (one-shot rename, nothing deleted) +
# census exclusion for the Phase E source builds - generate.py
# is still byte-identical to r33 and is NOT part of this update;
# get-pip.py stays in place from the r33.3 drop. Job 033 ran the
# r33.4 census and measured the sweep's whole venv damage in one
# pass: 68 distributions probed, 53 pass, 12 FAIL, 3 excluded
# (pip/torch/torchvision); the heal force-reinstalled 8 clean
# (certifi, click, httpcore, httpx, httpx2, idna, kornia,
# transformers) and 4 stayed broken - nvdiffrast==0.4.0 (a
# source build for this AMD port; the index answers "from
# versions: none"), open3d==0.20.0 (its import dies on the
# damaged dependency dash), requests==2.34.2 (dies on urllib3),
# rembg==2.0.86 (onnxruntime backend missing). The lesson: the
# damage also lives in submodules and dependencies, beneath
# passing top-level probes - per-package patching has hit its
# limit, and a fresh venv is the known-whole answer. Under
# Cory's standing A-to-Z charge (reinforced 2026-10-26 00:43:
# keep going, no permission asks) this drop rebuilds it: a
# one-shot block before Phase C renames the damaged venv to
# venv-broken-20261026 (kept on disk, never deleted) and moves
# aside exactly the markers gating venv-resident artifacts -
# torch.done, ext.done, painter.done, painter.failed (Phase A
# rocm.done, Phase D repos.done and Phase F models.done are NOT
# reset: ROCm is system, the source trees live under $ROOT, and
# the model weights live in $ROOT/hf) - then the script's normal
# fresh-install Phase C path runs and every r33.1-r33.4
# integrity check measures the NEW venv. A sentinel at
# $ROOT/.r33.5-rebuild-done (outside the venv, outside $MARK)
# makes the block one-shot: a re-run never renames the fresh
# venv. The census exclusion list also grows by the Phase E set
# - nvdiffrast, cumesh, flex-gemm, o-voxel are built from source
# by Phase E and import-gated there; on a fresh venv they are
# legitimately absent until Phase E runs, and pip-fetching them
# is wrong (033 proved it), so the census now excludes them by
# name with that reason logged. This supersedes the r33.3
# header's "a full venv rebuild is Cory's decision" framing -
# the standing charge decides, and nothing here asks.
# r33.4: venv-wide census heal + Phase H prep probe-first -
# generate.py is still byte-identical to r33 and is NOT part of
# this update; get-pip.py already shipped inside the r33.3 drop
# and stays in place on the PC (it is not re-shipped here).
# Job 032 split its verdict: the r33.3 REPAIR PASSED end to end
# (get-pip fixed pip, the pinned torch pair reinstalled clean,
# probes green, the GPU gate passed) - and then the run died in
# Phase H prep on the NEXT sweep casualty, the standalone idna
# package: idna/core.py deleted, distribution metadata intact,
# so the prep's dependency chase ran plain `pip install idna`
# 15 times, got "already satisfied" 15 times, and exited 60.
# Jobs 029-032 each surfaced exactly one more casualty because
# the sweep's damage list was never measured - only stumbled
# into, one job at a time. r33.4 measures it. Phase C gains a
# census: every installed distribution is import-probed in a
# subprocess per package (torch/torchvision and pip excluded -
# their own repair branches own them; packages whose files live
# outside the venv are probed but never auto-healed); each
# failing package is force-reinstalled at its installed
# version (--no-deps) and re-probed; and ANY package still
# failing exits 30 loud with the full list - the run does not
# proceed on a known-broken venv. The census totals are the
# first true count of the sweep's venv-wide damage. Phase H
# prep gets the same medicine: its dependency top-up and its
# chase are now probe-first (skip what imports),
# force-reinstall for broken-but-installed packages (never a
# plain install), and the chase caps repairs at 2 attempts per
# module before exiting loud naming the package.
# r33.3: get-pip rung - generate.py is still byte-identical to r33
# and is NOT part of this update; the r33 acceptance question is
# unchanged. Job 031 (the r33.2 run) failed in 13.4 s and its log
# proved the mechanism r33.2 missed: ensurepip ran and exited 0,
# but printed "Requirement already satisfied: pip in
# ./venv/lib/python3.12/site-packages (26.2.2)" - the venv's pip
# is NEWER than the wheel ensurepip bundles, so ensurepip compared
# versions, declared pip satisfied, and never replaced the
# sweep-damaged files. pip._vendor.idna.core stayed missing; the
# verify guard exited 30 before any reinstall, exactly as
# designed, with pip still broken. So this build adds ONE rung,
# Cory-approved 2026-10-26: when either post-ensurepip pip check
# fails, the branch runs get-pip.py (PyPA's official bootstrap,
# which embeds a complete pip and lays it down fresh - no version
# comparison, damaged files overwritten), then re-runs BOTH
# checks. The script ships inside this drop (payload target
# inside/get-pip.py, hash-verified by the launcher's update
# channel like every payload file); the PC downloads nothing at
# repair time beyond the torch wheels. If get-pip also leaves pip
# broken, the branch exits 30 with the ladder declared exhausted
# - a full venv rebuild is Cory's decision, never improvised
# here. See DIAGNOSIS-031.md.
# r33.2: pip bootstrap rung + extended import checks - generate.py is
# still byte-identical to r33 and is NOT part of this update.
# Job 030 (the r33.1 repair run) proved the r33.1 design and found
# the next layer of the same damage: the probe failed all four
# modules naming torch.masked.maskedtensor.core, the force-
# reinstall fired - and pip itself died before it could run:
# "ModuleNotFoundError: No module named 'pip._vendor.idna.core'".
# The sweep took pip's vendored idna core.py too, and a pip-based
# repair cannot start until pip runs. So this build adds ONE rung
# inside the repair branch, before the reinstall: pip is
# bootstrapped from ensurepip (python3 -m ensurepip --upgrade
# --default-pip) and verified (pip --version, plus the exact
# vendored module job 030 died on) - any failure exits 30 BEFORE
# the reinstall is attempted. ensurepip's bundled pip wheel lives
# in the SYSTEM stdlib (/usr/lib/python3.12/ensurepip/_bundled/),
# a tree the sweep never entered, and is a single .whl archive
# whose name cannot match 'core*' anyway - see
# DIAGNOSIS-ADDENDUM.md for the full reasoning. Second addition,
# same root cause (the sweep's damage list cannot be counted):
# after the torch probe passes, an extended probe checks the
# other top-level packages generate.py imports (numpy, trimesh,
# PIL, trellis2) - checks ONLY, no extra reinstalls in this drop;
# a failure exits 30 naming the module instead of surfacing hours
# later at Phase E/H. Everything else is r33.1, unchanged.
# r33.1: venv repair ONLY - generate.py is byte-identical to r33 and is
# NOT part of this update; the r33 acceptance question is unchanged.
# Job 029 (the r33 acceptance run) died in 10.9 s at the Phase C check:
# torch/masked/maskedtensor/core.py was missing from the venv, so
# `import torch` itself failed while the torch.done marker said
# "installed" and Phase C skipped. The deleter was the Oct 25 disk
# cleanup's core-dump sweep (DISK-CLEANUP.ps1 ITEM 3):
#   find /root /home /tmp ... -name 'core*' -type f -delete
# run across ALL of /root to catch core dumps - it also matches every
# Python module file named core.* in every tree under /root, the
# venv's torch included. The phase markers assert "was installed
# once", not "is intact now", so this build makes Phase C prove
# integrity on EVERY run: an import probe (torch, torch.masked,
# torch.masked.maskedtensor.core, torchvision - each named in the
# log, pass or fail) runs before the GPU gate; on failure the pinned
# pair is force-reinstalled from the same rocm7.2 index Phase C
# used, the probe re-runs, and any remaining failure exits 30
# BEFORE generation. A healthy venv costs one probe (~seconds) and
# no download; the repair is idempotent - probe-first means a
# second run finds nothing to fix.
# r33: chunk isolation for the chunked fill (generate.py) -
# Cory's route call 2026-10-25 after job 028's split verdict
# (the truncation disease dead; the subject cascade arm lost to
# a CuMesh boundary-components kernel failure, code 9, whose
# poisoned HIP context segfaulted the run). Every chunk's stock
# fill now runs in its own short-lived subprocess (generate.py
# re-invoked as --fill-chunk-worker): a kernel failure or
# segfault costs one chunk, which retries once as two
# half-chunks and otherwise falls back UNFILLED while the run
# survives to export. r32's chunk planning, append-only merge
# and verification are unchanged - one variable: isolation.
# r32: the fill_holes fix (generate.py) - Mesh.fill_holes
# round-trips the whole mesh through CuMesh::init's single-shot
# cudaMemcpy2D copies, which silently truncate above 2^20 rows
# on this stack, so the mesh pipeline.run returns is wounded
# AFTER r31's whole extraction (canvas: prefix 1,048,576 + zero
# desert + the fill's own appended patches as the end stub).
# generate.py wraps Mesh.fill_holes so big meshes are filled in
# bounded chunks whose every CuMesh copy stays under the wall,
# merged exactly (the stock fill is append-only), with a guard
# on Mesh.simplify's non-binding round trip. Also: the source
# capture below now includes trellis2/representations - r31's
# hunt had to read fill_holes from the upstream tree because
# the capture never shipped it.
# r31: the extraction fix (generate.py) - the FDG extraction is
# wrapped so the sparse coords are extracted in halo-closed chunks
# whose every output stays under the 2^20 wall, merged by global
# voxel identity; r30's verdict (cascade AND non-cascade control
# wounded identically at exactly 2^20) is the reason. Also: the
# source capture is additionally zipped into the output root -
# r30's loose capture folder reached Drive empty, this build's
# archive is the copy the hunt reads.
BUILD_STAMP="2026-10-27-r35.8-r34-bounds-small"
# r35.6 RESUME SHAPE, kept for job 047 (r35.8): RESUME_045=1
# still switches on the resume run shape - Phase I is
# SKIPPED after verifying the standing canvas GLB against
# job 044's hash-verified copy, and a resume-entry banner
# precedes Phase J (whose census must reproduce job 044's
# arm-2 numbers exactly, as jobs 045 and 046 did).
# r35.8 is the SMALL confirmation test for the r34 bounds
# fix (Cory's call, Oct 27: a smaller test that won't take
# as long). The ONLY code change vs job 046 is generate.py:
# the r34 sectioned-export cleanup override, rebased onto
# r35.7's generate.py (4 hunks; the fill is byte-untouched).
# Three switches narrow this run to the one question - does
# the subject cascade arm's merged export now pass bounds
# verification:
#   PROOF_SKIP=1   - the R1 fill worker is byte-identical
#     to job 046's (sha-locked); its calibration + corpus
#     equivalence stood proven in full THIS MORNING (job
#     046, and job 044 before it). r1-stage sections c/d
#     skip with banners; compile, load probe, battery and
#     the section (e) batch gate still run in full, so the
#     batch flag is re-minted this run, not inherited.
#   SKIP_CONTROL=1 - Phase J2 (the control arm) does not
#     run; it answered its question in 046.
#   SKIP_K2=1      - Phase K2 (CuMesh probe/build) does
#     not run; unrelated to the bounds question.
RESUME_045=1
PROOF_SKIP=1
SKIP_CONTROL=1
SKIP_K2=1
PKG_WSL="${1:?package path (WSL form) required}"
ROOT=/root/trellis-amd-port
MARK="$ROOT/markers"
HF_DIR="$ROOT/hf"
VENV="$ROOT/venv"
OUT="$PKG_WSL/output"

mkdir -p "$ROOT" "$MARK" "$OUT"
cd "$ROOT" || exit 99

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }
banner() { echo; echo "=================================================================="; log "INSIDE-WSL: $*"; echo "=================================================================="; }

# Heartbeat wrapper: long phases print elapsed time every 60 s so a quiet
# window never reads as hung. Usage: with_heartbeat "label" cmd args...
with_heartbeat() {
  local label="$1"; shift
  "$@" &
  local pid=$!
  local start=$SECONDS
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    if kill -0 "$pid" 2>/dev/null; then
      log "  ... still working: $label - elapsed $(( (SECONDS-start)/60 )) min"
    fi
  done
  wait "$pid"
  return $?
}

# ---- Optional local Hugging Face token (Oct 25, 2026, r26). If
# hf-token.local exists in the package's inside/ folder (Cory's local copy
# ONLY - never committed, never uploaded, never part of the public runner),
# its first line is read as a read-only HF_TOKEN so hub requests are
# authenticated and the "unauthenticated requests to the HF Hub" warning
# goes away. The models still load from the local HF cache; this is
# cosmetic today and future-proofs gated originals. The token VALUE is
# never printed or logged anywhere. An absent or empty file changes
# nothing - the build proceeds exactly as before.
HF_TOKEN_FILE="$PKG_WSL/inside/hf-token.local"
if [ -f "$HF_TOKEN_FILE" ]; then
  _hf_tok=$(head -n 1 "$HF_TOKEN_FILE" | tr -d '[:space:]')
  if [ -n "$_hf_tok" ]; then
    export HF_TOKEN="$_hf_tok"
    log "[inside] HF_TOKEN loaded from local token file (value not printed)"
  else
    log "[inside] no local HF token file - hub requests stay unauthenticated (warning only)"
  fi
  unset _hf_tok
else
  log "[inside] no local HF token file - hub requests stay unauthenticated (warning only)"
fi

# ---- Cory-only fan hook (Oct 24, 2026). If fan-hook.local exists in the
# package's inside/ folder (Cory's local copy ONLY - the file is never part of
# the public runner and the hook is inert without it), flip his Windows fan
# tool to full blast for the hot phases (E2 compile, H generation) and restore
# his normal profile after. The file holds two lines, FAN_ON_CMD= / FAN_OFF_CMD=,
# each a Windows command run via powershell.exe through WSL interop. A failed
# fan command never fails the build, and an EXIT trap restores the normal
# profile if the script dies mid-phase.
FAN_STATE=off
fan_hook() {
  local want="$1" f="$PKG_WSL/inside/fan-hook.local" cmd=""
  [ -f "$f" ] || return 0
  if [ "$want" = on ] && [ "$FAN_STATE" = off ]; then
    cmd=$(sed -n 's/^FAN_ON_CMD=//p' "$f" | head -1); FAN_STATE=on
  elif [ "$want" = off ] && [ "$FAN_STATE" = on ]; then
    cmd=$(sed -n 's/^FAN_OFF_CMD=//p' "$f" | head -1); FAN_STATE=off
  else
    return 0
  fi
  cmd=$(printf '%s' "$cmd" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
  [ -n "$cmd" ] || return 0
  log "Fan hook: fans $want (via Windows interop)"
  powershell.exe -NoProfile -Command "$cmd" >/dev/null 2>&1 || log "Fan hook: command failed - fans left as they are."
  return 0
}
trap 'fan_hook off' EXIT

export PYTORCH_ROCM_ARCH=gfx1201
export HSA_ENABLE_DXG_DETECTION=1
# GPU_ARCHS is the arch variable the TRELLIS-family extension setups read
# (o-voxel/setup.py line 22: os.getenv("GPU_ARCHS", "native"); upstream's own
# setup.sh builds with GPU_ARCHS=gfx942). Default "native" = clang autodetect,
# which finds NO GPU in WSL (no /sys/class/kfd) - the 2026-10-24 13:22 run
# failed every HIP compile with "cannot determine hip architecture" for this
# exact reason. Pin it to the card. The other two are the hipcc/CMake-level
# equivalents, for any build that takes its arch from the environment.
export GPU_ARCHS=gfx1201
export HCC_AMDGPU_TARGET=gfx1201
export AMDGPU_TARGETS=gfx1201
# bioritmovideo pitfall, verified A/B on gfx1201: expandable_segments on this
# stack produces SILENT NaNs. The upstream example sets it; we must not.
unset PYTORCH_CUDA_ALLOC_CONF
log "Env guard: PYTORCH_CUDA_ALLOC_CONF is unset (expandable_segments = silent NaNs on gfx1201)."
log "Inside build stamp: $BUILD_STAMP"
if [ "${RESUME_045:-0}" = "1" ]; then
  banner "RESUME RUN - job 047 (r35.8 small test) re-enters at Phase J (subject cascade)"
  log "RESUME: standing by reference to job 044 (drop 2026-10-26-r35.5-r1-fill-attempt6): R1 calibration + corpus equivalence (proven in 044, fill sources byte-identical) and Phase I canvas generation (output verified at the Phase I skip point below)."
  log "RESUME: still running in full: env/census safety phases, R1 gate compile + load probe + battery + batch gate (calibration/equivalence skipped - PROOF_SKIP, standing proof jobs 044+046), Phase H smoke (pre-arm seatbelt), Phase J (the re-entry arm, r34 export under test). NOT running: Phase J2 control (SKIP_CONTROL), Phase K2 (SKIP_K2)."
fi

# r27: clear stale VERDICT files from the persistent output dir before
# any phase writes its own. $OUT survives across runs on the PC, and in
# the r24 run a DOWNGRADED.txt written by the crashed first attempt sat
# in it, was harvested into the recovered run's results, and the
# runner's verdict trusted the file's presence - printing
# "1024_cascade failed" for a run whose inside log says "Phase I done
# at full 1024_cascade". A verdict file must only ever describe the run
# that wrote it. Phase markers in $MARK are NOT touched: they gate
# resume, not verdicts.
for _stale in "$OUT/DOWNGRADED.txt" "$OUT"/*.BOUNDS-FAIL.txt; do
  if [ -e "$_stale" ]; then
    log "r27: clearing stale verdict file from a previous run: $_stale"
    rm -f "$_stale"
  fi
done

# ---------------------------------------------------------------- Phase A: ROCm
if [ -f "$MARK/rocm.done" ]; then
  log "Phase A (ROCm 7.2.1) already done - skipping."
else
  banner "Phase A: install ROCm 7.2.1 (WSL2 way: --no-dkms, driver stays on Windows)"
  log "Download: amdgpu-install package + ROCm graphics,rocm usecase, roughly 6-8 GB. This takes a while."
  apt-get update -y || { log "FAIL: apt-get update"; exit 20; }
  DEB=/tmp/amdgpu-install_7.2.1.70201-1_all.deb
  # AMD moved the deb 2026-03-23: the build-number folder (7.2.1.70201/) is gone;
  # the same file now lives under 7.2.1/. Try the current path, then the old one.
  wget -q https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb -O "$DEB" \
    || wget -q https://repo.radeon.com/amdgpu-install/7.2.1.70201/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb -O "$DEB" \
    || { log "FAIL: could not download amdgpu-install 7.2.1 deb (if this 404s, AMD moved it - tell Joe)"; exit 20; }
  # --allow-downgrades: this distro already carries a NEWER amdgpu-install
  # (seen 2026-10-24 run 3); apt refuses to step down to the pinned 7.2.1
  # installer without it. The 7.2.1 pin is deliberate - the port + the
  # rocm7.2 torch wheels are validated against ROCm 7.2.1.
  # --force-confnew: dpkg otherwise STALLS on an interactive conffile prompt
  # when a config (rocm.list) was locally modified - seen live 2026-10-24 run 4,
  # the window sat waiting for a human Y. Take the package's version: the
  # whole point of the downgrade is a coherent 7.2.1 source list.
  apt-get install -y --allow-downgrades -o Dpkg::Options::="--force-confnew" "$DEB" || { log "FAIL: amdgpu-install deb install"; exit 20; }
  apt-get update -y || exit 20
  # Per the WSL2/RDNA4 survival guide: --usecase=wsl no longer exists in 7.2.1;
  # --no-dkms is essential because the kernel driver lives on the Windows side.
  with_heartbeat "ROCm install" amdgpu-install -y --usecase=graphics,rocm --no-dkms \
    || { log "FAIL: amdgpu-install run"; exit 20; }
  apt-get install -y git cmake ninja-build build-essential python3.12 python3.12-venv python3-pip wget curl \
    || { log "FAIL: build tools install"; exit 20; }
  touch "$MARK/rocm.done"
  log "Phase A done."
fi

# ---------------------------------------------------------------- Phase B: GPU gate
banner "Phase B: does ROCm see the RX 9070 XT (gfx1201) through WSL2 /dev/dxg?"
if /opt/rocm/bin/rocminfo 2>/dev/null | grep -m1 -i "9070" ; then
  log "GPU gate: rocminfo sees the 9070 XT."
elif HSA_OVERRIDE_GFX_VERSION=12.0.1 /opt/rocm/bin/rocminfo 2>/dev/null | grep -m1 -i "9070" ; then
  log "GPU gate: visible with HSA_OVERRIDE_GFX_VERSION=12.0.1 (will export it for later phases)."
  export HSA_OVERRIDE_GFX_VERSION=12.0.1
else
  log "FAIL: rocminfo does not see the GPU. Usual causes: Adrenalin driver older than 26.1.1,"
  log "WSL2 not updated (run 'wsl --update' in Windows), or the distro needs a restart (wsl --shutdown)."
  /opt/rocm/bin/rocminfo 2>&1 | head -40
  exit 10
fi

# ---------------------------------------------------------------- Phase C rebuild (r33.5): one-shot fresh venv
# Job 033's census verdict (see the r33.5 header): of 68 probed
# distributions, 12 were broken; the heal fixed 8 and 4 stayed
# broken - and 3 of those 4 fail BELOW their own top level
# (open3d on its missing dep dash, requests on missing urllib3,
# rembg on its missing onnxruntime backend), where a --no-deps
# force-reinstall cannot reach, while the fourth (nvdiffrast
# 0.4.0) is a source build no index carries. Per-package
# patching has hit its limit; the known-whole answer is a fresh
# venv. This block runs ONCE: the first run under this stamp
# renames the existing venv aside - venv-broken-20261026 (then
# -2, -3 if that name is taken) - and NEVER deletes it: the
# corpse stays on disk for forensics and rollback. Exactly the
# markers gating venv-resident artifacts are moved aside with
# it (to <name>.pre-r33.5, also kept, never deleted):
#   torch.done     Phase C - the venv create + pinned pair
#   ext.done       Phase E - the extensions register into the
#                  venv (editable installs + site-packages)
#   painter.done   Phase E2 - Painter3000's nvdiffrast installs
#   painter.failed into the venv by the same token
# Deliberately NOT reset: rocm.done (Phase A is system ROCm,
# not the venv), repos.done (Phase D's trees live under $ROOT,
# not in the venv - no re-clone), models.done (Phase F's
# weights live in $ROOT/hf). The sentinel lives at
# $ROOT/.r33.5-rebuild-done - outside the venv and outside
# $MARK - so a re-run can never rename the fresh venv; it skips
# this block entirely. After the rename the script falls
# through to the normal Phase C fresh-install path below (venv
# create, pip upgrade, the pinned torch pair from the rocm7.2
# index, the small deps), and every r33.1-r33.4 integrity
# check (probe, pip ladder, extended probe, census) runs
# against the NEW venv.
REBUILD_SENTINEL="$ROOT/.r33.5-rebuild-done"
if [ -f "$REBUILD_SENTINEL" ]; then
  log "r33.5 rebuild: sentinel present ($(cat "$REBUILD_SENTINEL")) - the fresh venv stays; skipping the rebuild block."
else
  banner "Phase C rebuild (r33.5): one-shot fresh venv - the sweep-damaged venv is renamed aside, never deleted"
  if [ -d "$VENV" ]; then
    CORPSE="$ROOT/venv-broken-20261026"
    _n=2
    while [ -e "$CORPSE" ]; do CORPSE="$ROOT/venv-broken-20261026-$_n"; _n=$((_n+1)); done
    mv "$VENV" "$CORPSE" || { log "FAIL: could not rename the damaged venv aside to $CORPSE - refusing to build over it."; exit 30; }
    log "r33.5 rebuild: old venv preserved at $CORPSE (renamed, NOT deleted - the forensics/rollback copy)."
  else
    CORPSE="(none - no pre-existing venv)"
    log "r33.5 rebuild: no existing venv at $VENV - nothing to rename; the fresh-install path below builds the first one."
  fi
  for _m in torch.done ext.done painter.done painter.failed; do
    if [ -f "$MARK/$_m" ]; then
      mv "$MARK/$_m" "$MARK/$_m.pre-r33.5" \
        && log "r33.5 rebuild: marker $_m moved aside to $_m.pre-r33.5 (it gated a venv-resident artifact; the fresh venv must re-earn it)."
    fi
  done
  echo "stamp $BUILD_STAMP rebuilt $(date -Iseconds); old venv: $CORPSE" > "$REBUILD_SENTINEL" \
    || { log "FAIL: could not write the rebuild sentinel - refusing to continue (a re-run must never rename the fresh venv)."; exit 30; }
  log "r33.5 rebuild: sentinel written ($REBUILD_SENTINEL). Falling through to the fresh-install Phase C path."
fi

# ---------------------------------------------------------------- Phase C: torch
if [ -f "$MARK/torch.done" ]; then
  log "Phase C (PyTorch ROCm) already done - skipping."
else
  banner "Phase C: PyTorch from the official rocm7.2 wheel index (download roughly 3.5 GB)"
  python3.12 -m venv "$VENV" || { log "FAIL: venv create"; exit 30; }
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  pip install --upgrade pip || exit 30
  # PINNED to torch 2.11.0 - PROVEN on Cory's machine 2026-10-24 (probe 8).
  # The ladder, all with the GPU enumerated via HSA_ENABLE_DXG_DETECTION=1:
  # 2.14.1 ABORTS, 2.13.0 ABORTS, 2.12.1 ABORTS (rocprofiler-sdk CHECK in
  # c10::cuda::device_count: 0 KFD agents vs 2 HSA agents - no /sys/class/kfd
  # exists in WSL; the integration landed in the 2.12 line, and emptying
  # HSA_TOOLS_LIB does not stop it). 2.11.0 PASSES the gate clean:
  # is_available True, device 0 = RX 9070 XT, GPU matmul OK. The port
  # author's 2.13.0 stack was validated on bare metal, where KFD exists.
  with_heartbeat "PyTorch download/install" pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/rocm7.2 \
    || { log "FAIL: torch install"; exit 30; }
  pip install trimesh huggingface_hub pillow || { log "FAIL: small deps install"; exit 30; }
  touch "$MARK/torch.done"
fi
# shellcheck disable=SC1091
. "$VENV/bin/activate"

# ---------------------------------------------------------------- Phase C integrity probe + repair (r33.1)
# The torch.done marker above proves torch was installed ONCE; it says
# nothing about the files still being there (job 029: the Oct 25 disk
# cleanup's core-dump sweep deleted every file named core* under /root,
# torch/masked/maskedtensor/core.py among them, and Phase C skipped on
# the marker's word). So: probe the install BEFORE the GPU gate, every
# run. The probe imports each module by name and logs pass/fail per
# module, so the log names exactly what is broken. Repair = force-
# reinstall of the SAME pinned pair from the SAME index Phase C used
# (a whole-package restore, because a sweep like that one does not
# leave a countable damage list - the first missing module is just
# where the import happened to die). Then the probe re-runs, and the
# Phase C GPU gate below remains the final fail-loud barrier: nothing
# reaches generation unless imports are clean AND a real GPU matmul
# runs. Idempotent: a healthy venv takes the probe only (~seconds,
# no download); a repaired venv probes clean on the next run.
# r33.2 adds two things to this block. (1) A pip bootstrap rung
# inside the repair branch: job 030 showed pip itself is sweep-
# damaged (pip._vendor.idna.core), so before the reinstall the
# branch bootstraps pip from ensurepip and verifies it - a broken
# pip exits 30 before any reinstall is attempted. (2) An extended
# probe after the torch probe passes: numpy, trimesh, PIL and
# trellis2 - the other top-level packages generate.py imports -
# checked by name, pass or fail. Checks only: this drop reinstalls
# nothing beyond the torch pair; a failure exits 30 naming the
# module, here instead of at Phase E/H.
torch_probe() {
python3 - <<'EOF'
import importlib, sys
mods = ["torch", "torch.masked", "torch.masked.maskedtensor.core", "torchvision"]
bad = 0
for m in mods:
    try:
        importlib.import_module(m)
        print(f"r33.1 probe OK: {m}")
    except Exception as e:
        bad += 1
        print(f"r33.1 probe FAIL: {m}: {type(e).__name__}: {e}")
if bad == 0:
    import torch
    print(f"r33.1 probe: torch {torch.__version__} imports whole - maskedtensor core present")
sys.exit(1 if bad else 0)
EOF
}
# r33.3: the get-pip rung - the fallback when a post-ensurepip pip
# check fails. Job 031 proved the hole in the ensurepip design:
# ensurepip compares versions first, and the venv's pip (26.2.2) is
# newer than ensurepip's bundled wheel, so ensurepip printed
# "Requirement already satisfied" and never replaced the damaged
# files - pip._vendor.idna.core stayed missing. get-pip.py does no
# version comparison: it lays a complete fresh pip over whatever
# is there. The script ships INSIDE this update drop (payload
# target inside/get-pip.py, sha256-verified by the launcher's
# update channel before it is applied - DIAGNOSIS-031.md holds its
# inspection record), so the PC downloads nothing at repair time.
# Returns 0 only if BOTH pip checks pass after it runs.
pip_getpip_rung() {
  local gp=""
  local cand
  for cand in "$(dirname "${BASH_SOURCE[0]}")/get-pip.py" "$PKG_WSL/inside/get-pip.py"; do
    if [ -f "$cand" ]; then gp="$cand"; break; fi
  done
  if [ -z "$gp" ]; then
    log "r33.3: get-pip.py NOT FOUND (looked beside build-inside.sh and at $PKG_WSL/inside/get-pip.py) - the r33.3 payload did not arrive whole."
    return 1
  fi
  log "r33.3: running the get-pip rung: python3 $gp - a fresh pip laid over the damaged one, no version comparison."
  python3 "$gp" || { log "r33.3: get-pip.py itself errored."; return 1; }
  python3 -m pip --version >/dev/null || { log "r33.3: pip still does not run after get-pip."; return 1; }
  python3 -c "import pip._vendor.idna.core" || { log "r33.3: pip._vendor.idna.core still does not import after get-pip."; return 1; }
  log "r33.3: get-pip rung worked - pip runs and pip._vendor.idna.core imports."
  return 0
}
banner "Phase C integrity (r33.2): the installed torch must import whole - probe first, repair only if broken"
if torch_probe; then
  log "r33.1: torch import probe clean - no repair needed."
else
  log "r33.1: torch import probe FAILED - the failing modules are named above. The install is damaged, not uninstalled (the marker said done)."
  log "r33.2: first make sure pip itself runs - job 030's reinstall died inside pip (pip._vendor.idna.core is a sweep casualty too), and no pip-based repair can start until pip works."
  log "r33.2: bootstrapping pip from ensurepip - its bundled wheel lives in the system stdlib, outside every tree the sweep walked..."
  python3 -m ensurepip --upgrade --default-pip \
    || { log "FAIL: ensurepip could not bootstrap pip - NOT attempting the reinstall with a broken pip."; exit 30; }
  python3 -m pip --version >/dev/null \
    || { log "r33.3: pip still does not run after the ensurepip bootstrap - trying the get-pip rung before giving up."; pip_getpip_rung || { log "FAIL: pip is still broken after BOTH ensurepip and get-pip - the repair ladder in this drop is exhausted; NOT attempting the reinstall. The next step (a full venv rebuild) is Cory's decision, not something this drop improvises."; exit 30; }; }
  python3 -c "import pip._vendor.idna.core" \
    || { log "r33.3: pip._vendor.idna.core still does not import after the ensurepip bootstrap - trying the get-pip rung before giving up."; pip_getpip_rung || { log "FAIL: pip is still broken after BOTH ensurepip and get-pip - the repair ladder in this drop is exhausted; NOT attempting the reinstall. The next step (a full venv rebuild) is Cory's decision, not something this drop improvises."; exit 30; }; }
  log "r33.2: pip bootstrapped and verified (version prints, pip._vendor.idna.core imports). The reinstall below can run."
  log "r33.1 repair: force-reinstalling the pinned pair torch==2.11.0 / torchvision==0.26.0 from the Phase C rocm7.2 index."
  log "Download: the two wheels, roughly 3.5 GB - the Oct 25 cleanup purged the pip cache, so this is a full re-download. This takes a while."
  with_heartbeat "torch force-reinstall" pip install --force-reinstall torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/rocm7.2 \
    || { log "FAIL: torch force-reinstall did not complete"; exit 30; }
  if torch_probe; then
    log "r33.1 repair: probe clean after the reinstall - torch is whole again. The GPU gate below still has to pass."
  else
    log "FAIL: torch still does not import whole after a force-reinstall - refusing to continue (the damage is wider than the torch pair)."
    exit 30
  fi
fi

# ---------------------------------------------------------------- Phase C extended probe (r33.2)
# Torch is whole (first probe, or repaired above). The sweep's
# damage list cannot be counted, so check the other top-level
# packages generate.py imports - numpy, trimesh, PIL, trellis2 -
# by name, here, while failing is still cheap. Checks ONLY: this
# drop reinstalls nothing beyond the torch pair. Deliberately NOT
# in this set: o_voxel / flex_gemm / cumesh / nvdiffrast (Phase E
# gates those imports itself, with a rebuild), and huggingface_hub
# (generate.py does not import it directly; Phase H's dependency
# chase covers the deep pipeline import, with auto-repair).
extended_probe() {
TRELLIS2_TREE="$ROOT/TRELLIS.2" python3 - <<'EOF'
import importlib, os, sys
tree = os.environ.get("TRELLIS2_TREE")
if tree and os.path.isdir(tree):
    # generate.py imports the trellis2 package from the source tree;
    # the Phase E gate resolves it the same way (PYTHONPATH fallback).
    sys.path.insert(0, tree)
mods = ["numpy", "trimesh", "PIL", "trellis2"]
bad = 0
for m in mods:
    try:
        importlib.import_module(m)
        print(f"r33.2 probe OK: {m}")
    except Exception as e:
        bad += 1
        print(f"r33.2 probe FAIL: {m}: {type(e).__name__}: {e}")
sys.exit(1 if bad else 0)
EOF
}
banner "Phase C integrity (r33.2): the pipeline's other core imports must be whole too (numpy, trimesh, PIL, trellis2)"
if extended_probe; then
  log "r33.2: extended import probe clean - every named package imports."
else
  log "FAIL: a core pipeline package does not import (named above) - sweep damage beyond the torch pair. This drop reinstalls nothing beyond the torch pair; refusing to continue here instead of at Phase E/H."
  exit 30
fi

# ---------------------------------------------------------------- Phase C census heal (r33.4)
# The torch probe and the extended probe above check the packages
# Joe remembered to name. Job 032 proved that is not enough: the
# run passed every Phase C gate, then died in Phase H prep on
# idna - a package nobody had named, broken the same way (files
# deleted by the core* sweep, metadata intact, pip "satisfied").
# So Phase C now measures the WHOLE venv before the GPU gate: a
# census import-probes every installed distribution, each in its
# own subprocess (a segfaulting or hanging import costs one
# package, not the census; 30 s per package), logging PASS/FAIL
# by name with [r33.4-heal] tags. Excluded from auto-heal:
# torch/torchvision (the pinned ROCm pair - the repair branch
# above owns them), pip (the ensurepip/get-pip ladder owns it),
# the Phase E source builds (nvdiffrast, cumesh, flex-gemm,
# o-voxel - Phase E builds them from source and gates their
# imports itself; r33.5 adds these after job 033 showed the
# census demanding a pip fetch for a package no index carries),
# and any package whose files live outside the venv (probed and
# reported, never reinstalled over). Every other failing package
# is force-reinstalled at its installed version with --no-deps -
# its own files are what is broken, and its dependencies get
# their own census rows. A pinned version pip cannot fetch is
# logged UNHEALED with the reason and the census CONTINUES, so
# one pass measures the full damage list; but after the heal
# pass every healed package is re-probed, and if ANY package
# remains failing the run exits 30 with the list - a generation
# run does not proceed on a known-broken venv. Healthy venv:
# probes only, no downloads; the census is the price of never
# again discovering this damage one job at a time.
# r33.6: the heal learns a SECOND failure class. Job 035's three
# survivors of the --no-deps heal were not damaged at all - their
# own files were whole; a distribution they require was never
# installed (open3d/dash, requests/urllib3, rembg/onnxruntime -
# the last is not even a base requirement of rembg; it ships only
# in rembg's [cpu]/[gpu] extras). A --no-deps reinstall restores a
# package's own files and can never ADD a missing distribution,
# so survivors now get up to 2 dependency-class attempts (the
# census_pkg_missing prober below names what an import is
# missing; the rung installs that distribution WITH deps, or
# force-reinstalls the package WITH deps when no single module
# is named) before the exit-30 verdict. And because this census
# stands BEFORE the phases whose installers populate the broad
# dependency set, a second census gate with the same two heal
# classes now stands after Phase H prep (r33.6) - see there.
# census_pkg_missing <dist>: probes the distribution's top-level
# imports (output captured, never leaked) and reports the first
# failure. stdout: the missing top-level module, or the literal
# "onnxruntime" when the failure text names rembg's backend
# instead of a module, or nothing. Exit: 0 = a module was named,
# 1 = the distribution imports whole, 2 = it fails naming no
# single module (or is not installed at all).
census_pkg_missing() {
  python3 - "$1" <<'PYEOF'
import contextlib, importlib, importlib.metadata as md, io, re, sys
name = sys.argv[1]
try:
    dist = md.distribution(name)
except Exception:
    sys.exit(2)
tops = []
try:
    tl = dist.read_text("top_level.txt")
except Exception:
    tl = None
if tl:
    tops = [ln.strip() for ln in tl.splitlines() if ln.strip()]
if not tops:
    seen = []
    for f in (dist.files or []):
        p = str(f).replace("\\", "/")
        if p.startswith("../") or p.startswith("/"):
            continue
        first = p.split("/")[0]
        if first.endswith(".py"):
            first = first[:-3]
        if (first and first not in seen
                and not first.endswith(".dist-info")
                and not first.endswith(".egg-info")):
            seen.append(first)
    tops = seen
if not tops:
    sys.exit(2)
for t in tops:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            importlib.import_module(t)
    except BaseException as e:
        hay = f"{type(e).__name__}: {e}\n{buf.getvalue()}"
        m = re.search(r"No module named '([A-Za-z0-9_.]+)'", hay)
        if m:
            print(m.group(1).split(".")[0])
            sys.exit(0)
        if "onnxruntime" in hay:
            print("onnxruntime")
            sys.exit(0)
        sys.exit(2)
sys.exit(1)
PYEOF
}
banner "Phase C integrity (r33.4): venv-wide census - probe every installed package, heal what the sweep broke"
cat > "$MARK/venv_census.py" <<'CENSUS_EOF'
#!/usr/bin/env python3
# venv_census.py - r33.4 venv-wide integrity census probe.
#
# Canonical source for the copy embedded into build-inside.sh (the
# r33.4 drop ships build-inside.sh only; at runtime the script writes
# this file to $MARK/venv_census.py and runs it). Kept here so the
# logic can be read and unit-checked without unpacking the shell
# heredoc.
#
# Modes:
#   probe-all <failfile> <exclfile>
#       Enumerate every installed distribution (importlib.metadata);
#       for each, derive its top-level importables (top_level.txt
#       when present, else derived from the distribution's file
#       list) and try importing them in a SUBPROCESS per package
#       (30 s timeout - a segfaulting or hanging import costs one
#       package, not the census). Logs PASS/FAIL by package name.
#       Failing HEALABLE packages are written to <failfile> as
#       name<TAB>version lines; exclusions (with reasons) to
#       <exclfile>. Excluded from auto-heal: torch / torchvision
#       (pinned ROCm pair, owned by the Phase C repair branch),
#       pip (owned by the ensurepip/get-pip ladder), the Phase E
#       source builds nvdiffrast / cumesh / flex-gemm / o-voxel
#       (built from source by Phase E and import-gated there -
#       r33.5; on a fresh venv they are legitimately absent
#       until Phase E runs, and pip-fetching them is wrong), and
#       any package whose files live outside the venv (probed
#       and reported, never reinstalled over).
#   probe-list <listfile> <failfile>
#       Re-probe only the packages named in <listfile>
#       (name<TAB>version lines); still-failing ones go to
#       <failfile> in the same format.
#
# Exit code is 0 when the census itself ran; results live in the
# output files and the [r33.4-heal] log lines. A driver-level error
# exits 2 so the caller can fail loud instead of guessing.
import importlib.metadata as md
import os
import subprocess
import sys

TIMEOUT = 30
# Excluded from census auto-heal, each mapped to the reason logged
# beside it. The pinned pair and pip are owned by their own Phase C
# repair branches. The Phase E set (r33.5) is built from source by
# Phase E and import-gated there: on a fresh venv those four are
# legitimately ABSENT until Phase E runs, and a pip fetch for them
# is wrong (job 033: nvdiffrast==0.4.0 is not on any index - "from
# versions: none"). If one of them IS registered in the venv (an
# earlier install), Phase E's own import gate is its verdict.
NEVER_HEAL = {
    "torch": "own repair branch (pinned ROCm pair)",
    "torchvision": "own repair branch (pinned ROCm pair)",
    "pip": "own repair branch (ensurepip/get-pip ladder)",
    "nvdiffrast": "built from source by Phase E, gated there",
    "cumesh": "built from source by Phase E, gated there",
    "flex-gemm": "built from source by Phase E, gated there",
    "o-voxel": "built from source by Phase E, gated there",
}

PROBE_SRC = r"""
import importlib, sys
bad = []
for t in sys.argv[1:]:
    try:
        importlib.import_module(t)
    except BaseException as e:
        bad.append((t, type(e).__name__, str(e)[:160]))
for t, k, m in bad:
    print(f"[r33.4-heal]   FAILTOP {t}: {k}: {m}")
sys.exit(1 if bad else 0)
"""


def norm(name):
    return name.lower().replace("_", "-").replace(".", "-")


def tops_for(dist):
    """Top-level importables: top_level.txt, else derived from files."""
    tl = None
    try:
        tl = dist.read_text("top_level.txt")
    except Exception:
        tl = None
    tops = []
    if tl:
        tops = [ln.strip() for ln in tl.splitlines() if ln.strip()]
    if not tops:
        seen = []
        for f in (dist.files or []):
            p = str(f).replace("\\", "/")
            if p.startswith("../") or p.startswith("/"):
                continue
            first = p.split("/")[0]
            if first.endswith((".dist-info", ".egg-info")):
                continue
            if first == "__pycache__":
                continue
            if "/" not in p:
                if first.endswith(".py"):
                    first = first[:-3]
                else:
                    continue
            if first.isidentifier() and first not in seen:
                seen.append(first)
        tops = seen
    return [t for t in tops if t.isidentifier()]


def files_outside_venv(dist, venv):
    for f in (dist.files or []):
        try:
            ap = os.path.realpath(str(dist.locate_file(f)))
        except Exception:
            continue
        if os.path.commonpath([ap, venv]) != venv:
            return True
    return False


def probe_pkg(tops):
    """Import all top-levels in one subprocess. Returns (ok, why)."""
    if not tops:
        return (True, "no importable top-levels recorded")
    try:
        r = subprocess.run([sys.executable, "-c", PROBE_SRC] + tops,
                           capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return (False, f"probe subprocess timed out after {TIMEOUT}s")
    out = (r.stdout or "").strip()
    if out:
        print(out)
        sys.stdout.flush()
    if r.returncode == 0:
        return (True, "")
    if r.returncode < 0:
        return (False, f"probe subprocess died on signal {-r.returncode} (segfault-class)")
    err = (r.stderr or "").strip().splitlines()
    return (False, err[-1][:200] if err else f"probe exited {r.returncode}")


def census_items():
    venv = os.path.realpath(sys.prefix)
    items = []
    for dist in md.distributions():
        name = dist.metadata.get("Name") or dist.name
        items.append((name, dist.version, dist))
    items.sort(key=lambda x: norm(x[0]))
    return venv, items


def probe_all(failpath, exclpath):
    venv, items = census_items()
    print(f"[r33.4-heal] census: {len(items)} distributions installed under {venv}")
    sys.stdout.flush()
    fails, excls = [], []
    npass = 0
    for name, ver, dist in items:
        tops = tops_for(dist)
        ok, why = probe_pkg(tops)
        if norm(name) in NEVER_HEAL_SET:
            reason = NEVER_HEAL[norm(name)]
            excls.append((name, ver, reason))
            print(f"[r33.4-heal] EXCLUDE {name} {ver}: {reason} - probed, never census-healed")
            if not ok:
                print(f"[r33.4-heal]   NOTE: excluded {name} also fails its probe ({why}) - its own branch/gate owns that verdict")
            sys.stdout.flush()
            continue
        if files_outside_venv(dist, venv):
            excls.append((name, ver, "files outside the venv"))
            print(f"[r33.4-heal] EXCLUDE {name} {ver}: files outside the venv - probed, never auto-healed")
            if not ok:
                print(f"[r33.4-heal]   NOTE: excluded {name} fails its probe ({why}) - outside the venv, reported not healed")
            sys.stdout.flush()
            continue
        if ok:
            npass += 1
            print(f"[r33.4-heal] PASS {name} {ver}")
        else:
            print(f"[r33.4-heal] FAIL {name} {ver}: {why}")
            fails.append((name, ver))
        sys.stdout.flush()
    with open(failpath, "w") as fh:
        for n, v in fails:
            fh.write(f"{n}\t{v}\n")
    with open(exclpath, "w") as fh:
        for n, v, r in excls:
            fh.write(f"{n}\t{v}\t{r}\n")
    print(f"[r33.4-heal] census totals: {len(items)} probed, {npass} pass, "
          f"{len(fails)} FAIL (healable), {len(excls)} excluded")
    return 0


def probe_list(listpath, failpath):
    want = {}
    with open(listpath) as fh:
        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            if parts and parts[0]:
                want[norm(parts[0])] = parts[1] if len(parts) > 1 else ""
    _venv, items = census_items()
    byname = {norm(n): (n, v, d) for n, v, d in items}
    fails = []
    for key in sorted(want):
        if key not in byname:
            print(f"[r33.4-heal] REPROBE FAIL {key}: distribution gone after heal attempt")
            fails.append((key, want[key]))
            continue
        name, ver, dist = byname[key]
        ok, why = probe_pkg(tops_for(dist))
        if ok:
            print(f"[r33.4-heal] REPROBE PASS {name} {ver}")
        else:
            print(f"[r33.4-heal] REPROBE FAIL {name} {ver}: {why}")
            fails.append((name, ver))
        sys.stdout.flush()
    with open(failpath, "w") as fh:
        for n, v in fails:
            fh.write(f"{n}\t{v}\n")
    return 0


NEVER_HEAL_SET = {norm(n) for n in NEVER_HEAL}


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "probe-all" and len(sys.argv) == 4:
        return probe_all(sys.argv[2], sys.argv[3])
    if len(sys.argv) >= 2 and sys.argv[1] == "probe-list" and len(sys.argv) == 4:
        return probe_list(sys.argv[2], sys.argv[3])
    print("usage: venv_census.py probe-all <failfile> <exclfile> | probe-list <listfile> <failfile>")
    return 2


if __name__ == "__main__":
    sys.exit(main())
CENSUS_EOF
CENSUS_FAIL="$MARK/census-fail.tsv"
CENSUS_EXCL="$MARK/census-excl.tsv"
CENSUS_UNHEALED="$MARK/census-unhealed.tsv"
CENSUS_STILL="$MARK/census-stillbroken.tsv"
: > "$CENSUS_FAIL"; : > "$CENSUS_EXCL"; : > "$CENSUS_UNHEALED"; : > "$CENSUS_STILL"
if ! python3 "$MARK/venv_census.py" probe-all "$CENSUS_FAIL" "$CENSUS_EXCL"; then
  log "FAIL: the r33.4 census probe itself errored - refusing to continue on an unmeasured venv."
  exit 30
fi
CENSUS_NFAIL=$(grep -c . "$CENSUS_FAIL" || true)
if [ "$CENSUS_NFAIL" -eq 0 ]; then
  log "r33.4 heal: census clean - every healable package in the venv imports whole. No reinstalls needed."
else
  log "r33.4 heal: census found $CENSUS_NFAIL broken package(s) - force-reinstalling each at its installed version (--no-deps; a plain install would answer 'already satisfied' over the missing files)."
  while IFS=$'\t' read -r PKGNAME PKGVER; do
    [ -n "$PKGNAME" ] || continue
    if with_heartbeat "r33.4 heal $PKGNAME" pip install --force-reinstall --no-deps "$PKGNAME==$PKGVER"; then
      log "r33.4 heal: reinstalled $PKGNAME==$PKGVER - re-probe below decides whether it healed."
    else
      log "r33.4 heal: UNHEALED $PKGNAME==$PKGVER - pip could not fetch or reinstall the pinned version (pip output above); continuing so one pass measures the full damage list."
      printf '%s\t%s\t%s\n' "$PKGNAME" "$PKGVER" "pip force-reinstall failed" >> "$CENSUS_UNHEALED"
    fi
  done < "$CENSUS_FAIL"
  if ! python3 "$MARK/venv_census.py" probe-list "$CENSUS_FAIL" "$CENSUS_STILL"; then
    log "FAIL: the r33.4 heal re-probe itself errored - refusing to continue."
    exit 30
  fi
  CENSUS_NSTILL=$(grep -c . "$CENSUS_STILL" || true)
  log "r33.4 heal totals: $CENSUS_NFAIL package(s) failed the census, $((CENSUS_NFAIL - CENSUS_NSTILL)) healed and re-probed whole, $CENSUS_NSTILL still broken."
  if [ -s "$CENSUS_STILL" ]; then
    # r33.6 deps-heal rung, r33.7 v2: whatever survives the
    # --no-deps heal is the second failure class - the package's
    # own files are whole but a distribution it requires was
    # never installed (job 035: open3d without dash, requests
    # without urllib3, rembg without its onnxruntime backend). A
    # --no-deps reinstall can never ADD a missing distribution,
    # so each survivor climbs a dependency ladder before this run
    # refuses: named-module installs first (cap 2 - cheap, and
    # they went 2/2 in job 036), then the r33.7 WHOLE-CLOSURE
    # step for chains and extras-gated packages (one plain
    # pip install "pkg[extras]" resolver pass; the changed
    # distributions are logged and the torch probe re-runs after
    # it), and force-reinstall-with-deps as the last resort when
    # no single module is named. The survivors are then
    # re-probed, and anything still broken exits 30 loud below -
    # unchanged.
    log "r33.6 deps-heal: $CENSUS_NSTILL package(s) survived the file-damage heal - trying the dependency-class heal before refusing (ladder: named-module installs cap 2, r33.7 whole-closure pass for chains/extras-gated, force-reinstall-with-deps last)."
    CENSUS_STILL2="$MARK/census-stillbroken2.tsv"
    : > "$CENSUS_STILL2"
    while IFS=$'\t' read -r PKGNAME PKGVER; do
      [ -n "$PKGNAME" ] || continue
      HAMMERED=0
      CLOSURED=0
      NAMED_TRIES=0
      FIRSTMISS=""
      CLOSURE_EXTRAS=""
      case "$PKGNAME" in
        rembg) CLOSURE_EXTRAS="cpu" ;;
      esac
      while :; do
        MISSFILE="$MARK/.census-miss.txt"
        census_pkg_missing "$PKGNAME" > "$MISSFILE"
        MISSRC=$?
        MISS=$(head -n 1 "$MISSFILE" | sed 's/[^A-Za-z0-9_.]//g')
        rm -f "$MISSFILE"
        if [ "$MISSRC" -eq 1 ]; then
          log "r33.6 deps-heal: '$PKGNAME' imports whole now - the dependency rung is done with it (the re-probe below is the verdict)."
          break
        fi
        if [ "$CLOSURED" -eq 0 ] && { [ -n "$CLOSURE_EXTRAS" ] || { [ "$MISSRC" -eq 0 ] && [ -n "$MISS" ] && [ -n "$FIRSTMISS" ] && [ "$MISS" != "$FIRSTMISS" ]; }; }; then
          CLOSURED=1
          if [ -n "$CLOSURE_EXTRAS" ]; then
            CLOSURE_TARGET="$PKGNAME[$CLOSURE_EXTRAS]"
            CLOSURE_WHY="'$PKGNAME' is in the extras-gated table ([$CLOSURE_EXTRAS]) - its missing pieces ship behind the extra, so chasing named modules can never finish it"
          else
            CLOSURE_TARGET="$PKGNAME"
            CLOSURE_WHY="'$PKGNAME' named a SECOND distinct missing module ('$FIRSTMISS', now '$MISS') - the chain signature; hand-chasing modules reinvents pip's resolver one link at a time (job 036: rembg onnxruntime -> pymatting -> scikit-image, cap spent mid-chain)"
          fi
          log "[r33.7 closure-heal] $CLOSURE_WHY - running ONE resolver pass: pip install \"$CLOSURE_TARGET\" (plain install, NOT force-reinstall: the package's own files are whole, its closure is incomplete; a plain install fills the missing members and leaves satisfied and pinned distributions alone)."
          CLOUT="$MARK/.closure-out.txt"
          : > "$CLOUT"
          with_heartbeat "r33.7 closure $PKGNAME" bash -c 'set -o pipefail; pip install "$1" 2>&1 | tee "$2"' _ "$CLOSURE_TARGET" "$CLOUT" < /dev/null \
            || log "WARN: [r33.7 closure-heal] the resolver pass for $CLOSURE_TARGET reported problems (continuing; the re-probe decides)."
          CLOSEDIST=$(grep -E "^Installing collected packages:" "$CLOUT" | tail -n 1 | sed "s/^Installing collected packages: //")
          rm -f "$CLOUT"
          log "[r33.7 closure-heal] '$PKGNAME' closure pass - distributions pip installed or changed: ${CLOSEDIST:-<none - pip printed no 'Installing collected packages' line>}"
          if torch_probe; then
            log "[r33.7 closure-heal] torch probe after the '$PKGNAME' closure pass: clean - the pinned stack stood (any drift would be named in the changed-distributions line above, never silent)."
          else
            log "FAIL: [r33.7 closure-heal] the torch probe FAILS after the '$PKGNAME' closure pass - the closure pass is the suspect (distributions it changed: ${CLOSEDIST:-unknown}). Refusing to continue on a disturbed pinned stack."
            exit 30
          fi
          continue
        fi
        if [ "$MISSRC" -eq 0 ] && [ -n "$MISS" ]; then
          if [ "$NAMED_TRIES" -ge 2 ]; then
            log "r33.6 deps-heal: '$PKGNAME' - both named-module attempts are spent and it still names missing module '$MISS'. Leaving it for the verdict below."
            break
          fi
          NAMED_TRIES=$((NAMED_TRIES + 1))
          if [ -z "$FIRSTMISS" ]; then FIRSTMISS="$MISS"; fi
          case "$MISS" in
            PIL) DEPSDIST=Pillow ;;
            cv2) DEPSDIST=opencv-python ;;
            yaml) DEPSDIST=PyYAML ;;
            sklearn) DEPSDIST=scikit-learn ;;
            *) DEPSDIST="$MISS" ;;
          esac
          if python3 -c "import importlib.metadata as m, sys; m.version(sys.argv[1])" "$DEPSDIST" >/dev/null 2>&1; then
            DEPSVER=$(python3 -c "import importlib.metadata as m, sys; print(m.version(sys.argv[1]))" "$DEPSDIST")
            log "r33.6 deps-heal: '$PKGNAME' names missing module '$MISS', but distribution '$DEPSDIST' IS installed - the damage is inside it; force-reinstalling '$DEPSDIST==$DEPSVER' (--no-deps: its own files are the problem). Attempt $NAMED_TRIES of 2 for '$PKGNAME'."
            with_heartbeat "r33.6 deps $PKGNAME" pip install --force-reinstall --no-deps "$DEPSDIST==$DEPSVER" < /dev/null \
              || log "WARN: r33.6 deps-heal force-reinstall of $DEPSDIST reported problems (continuing; the re-probe decides)."
          else
            log "r33.6 deps-heal: '$PKGNAME' names missing module '$MISS' and distribution '$DEPSDIST' is not installed - installing it WITH its dependencies. Attempt $NAMED_TRIES of 2 for '$PKGNAME'."
            with_heartbeat "r33.6 deps $PKGNAME" pip install "$DEPSDIST" < /dev/null \
              || log "WARN: r33.6 deps-heal install of $DEPSDIST reported problems (continuing; the re-probe decides)."
          fi
        elif [ "$HAMMERED" -eq 0 ]; then
          HAMMERED=1
          log "r33.6 deps-heal: '$PKGNAME' names no single missing module - force-reinstalling '$PKGNAME==$PKGVER' WITH dependencies so pip resolves its whole declared closure (last-resort step for '$PKGNAME')."
          with_heartbeat "r33.6 deps $PKGNAME" pip install --force-reinstall "$PKGNAME==$PKGVER" < /dev/null \
            || log "WARN: r33.6 deps-heal force-reinstall of $PKGNAME reported problems (continuing; the re-probe decides)."
        else
          log "r33.6 deps-heal: '$PKGNAME' - the dependency ladder is spent (named-module cap, closure pass, force-reinstall) and it still names no single missing module. Leaving it for the verdict below."
          break
        fi
      done
    done < "$CENSUS_STILL"
    if ! python3 "$MARK/venv_census.py" probe-list "$CENSUS_STILL" "$CENSUS_STILL2"; then
      log "FAIL: the r33.6 deps-heal re-probe itself errored - refusing to continue."
      exit 30
    fi
    CENSUS_NSTILL2=$(grep -c . "$CENSUS_STILL2" || true)
    log "r33.6 deps-heal totals: $CENSUS_NSTILL package(s) entered the dependency rung, $((CENSUS_NSTILL - CENSUS_NSTILL2)) healed and re-probed whole, $CENSUS_NSTILL2 still broken."
    if [ -s "$CENSUS_STILL2" ]; then
      log "FAIL: these packages STILL do not import after BOTH heal classes (file-damage --no-deps heal, then the dependency rung):"
      while IFS=$'\t' read -r PKGNAME PKGVER; do
        [ -n "$PKGNAME" ] || continue
        REASON=$(awk -F'\t' -v n="$PKGNAME" '$1==n {print $3}' "$CENSUS_UNHEALED" | head -1)
        log "  still broken: $PKGNAME==$PKGVER${REASON:+ ($REASON)}"
      done < "$CENSUS_STILL2"
      log "Refusing to continue: the run must not proceed on a known-broken venv. The list above is what neither heal class could repair."
      exit 30
    fi
    log "r33.6 deps-heal: every survivor of the file-damage heal was healed by the dependency rung and re-probed whole."
  fi
  log "r33.4 heal: every census failure healed and re-probed whole - the venv is measured clean."
fi

banner "Phase C check: torch must report the GPU (this is the anti-silent-CPU-fallback gate)"
python3 - <<'EOF'
import sys, torch
print(f"torch {torch.__version__}  compiled hip: {getattr(torch.version, 'hip', None)}")
ok = torch.cuda.is_available()
print("torch.cuda.is_available():", ok)
if ok:
    print("device 0:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device='cuda'); y = x @ x
    print("1024x1024 matmul on GPU ok, mean:", float(y.mean()))
sys.exit(0 if ok else 1)
EOF
[ $? -eq 0 ] || { log "FAIL: torch does not see a working GPU"; exit 30; }

# ---------------------------------------------------------------- Phase D: source trees
if [ -f "$MARK/repos.done" ]; then
  log "Phase D (source trees) already done - skipping."
else
  banner "Phase D: clone upstream TRELLIS.2 + the community AMD patch stacks (under 1 GB total)"
  [ -d "$ROOT/TRELLIS.2" ] || git clone --depth 1 https://github.com/microsoft/TRELLIS.2 "$ROOT/TRELLIS.2" || { log "FAIL: clone TRELLIS.2"; exit 40; }
  [ -d "$ROOT/port-patches" ] || git clone --depth 1 https://github.com/bioritmovideo/trellis2-rocm-gfx1201 "$ROOT/port-patches" || { log "FAIL: clone bioritmovideo patches"; exit 40; }
  [ -d "$ROOT/egore" ] || git clone --depth 1 https://github.com/egore/comfyui-trellis2-gguf-rocm "$ROOT/egore" || { log "FAIL: clone egore patches"; exit 40; }
  touch "$MARK/repos.done"
fi

# ---------------------------------------------------------------- Phase E: native extensions (THE long phase)
# Self-heal: a failed run touches ext.done BEFORE its import gate runs (the
# 2026-10-24 13:22 run did exactly that, with nothing built). The marker only
# counts if the extensions import right now; otherwise rebuild Phase E.
if [ -f "$ROOT/pythonpath.txt" ]; then export PYTHONPATH="$(cat "$ROOT/pythonpath.txt"):$PYTHONPATH"; fi
if [ -f "$MARK/ext.done" ] && python3 -c "import o_voxel, flex_gemm, cumesh, nvdiffrast" 2>/dev/null; then
  log "Phase E (native extensions) already done - skipping."
else
  rm -f "$MARK/ext.done"
  banner "Phase E: build the six native extensions for gfx1201 via HIP"
  log "This is the long, quiet phase. On the port author's machine the CuMesh compile"
  log "alone took ~70 minutes serial. Expect 45-120 minutes total. The heartbeat"
  log "prints every minute; a silent window is a WORKING window here."
  cd "$ROOT/TRELLIS.2" || exit 40
  git submodule update --init --recursive || log "WARN: submodule init reported problems (continuing; import gate decides)"
  # Community diff for the TRELLIS.2 tree (chunked fp32 GEMM, sdpa sparse attention,
  # BiRefNet fp32, DINOv3 mirror swap). If upstream drifted, this may not apply cleanly.
  if git apply --check "$ROOT/port-patches/patches/trellis2_gfx1201.diff" 2>/dev/null; then
    git apply "$ROOT/port-patches/patches/trellis2_gfx1201.diff" && log "bioritmovideo diff applied cleanly."
  else
    log "WARN: bioritmovideo diff does not apply cleanly (upstream drifted?). Trying 3-way..."
    git apply --3way "$ROOT/port-patches/patches/trellis2_gfx1201.diff" \
      || log "WARN: diff failed even 3-way. Continuing - the import + generation gates will tell the truth."
  fi
  # Stack 1: build whatever extension dirs exist in the upstream tree, one by one.
  for d in "$ROOT"/TRELLIS.2/extensions/*/ "$ROOT"/TRELLIS.2/o-voxel "$ROOT"/TRELLIS.2/flex_gemm "$ROOT"/TRELLIS.2/cumesh "$ROOT"/TRELLIS.2/nvdiffrast "$ROOT"/TRELLIS.2/nvdiffrec_render; do
    [ -d "$d" ] || continue
    if [ -f "$d/setup.py" ] || [ -f "$d/pyproject.toml" ]; then
      log "Building extension in $d ..."
      with_heartbeat "extension build $(basename "$d")" pip install -e "$d" --no-build-isolation \
        || log "WARN: build failed in $d (continuing; import gate decides)"
    fi
  done
  # Stack 2: egore's installer builds the same extensions with his HIP patch set
  # (inside his GGUF-nodes tree). Second documented stack; the import gate arbitrates.
  if [ -f "$ROOT/egore/install-trellis2-gguf-rocm.sh" ]; then
    log "Running egore's ROCm installer for its patched extension builds..."
    ( cd "$ROOT/egore" && with_heartbeat "egore extension builds" bash install-trellis2-gguf-rocm.sh ) \
      || log "WARN: egore installer reported failures (continuing; import gate decides)"
  fi
  touch "$MARK/ext.done"
fi

# Import gate: the extensions must actually load, from one stack or the other.
banner "Phase E gate: can Python import the native extensions?"
EXTRA_PATHS=$(find "$ROOT/egore" "$ROOT" -maxdepth 6 -name "*.so" -path "*trellis*" 2>/dev/null | head -20)
if ! python3 -c "import o_voxel, flex_gemm, cumesh, nvdiffrast" 2>/dev/null; then
  # try adding directories that hold built trellis extension modules
  CANDIDATES=$(find "$ROOT" -name "o_voxel" -type d 2>/dev/null | head -5)
  for c in $CANDIDATES; do export PYTHONPATH="$(dirname "$c"):$PYTHONPATH"; done
  export PYTHONPATH="$ROOT/TRELLIS.2:$PYTHONPATH"
fi
echo "$PYTHONPATH" > "$ROOT/pythonpath.txt"
python3 -c "import o_voxel, flex_gemm, cumesh, nvdiffrast; print('extensions import OK')" \
  || { log "FAIL: native extensions do not import from either patch stack. Build logs above are the evidence."; exit 40; }

# ---------------------------------------------------------------- Phase E2: Painter3000 nvdiffrast (HIP rasterizer for the texture bake)
fan_hook on
# o_voxel.postprocess.to_glb hardcodes dr.RasterizeCudaContext() for texture
# baking. The egore-stack nvdiffrast refuses it on ROCm ("CUDA rasterizer not
# available on ROCm. Use RasterizeGLContext"), and its GL context is only a
# deprecated alias of the same CUDA path (probe 9, 2026-10-24). Painter3000's
# v52 stack implements the CUDA-context API on HIP and is the build the
# TRELLIS.2 port author validated textures with (2048px, 110/110). It
# REPLACES the venv's nvdiffrast. His code stays his: we run his installer
# as a dependency; none of it is copied into our files.
raster_gate() {
python3 - <<'PY' 2>/dev/null
import torch, nvdiffrast.torch as dr
pos = torch.tensor([[-0.5,-0.5,0.0,1.0],[0.5,-0.5,0.0,1.0],[0.5,0.5,0.0,1.0],[-0.5,0.5,0.0,1.0]], dtype=torch.float32, device="cuda").unsqueeze(0)
tri = torch.tensor([[0,1,2],[0,2,3]], dtype=torch.int32, device="cuda")
ctx = dr.RasterizeCudaContext()
out, _ = dr.rasterize(ctx, pos, tri, resolution=[64,64])
assert int((out[...,3] > 0).sum()) > 100
PY
}
if [ -f "$MARK/painter.done" ] && raster_gate; then
  log "Phase E2 (Painter3000 nvdiffrast) already done - skipping."
else
  rm -f "$MARK/painter.done"
  banner "Phase E2: install Painter3000's HIP nvdiffrast (the texture-bake rasterizer)"
  log "Two nvdiffrast builds plus his quick validation - roughly 5-15 minutes (unmeasured on this machine)."
  log "Textures are an UPGRADE, not a gate: if this phase fails, the run CONTINUES shape-only and says so."
  P_OK=1
  [ -d "$ROOT/painter3000" ] || git clone --depth 1 https://github.com/Painter3000/amd-nvdiffrast-rocm72-gfx1201 "$ROOT/painter3000" \
    || { log "E2-FAIL: could not clone Painter3000's repo - continuing without textures."; P_OK=0; }
  if [ "$P_OK" = 1 ]; then
    # Compat headers, all ours. Real NVIDIA headers cannot coexist with
    # HIP headers in one translation unit (typedef collisions - r10's
    # lesson), so cuda_runtime_api.h FORWARDS to HIP, and the five library
    # headers torch's CUDAContext chain includes but this box lacks
    # (probe 11: cublas_v2, cublasLt, cusparse, cusolverDn, cudss) are
    # opaque-type bridges. Written for this package, not copied.
    COMPAT_OURS="$ROOT/cuda-compat-ours"; mkdir -p "$COMPAT_OURS"
    cat > "$COMPAT_OURS/cuda_runtime_api.h" <<'HEOF'
#pragma once
/* Bridge header (ours): forwards CUDA runtime names to HIP. */
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
typedef hipError_t cudaError_t;
typedef hipStream_t cudaStream_t;
typedef hipEvent_t cudaEvent_t;
typedef hipDeviceProp_t cudaDeviceProp_t;
typedef hipMemcpyKind cudaMemcpyKind_t;
typedef hipPointerAttribute_t cudaPointerAttributes;
typedef hipFunction_t cudaFunction_t;
typedef hipModule_t cudaModule_t;
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamQuery hipStreamQuery
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
/* r14: plain tag-style spellings + the stream-capture family. Real CUDA
   declares these types as enum/struct tags, so torch's headers use the
   names WITHOUT the _t suffix; r13's compile named every one of these
   (the compiler's own "did you mean" hints confirm the HIP spellings). */
typedef hipDeviceProp_t cudaDeviceProp;
typedef hipMemcpyKind cudaMemcpyKind;
typedef hipStreamCaptureMode cudaStreamCaptureMode;
typedef hipStreamCaptureStatus cudaStreamCaptureStatus;
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusInvalidated hipStreamCaptureStatusInvalidated
#define cudaStreamGetPriority hipStreamGetPriority
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode
#define cudaStreamGetCaptureInfo hipStreamGetCaptureInfo
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaEventQuery hipEventQuery
/* CUDAContextLight.h names cusolverDnHandle_t without including a header
   that declares it on this stack (r13) - declared here identically to our
   cusolverDn.h bridge, so both may appear in one translation unit. */
struct cusolverDnContext; typedef struct cusolverDnContext *cusolverDnHandle_t;

// r15 additions, from the r14 run's compile errors (CUDAGuardImpl.h /
// CUDACachingAllocator.h) plus their certain siblings. HIP's hip_runtime_api.h
// mirrors CUDA's error/event enumerators 1:1 by design, and #defines expand
// only where used, so bridging the whole families is safe: a name a given TU
// never uses costs nothing. CONFIRMED by the r14 diagnostic compile:
// cudaMemGetInfo, cudaEventDefault, cudaEventDisableTiming, cudaErrorNotReady.
// HIP spellings follow the compiler's own "did you mean" hints across
// r13-r14 and hip_runtime_api.h's declarations (note HIP spells the host
// family hipHost*, not hipMallocHost).
#define cudaMemGetInfo hipMemGetInfo
#define cudaEventDefault hipEventDefault
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventBlockingSync hipEventBlockingSync
#define cudaEventInterprocess hipEventInterprocess
#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorMemoryAllocation hipErrorMemoryAllocation
#define cudaErrorInitializationError hipErrorInitializationError
#define cudaErrorLaunchFailure hipErrorLaunchFailure
#define cudaErrorInvalidDevice hipErrorInvalidDevice
#define cudaErrorNoDevice hipErrorNoDevice
#define cudaErrorUnknown hipErrorUnknown
#define cudaErrorInvalidConfiguration hipErrorInvalidConfiguration
#define cudaErrorPriorLaunchFailure hipErrorPriorLaunchFailure
#define cudaErrorLaunchTimeout hipErrorLaunchTimeout
#define cudaErrorLaunchOutOfResources hipErrorLaunchOutOfResources
#define cudaErrorInvalidSymbol hipErrorInvalidSymbol
#define cudaErrorInvalidHostPointer hipErrorInvalidHostPointer
#define cudaErrorInvalidDevicePointer hipErrorInvalidDevicePointer
#define cudaErrorOperatingSystem hipErrorOperatingSystem
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#define cudaErrorSetOnActiveProcess hipErrorSetOnActiveProcess
#define cudaErrorContextIsDestroyed hipErrorContextIsDestroyed
#define cudaErrorAssert hipErrorAssert
#define cudaErrorHostMemoryAlreadyRegistered hipErrorHostMemoryAlreadyRegistered
#define cudaErrorHostMemoryNotRegistered hipErrorHostMemoryNotRegistered
#define cudaErrorECCUncorrectable hipErrorECCUncorrectable
#define cudaErrorUnsupportedLimit hipErrorUnsupportedLimit
#define cudaErrorDeviceUninitialized hipErrorDeviceUninitialized
#define cudaErrorStreamCaptureUnsupported hipErrorStreamCaptureUnsupported
#define cudaErrorStreamCaptureInvalidated hipErrorStreamCaptureInvalidated
#define cudaErrorStreamCaptureMerge hipErrorStreamCaptureMerge
#define cudaErrorStreamCaptureUnmatched hipErrorStreamCaptureUnmatched
#define cudaErrorStreamCaptureUnjoined hipErrorStreamCaptureUnjoined
#define cudaErrorStreamCaptureIsolation hipErrorStreamCaptureIsolation
#define cudaErrorStreamCaptureImplicit hipErrorStreamCaptureImplicit
#define cudaErrorCapturedEvent hipErrorCapturedEvent
#define cudaErrorStreamCaptureWrongThread hipErrorStreamCaptureWrongThread
#define cudaErrorTimeout hipErrorTimeout
#define cudaErrorGraphExecUpdateFailure hipErrorGraphExecUpdateFailure
#define cudaMallocHost hipHostMalloc
#define cudaHostAlloc hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDeviceGetLimit hipDeviceGetLimit
#define cudaDeviceSetLimit hipDeviceSetLimit
#define cudaStreamGetFlags hipStreamGetFlags

// r16: the r15 run's compile is down to ONE undeclared name -
// cudaEventCreateWithFlags (CUDAGuardImpl.h:106, "1 error generated") - the
// enumerator families held. This round bridges the function families of the
// same vocabulary: event, stream, device, and memory entry points c10/ATen
// headers call inline. Every HIP twin below is a mechanical rename in
// hip_runtime_api.h; macros expand only where a TU actually uses the name.
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamQuery hipStreamQuery
#define cudaStreamAddCallback hipStreamAddCallback
#define cudaStreamAttachMemAsync hipStreamAttachMemAsync
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceReset hipDeviceReset
#define cudaSetDeviceFlags hipSetDeviceFlags
#define cudaGetDeviceFlags hipGetDeviceFlags
#define cudaChooseDevice hipChooseDevice
#define cudaDeviceGetByPCIBusId hipDeviceGetByPCIBusId
#define cudaDeviceTotalMem hipDeviceTotalMem
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMallocManaged hipMallocManaged
#define cudaMemPrefetchAsync hipMemPrefetchAsync
#define cudaMemAdvise hipMemAdvise
typedef enum cudaDataType_t { CUDA_R_32F=0, CUDA_R_64F=1, CUDA_R_16F=2, CUDA_R_8I=3, CUDA_C_32F=4, CUDA_C_64F=5, CUDA_C_16F=6, CUDA_R_8U=8, CUDA_R_32I=10, CUDA_R_32U=11, CUDA_R_64I=12, CUDA_R_64U=13, CUDA_R_16BF=14, CUDA_C_8I=15, CUDA_C_8U=16, CUDA_C_32I=17, CUDA_C_32U=18, CUDA_C_64I=19, CUDA_C_64U=20, CUDA_C_16BF=21 } cudaDataType;
HEOF
    cat > "$COMPAT_OURS/cuda.h" <<'HEOF'
#pragma once
/* Bridge header (ours): CUDA driver API names forwarded to HIP. */
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
typedef int CUdevice;
typedef enum CUresult_enum { CUDA_SUCCESS=0, CUDA_ERROR_INVALID_VALUE=1, CUDA_ERROR_OUT_OF_MEMORY=2, CUDA_ERROR_NOT_INITIALIZED=3, CUDA_ERROR_DEINITIALIZED=4, CUDA_ERROR_NO_DEVICE=100, CUDA_ERROR_INVALID_CONTEXT=201, CUDA_ERROR_INVALID_HANDLE=400, CUDA_ERROR_NOT_FOUND=500, CUDA_ERROR_NOT_READY=600, CUDA_ERROR_LAUNCH_FAILED=719, CUDA_ERROR_UNKNOWN=999 } CUresult;
typedef hipCtx_t CUcontext;
typedef hipStream_t CUstream;
typedef hipEvent_t CUevent;
typedef hipFunction_t CUfunction;
typedef hipModule_t CUmodule;
typedef hipDeviceptr_t CUdeviceptr;
#define cuGetErrorName hipGetErrorName
#define cuGetErrorString hipGetErrorString
#define cuCtxGetDevice hipCtxGetDevice
#define cuCtxPushCurrent hipCtxPushCurrent
#define cuCtxPopCurrent hipCtxPopCurrent
#define cuCtxSynchronize hipCtxSynchronize
#define cuDeviceGetCount hipGetDeviceCount
#define cuDeviceGet hipDeviceGet
#define cuInit hipInit
#define cuDriverGetVersion hipDriverGetVersion
#define cuStreamGetCtx hipStreamGetCtx
HEOF
      cat > "$COMPAT_OURS/cublas_v2.h" <<'HEOF'
#pragma once
/* Bridge header (ours): opaque types only, so torch's ROCm headers parse
   on a box with no CUDA toolkit. nvdiffrast never calls cuBLAS. */
#include <cuda_runtime_api.h>
typedef enum cublasStatus_t { CUBLAS_STATUS_SUCCESS=0, CUBLAS_STATUS_NOT_INITIALIZED=1, CUBLAS_STATUS_ALLOC_FAILED=3, CUBLAS_STATUS_INVALID_VALUE=7, CUBLAS_STATUS_ARCH_MISMATCH=8, CUBLAS_STATUS_MAPPING_ERROR=11, CUBLAS_STATUS_EXECUTION_FAILED=13, CUBLAS_STATUS_INTERNAL_ERROR=14, CUBLAS_STATUS_NOT_SUPPORTED=15, CUBLAS_STATUS_LICENSE_ERROR=16 } cublasStatus_t;
typedef enum cublasOperation_t { CUBLAS_OP_N=0, CUBLAS_OP_T=1, CUBLAS_OP_C=2, CUBLAS_OP_HERMITAN=3, CUBLAS_OP_CONJG=4 } cublasOperation_t;
typedef enum cublasPointerMode_t { CUBLAS_POINTER_MODE_HOST=0, CUBLAS_POINTER_MODE_DEVICE=1 } cublasPointerMode_t;
typedef enum cublasAtomicsMode_t { CUBLAS_ATOMICS_NOT_ALLOWED=0, CUBLAS_ATOMICS_ALLOWED=1 } cublasAtomicsMode_t;
typedef enum cublasMathMode_t { CUBLAS_DEFAULT_MATH=0, CUBLAS_TENSOR_OP_MATH=1, CUBLAS_TENSOR_OP_MATH_ALLOW_CONVERSION=2 } cublasMathMode_t;
typedef cudaDataType cublasDataType_t;
typedef enum cublasComputeType_t { CUBLAS_COMPUTE_32F=0, CUBLAS_COMPUTE_64F=1, CUBLAS_COMPUTE_16F=2 } cublasComputeType_t;
struct cublasContext; typedef struct cublasContext *cublasHandle_t;
HEOF
      cat > "$COMPAT_OURS/cublasLt.h" <<'HEOF'
#pragma once
/* Bridge header (ours): opaque types only. */
#include <cublas_v2.h>
struct cublasLtContext; typedef struct cublasLtContext *cublasLtHandle_t;
struct cublasLtMatmulDescOpaque; typedef struct cublasLtMatmulDescOpaque *cublasLtMatmulDesc_t;
struct cublasLtMatrixLayoutOpaque; typedef struct cublasLtMatrixLayoutOpaque *cublasLtMatrixLayout_t;
struct cublasLtMatmulPreferenceOpaque; typedef struct cublasLtMatmulPreferenceOpaque *cublasLtMatmulPreference_t;
struct cublasLtMatrixTransformDescOpaque; typedef struct cublasLtMatrixTransformDescOpaque *cublasLtMatrixTransformDesc_t;
typedef enum cublasLtStatus_t { CUBLASLT_STATUS_SUCCESS=0 } cublasLtStatus_t;
HEOF
      cat > "$COMPAT_OURS/cusparse.h" <<'HEOF'
#pragma once
/* Bridge header (ours): opaque types only. */
typedef enum cusparseStatus_t { CUSPARSE_STATUS_SUCCESS=0, CUSPARSE_STATUS_NOT_INITIALIZED=1, CUSPARSE_STATUS_ALLOC_FAILED=2, CUSPARSE_STATUS_INVALID_VALUE=3, CUSPARSE_STATUS_ARCH_MISMATCH=4, CUSPARSE_STATUS_MAPPING_ERROR=5, CUSPARSE_STATUS_EXECUTION_FAILED=6, CUSPARSE_STATUS_INTERNAL_ERROR=7, CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8, CUSPARSE_STATUS_ZERO_PIVOT=9, CUSPARSE_STATUS_NOT_SUPPORTED=10, CUSPARSE_STATUS_INSUFFICIENT_RESOURCES=11 } cusparseStatus_t;
struct cusparseContext; typedef struct cusparseContext *cusparseHandle_t;
typedef enum cusparseDirection_t { CUSPARSE_DIRECTION_ROW=0, CUSPARSE_DIRECTION_COLUMN=1 } cusparseDirection_t;
typedef enum cusparseMatrixType_t { CUSPARSE_MATRIX_TYPE_GENERAL=0, CUSPARSE_MATRIX_TYPE_SYMMETRIC=1, CUSPARSE_MATRIX_TYPE_HERMITIAN=2, CUSPARSE_MATRIX_TYPE_TRIANGULAR=3 } cusparseMatrixType_t;
typedef enum cusparseFillMode_t { CUSPARSE_FILL_MODE_LOWER=0, CUSPARSE_FILL_MODE_UPPER=1, CUSPARSE_FILL_MODE_FULL=2 } cusparseFillMode_t;
typedef enum cusparseDiagType_t { CUSPARSE_DIAG_TYPE_NON_UNIT=0, CUSPARSE_DIAG_TYPE_UNIT=1 } cusparseDiagType_t;
typedef enum cusparseIndexBase_t { CUSPARSE_INDEX_BASE_ZERO=0, CUSPARSE_INDEX_BASE_ONE=1 } cusparseIndexBase_t;
typedef enum cusparseOperation_t { CUSPARSE_OPERATION_NON_TRANSPOSE=0, CUSPARSE_OPERATION_TRANSPOSE=1, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE=2 } cusparseOperation_t;
struct cusparseMatDescr; typedef struct cusparseMatDescr *cusparseMatDescr_t;
struct cusparseSpMatDescr; typedef struct cusparseSpMatDescr *cusparseSpMatDescr_t;
struct cusparseDnMatDescr; typedef struct cusparseDnMatDescr *cusparseDnMatDescr_t;
struct cusparseDnVecDescr; typedef struct cusparseDnVecDescr *cusparseDnVecDescr_t;
HEOF
      cat > "$COMPAT_OURS/cusolverDn.h" <<'HEOF'
#pragma once
/* Bridge header (ours): opaque types only. */
typedef enum cusolverStatus_t { CUSOLVER_STATUS_SUCCESS=0, CUSOLVER_STATUS_NOT_INITIALIZED=1, CUSOLVER_STATUS_ALLOC_FAILED=2, CUSOLVER_STATUS_INVALID_VALUE=3, CUSOLVER_STATUS_ARCH_MISMATCH=4, CUSOLVER_STATUS_MAPPING_ERROR=5, CUSOLVER_STATUS_EXECUTION_FAILED=6, CUSOLVER_STATUS_INTERNAL_ERROR=7, CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED=8, CUSOLVER_STATUS_NOT_SUPPORTED=9, CUSOLVER_STATUS_ZERO_PIVOT=10, CUSOLVER_STATUS_INVALID_LICENSE=11 } cusolverStatus_t;
struct cusolverDnContext; typedef struct cusolverDnContext *cusolverDnHandle_t;
typedef enum cusolverEigType_t { CUSOLVER_EIG_TYPE_1=1, CUSOLVER_EIG_TYPE_2=2, CUSOLVER_EIG_TYPE_3=3 } cusolverEigType_t;
typedef enum cusolverEigMode_t { CUSOLVER_EIG_MODE_NOVECTOR=0, CUSOLVER_EIG_MODE_VECTOR=1 } cusolverEigMode_t;
HEOF
      cat > "$COMPAT_OURS/cudss.h" <<'HEOF'
#pragma once
/* Bridge header (ours): opaque types only. */
typedef enum cudssStatus_t { CUDSS_STATUS_SUCCESS=0, CUDSS_STATUS_NOT_INITIALIZED=1, CUDSS_STATUS_ALLOC_FAILED=2, CUDSS_STATUS_INVALID_VALUE=3, CUDSS_STATUS_NOT_SUPPORTED=4, CUDSS_STATUS_EXECUTION_FAILED=5, CUDSS_STATUS_INTERNAL_ERROR=6 } cudssStatus_t;
struct cudssContext; typedef struct cudssContext *cudssHandle_t;
struct cudssDataOpaque; typedef struct cudssDataOpaque *cudssData_t;
struct cudssConfigOpaque; typedef struct cudssConfigOpaque *cudssConfig_t;
struct cudssMatrixOpaque; typedef struct cudssMatrixOpaque *cudssMatrix_t;
HEOF
    # Snapshot the working nvdiffrast: his setup uninstalls before it
    # installs, so a failed E2 must never leave the venv without one (r10).
    SNAP="$ROOT/nvdiffrast-snapshot"; rm -rf "$SNAP"; mkdir -p "$SNAP"
    SP_DIR=$(python3 -c "import nvdiffrast, os; print(os.path.dirname(os.path.dirname(nvdiffrast.__file__)))" 2>/dev/null || echo "")
    if [ -n "$SP_DIR" ] && [ -d "$SP_DIR/nvdiffrast" ]; then
      cp -a "$SP_DIR/nvdiffrast" "$SNAP/" 2>/dev/null || true
      cp -a "$SP_DIR"/nvdiffrast-*.dist-info "$SNAP/" 2>/dev/null || true
      cp -a "$SP_DIR"/_nvdiffrast_c*.so "$SNAP/" 2>/dev/null || true
      log "Phase E2: working nvdiffrast snapshotted - a failed texture install restores it."
    fi
    # r13: his baseline step expects a PRISTINE nvdiffrast work tree, but a
    # previous attempt leaves it fully patched (r12's log: his own driver
    # warns "working tree has uncommitted or untracked changes") and r12's
    # baseline then died in seconds. Reset the tree to HEAD in place when
    # it is a git checkout (no network round-trip); if that is impossible,
    # remove it and let his setup re-clone. Generated bundle dirs go too -
    # his generator recreates them every attempt.
    NV_SRC="$ROOT/painter-work/nvdiffrast"
    if [ -d "$NV_SRC/.git" ] && git -C "$NV_SRC" checkout -- . 2>/dev/null && git -C "$NV_SRC" clean -fdx >/dev/null 2>&1; then
      log "Phase E2: nvdiffrast work tree reset to pristine (git checkout + clean)."
    elif [ -d "$NV_SRC" ]; then
      rm -rf "$NV_SRC"
      log "Phase E2: removed the patched nvdiffrast work tree - his setup will re-clone it pristine."
    fi
    rm -rf "$ROOT/painter-work"/nvdiffrast_rocm72_gfx1201_*_bundle 2>/dev/null || true
    log "Phase E2: our compat headers written - adding them to the include path for his builds."
    (cd "$ROOT/painter3000" && CPLUS_INCLUDE_PATH="$COMPAT_OURS${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}" python3 amd_nvdiffrast_setup.py --workdir "$ROOT/painter-work" --venv "$ROOT/venv" --arch gfx1201 --max-jobs 8 --validation quick) \
      || { log "E2-FAIL: Painter3000 nvdiffrast setup did not complete - continuing without textures."
           # r13 diagnosis: his driver surfaces a failed child step as a bare
           # CalledProcessError - the step's own output never reaches this
           # log (r12's baseline died in 17 seconds, silently). Re-run the
           # baseline reinstall script DIRECTLY so its real error (or its
           # success) lands here unfiltered. Diagnostic only: this run still
           # counts E2 as failed; the snapshot restore below still guards
           # the venv against anything this re-run leaves half-installed.
           DIAG_BUNDLE="$ROOT/painter-work/nvdiffrast_rocm72_gfx1201_final_v52_bundle"
           if [ -f "$DIAG_BUNDLE/reinstall_nvdiffrast_rocm72_gfx1201.sh" ]; then
             log "Phase E2 diagnosis: re-running the baseline reinstall directly, with his driver's environment replicated - its unfiltered output follows."
             (cd "$DIAG_BUNDLE" && env ROOT="$ROOT/painter-work" REPO="$ROOT/painter-work/nvdiffrast" VENV="$ROOT/venv" BUNDLE_DIR="$DIAG_BUNDLE" ROCM_PATH="/opt/rocm" MODE="runtime" PYTORCH_ROCM_ARCH="gfx1201" FORCE_CUDA="1" MAX_JOBS="8" CPPFLAGS="-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE" CPATH="$ROOT/painter-work/nvdiffrast_rocm_cuda_compat:/opt/rocm/include/hipsparse" CC="/opt/rocm/llvm/bin/clang" CXX="/opt/rocm/llvm/bin/clang++" CPLUS_INCLUDE_PATH="$COMPAT_OURS${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}" bash reinstall_nvdiffrast_rocm72_gfx1201.sh) \
               && log "Phase E2 diagnosis: the baseline reinstall SUCCEEDED run directly - the failure lives in his driver's orchestration, not in the build itself." \
               || log "Phase E2 diagnosis: the baseline reinstall failed run directly too - the error printed above is the real one."
           else
             log "Phase E2 diagnosis: no baseline reinstall script on disk - his driver failed before the baseline step."
           fi
           P_OK=0; }
  fi
  if [ "$P_OK" = 1 ]; then
    raster_gate || { log "E2-FAIL: Painter3000 nvdiffrast installed, but a CUDA-context rasterize still does not run - continuing without textures."; P_OK=0; }
  fi
  if [ "$P_OK" = 0 ] && [ -n "${SNAP:-}" ] && [ -d "$SNAP/nvdiffrast" ] && ! python3 -c "import nvdiffrast" 2>/dev/null; then
    SP_LIB=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    rm -rf "$SP_LIB/nvdiffrast" "$SP_LIB"/nvdiffrast-*.dist-info "$SP_LIB"/_nvdiffrast_c*.so
    cp -a "$SNAP/nvdiffrast" "$SP_LIB/" 2>/dev/null || true
    cp -a "$SNAP"/nvdiffrast-*.dist-info "$SP_LIB/" 2>/dev/null || true
    cp -a "$SNAP"/_nvdiffrast_c*.so "$SP_LIB/" 2>/dev/null || true
    log "Phase E2: restored the snapshotted working nvdiffrast - generation stays shape-only but WORKING."
  fi
  if [ "$P_OK" = 1 ]; then
    rm -f "$MARK/painter.failed"; touch "$MARK/painter.done"; log "Phase E2 done: the CUDA-context rasterizer runs on the 9070 XT."
  else
    echo "stamp $BUILD_STAMP: E2 failed $(date -Iseconds) - see the run log; generation continues shape-only" > "$MARK/painter.failed"
  fi
fi

fan_hook off
# ---------------------------------------------------------------- Phase F: model weights
export HF_HOME="$HF_DIR"
if [ -f "$MARK/models.done" ]; then
  log "Phase F (model weights) already done - skipping."
else
  banner "Phase F: download model weights - sizes announced BEFORE the download"
  log "  microsoft/TRELLIS.2-4B .......... 16.3 GB (computed from Hugging Face file metadata)"
  log "  DINOv3 conditioner + BiRefNet ... ~2 GB more, fetched by the pipeline at first load"
  log "Total this phase: roughly 16.3 GB now. Watch the counter; this is download-bound."
  with_heartbeat "TRELLIS.2-4B download" python3 - <<'EOF' \
    || { log "FAIL: model download"; exit 60; }
from huggingface_hub import snapshot_download
p = snapshot_download("microsoft/TRELLIS.2-4B")
print("weights at:", p)
EOF
  touch "$MARK/models.done"
fi

# ---------------------------------------------------------------- Phase G: corruption reproducer (informational, logged)
banner "Phase G: fp32 GEMM corruption reproducer (known gfx1201 bug, ROCm issue 6595)"
log "This script DEMONSTRATES the upstream bug on unchunked large fp32 GEMMs. It is"
log "expected to show corruption - that is the bug existing, not our port failing."
log "The pipeline is protected by the fork's chunked-GEMM patch, and Phase H's output"
log "gets NaN/structure scans. Full reproducer output is in the log either way."
if [ -f "$ROOT/port-patches/repro_sgemm_fp32_gfx1201.py" ]; then
  python3 "$ROOT/port-patches/repro_sgemm_fp32_gfx1201.py" 2>&1 | tail -20 || true
else
  log "WARN: reproducer file not found in the patches repo (layout changed?) - continuing."
fi

# ---------------------------------------------------------------- r30 source capture
# Copy the pipeline + extraction Python sources that build the mesh
# into this run's output folder, so the extraction hunt reads the
# actual code instead of inferring it (HUNT-NOTE-3: three rounds were
# spent inferring code that could have been read). Read-only,
# best-effort, never fatal to the run.
CAP="$OUT/source-capture"
mkdir -p "$CAP" 2>/dev/null || true
{
  echo "captured $(date -Iseconds) stamp $BUILD_STAMP"
  for _src in "$ROOT/TRELLIS.2/trellis2/pipelines" "$ROOT/TRELLIS.2/trellis2/models" "$ROOT/TRELLIS.2/trellis2/representations" "$ROOT/TRELLIS.2/o-voxel"; do
    if [ -d "$_src" ]; then
      find "$_src" -name "*.py" 2>/dev/null | while read -r _f; do
        _rel=$(echo "$_f" | sed "s|^$ROOT/TRELLIS.2/||; s|/|__|g")
        cp "$_f" "$CAP/$_rel" 2>/dev/null && echo "OK  $_f"
      done
    else
      echo "MISSING DIR $_src"
    fi
  done
} > "$CAP/CAPTURE-MANIFEST.txt" 2>/dev/null || true
log "r30 source capture: $(ls "$CAP" 2>/dev/null | wc -l) files in output/source-capture"

# r31: ALSO ship the capture as ONE archive in the output root. In
# r30 the loose folder reached Drive EMPTY (44 files captured on the
# PC, the worker collect step carried 0 MB of them) - single files in
# the output root sync reliably, loose subfolder files did not. The
# archive is the copy the hunt reads; the folder stays for on-PC
# inspection. Best-effort like the capture itself: zip if present,
# else python's shutil; a failure logs WARN and never stops the run.
if [ -d "$CAP" ]; then
  _r31_zip_ok=0
  if command -v zip >/dev/null 2>&1; then
    (cd "$OUT" && zip -q -r "source-capture.zip" "source-capture") && _r31_zip_ok=1
  fi
  if [ "$_r31_zip_ok" != "1" ]; then
    python3 -c "import shutil, sys; shutil.make_archive(sys.argv[1], 'zip', root_dir=sys.argv[2], base_dir='source-capture')" "$OUT/source-capture" "$OUT" 2>/dev/null && _r31_zip_ok=1
  fi
  if [ "$_r31_zip_ok" = "1" ] && [ -f "$OUT/source-capture.zip" ]; then
    log "r31 source capture archive: output/source-capture.zip ($(du -h "$OUT/source-capture.zip" | cut -f1), $(ls "$CAP" | wc -l) captured files inside)"
  else
    log "WARN: r31 source-capture archive could not be created - loose folder only; the Drive copy may again arrive empty"
  fi
fi

# ---------------------------------------------------------------- Phase H: smoke generation (canvas @512)
RUN_ENV="ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa SPARSE_CONV_BACKEND=flex_gemm TORCH_BLAS_PREFER_HIPBLASLT=0"
GEN="$PKG_WSL/inside/generate.py"
CHK="$PKG_WSL/inside/check_glb.py"
CANVAS_SRC="$PKG_WSL/input/canvas-front.webp"
cp "$CANVAS_SRC" "$ROOT/canvas-front.webp" || { log "FAIL: canvas input missing at $CANVAS_SRC"; exit 60; }

# r29 DECLOTH TEST, stage 1 (Cory's design, Oct 25): if a declothed canvas
# plate sits in the package input, it replaces the clothed canvas as the
# conditioning image for BOTH canvas phases (smoke + cascade) this run -
# the figure is generated from the plain clay body instead of the clothed
# plate, so garment shapes stop fusing into her silhouette. Stage 1 is
# geometry-first: the texture pass conditions on the same declothed plate
# (plain clay paint) - garment paint returning via a split texture pass is
# stage 2, pending a read of the pipeline's stage API. INERT when the file
# is absent: canvas behavior is then identical to r28. The marker file in
# the output makes the substitution impossible to miss at grade time.
for _dc in "$PKG_WSL/input/canvas-front-decloth.webp" "$PKG_WSL/input/canvas-front-decloth.png" "$PKG_WSL/input/canvas-front-decloth.jpg"; do
  if [ -f "$_dc" ]; then
    cp "$_dc" "$ROOT/canvas-front.webp" || { log "FAIL: decloth plate copy failed"; exit 60; }
    log "r29 DECLOTH TEST ACTIVE: canvas conditioned on $(basename "$_dc") (plain clay body); clothed canvas-front.webp NOT used this run"
    echo "This run's canvas was generated from the DECLOTHED plate (plain clay body) - r29 stage-1 decloth test." > "$OUT/DECLOTH-TEST.txt"
    break
  fi
done

# ---------------------------------------------------------------- Phase H prep: runtime deps
# TRELLIS.2's own setup.sh dependency line, minus torch (Phase C pins it),
# gradio/tensorboard (demo/training only), pillow-simd and opencv-headless
# (cv2 and Pillow are already present). Needed because egore's installer
# pulled transformers 5.2.0 WITHOUT its 'regex' import dependency - the
# 2026-10-24 13:39 run died at the smoke generation on exactly that.
# r33.4: probe-first (was: one plain pip install of the whole line.
# "Idempotent: pip no-ops whatever is already satisfied" is exactly
# the trap job 032 died on - for a sweep-damaged package, pip's
# no-op means the missing files stay missing). Each entry below is
# "install spec|distribution|import module": if the module imports,
# the package is skipped outright; if the distribution is installed
# but its module does not import, it is force-reinstalled at the
# installed version (--no-deps); only a genuinely absent package
# gets a plain install. Failures stay WARN - the chase below names
# what is actually missing - as before.
banner "Phase H prep: TRELLIS.2 runtime dependency top-up"
dep_ensure() {
  local spec="$1" dist="$2" mod="$3" ver=""
  if python3 -c "import $mod" >/dev/null 2>&1; then
    log "Phase H prep [r33.4]: '$mod' imports - package '$dist' whole, skipping."
    return 0
  fi
  ver=$(python3 -c "import importlib.metadata as m, sys
try:
    print(m.version(sys.argv[1]))
except Exception:
    pass" "$dist" 2>/dev/null)
  if [ -n "$ver" ]; then
    log "Phase H prep [r33.4]: module '$mod' does not import but '$dist' $ver is installed - force-reinstalling the pinned version (a plain install would no-op on the intact metadata)."
    with_heartbeat "dep $dist" pip install --force-reinstall --no-deps "$dist==$ver" \
      || log "WARN: force-reinstall of $dist reported problems (continuing; the chase below will name what is missing)"
  else
    log "Phase H prep [r33.4]: module '$mod' missing and '$dist' not installed - installing '$spec'."
    with_heartbeat "dep $dist" pip install "$spec" \
      || log "WARN: install of $spec reported problems (continuing; the chase below will name what is missing)"
  fi
}
while IFS='|' read -r DEPSPEC DEPDIST DEPMOD; do
  [ -n "$DEPSPEC" ] || continue
  dep_ensure "$DEPSPEC" "$DEPDIST" "$DEPMOD"
done <<'DEP_EOF'
regex|regex|regex
httpx|httpx|httpx
einops|einops|einops
huggingface-hub>=1.3.0,<2.0|huggingface-hub|huggingface_hub
imageio|imageio|imageio
imageio-ffmpeg|imageio-ffmpeg|imageio_ffmpeg
easydict|easydict|easydict
trimesh|trimesh|trimesh
pandas|pandas|pandas
lpips|lpips|lpips
zstandard|zstandard|zstandard
kornia|kornia|kornia
timm|timm|timm
tqdm|tqdm|tqdm
DEP_EOF

# Dependency chase: importing the pipeline names each still-missing module;
# repair it and retry (bounded at 15 rounds). This kills the
# one-module-per-rerun whack-a-mole as a class (regex, then httpx,
# 2026-10-24) - the import chain of the CURRENT library versions is longer
# than any static list I can guess.
# r33.4: probe-first, force-reinstall, per-module cap of 2. Job 032 spun
# this loop 15 times on ONE module: the sweep had deleted idna/core.py but
# left the distribution metadata intact, so every plain `pip install idna`
# answered "already satisfied" (3.20), changed nothing, and the loop
# burned all 15 attempts before its guard exited 60. Now the named module
# (the FULL dotted path - idna.core, not just idna) is import-probed
# directly before anything is installed, and the install is skipped when
# it imports; a broken-but-installed package is force-reinstalled at its
# installed version (--no-deps), never a plain install; and the same
# module gets at most 2 repair attempts before the loop exits loud
# naming it, instead of spinning to 15.
declare -A CHASE_TRIES=()
for attempt in $(seq 1 15); do
  PROBE_OUT=$(env $RUN_ENV python3 -c "from trellis2.pipelines import Trellis2ImageTo3DPipeline" 2>&1) && { log "Import probe: pipeline imports clean."; break; }
  FULLMOD=$(printf '%s' "$PROBE_OUT" | grep -oE "No module named '[a-zA-Z0-9_.]+'" | head -1 | sed "s/No module named '//; s/'//")
  MOD=$(printf '%s' "$FULLMOD" | cut -d. -f1)
  if [ -z "$MOD" ]; then
    # Not a missing module - maybe a version conflict, which transformers
    # reports as "pkg>=a,<b is required ... but found pkg==v". Install the
    # stated range and retry; anything else falls through to the generation.
    REQ=$(printf '%s' "$PROBE_OUT" | grep -oE "[a-zA-Z0-9_.-]+(>=|<=|==|>|<)[0-9][0-9a-zA-Z.,<>= ]* is required" | head -1 | sed 's/ is required//')
    if [ -n "$REQ" ]; then
      PKGSPEC=$(printf '%s' "$REQ" | tr -d ' ')
      log "Import probe: version conflict - installing '$PKGSPEC' (attempt $attempt of 15)"
      pip install "$PKGSPEC" || { log "FAIL: could not install $PKGSPEC"; exit 60; }
      if [ "$attempt" = "15" ]; then log "FAIL: import probe still failing after 15 repairs"; exit 60; fi
      continue
    fi
    log "Import probe failed for a reason that is not a missing module or a version conflict - last lines:"
    printf '%s\n' "$PROBE_OUT" | tail -15
    break
  fi
  case "$MOD" in
    PIL) PKG=Pillow ;;
    cv2) PKG=opencv-python ;;
    yaml) PKG=PyYAML ;;
    sklearn) PKG=scikit-learn ;;
    *) PKG="$MOD" ;;
  esac
  CHASE_TRIES[$MOD]=$(( ${CHASE_TRIES[$MOD]:-0} + 1 ))
  if [ "${CHASE_TRIES[$MOD]}" -gt 2 ]; then
    log "FAIL: module '$MOD' (package '$PKG') is still named by the pipeline import after 2 repair attempts - refusing to spin (job 032's 15x satisfied-loop is the failure this replaces). Last probe lines:"
    printf '%s\n' "$PROBE_OUT" | tail -15
    exit 60
  fi
  if python3 -c "import $FULLMOD" >/dev/null 2>&1; then
    log "Import probe: module '$FULLMOD' imports clean on its own (repair attempt ${CHASE_TRIES[$MOD]} of 2 for '$MOD') - no install can fix a contextual failure; re-probing the pipeline."
    if [ "$attempt" = "15" ]; then log "FAIL: import probe still failing after 15 repairs"; exit 60; fi
    continue
  fi
  INSTVER=$(python3 -c "import importlib.metadata as m
try:
    print(m.version('$PKG'))
except Exception:
    pass" 2>/dev/null)
  if [ -n "$INSTVER" ]; then
    log "Import probe: module '$FULLMOD' does not import but package '$PKG' $INSTVER is installed - force-reinstalling the pinned version (repair attempt ${CHASE_TRIES[$MOD]} of 2 for '$MOD')"
    pip install --force-reinstall --no-deps "$PKG==$INSTVER" || { log "FAIL: could not force-reinstall $PKG for module $MOD"; exit 60; }
  else
    log "Import probe: module '$FULLMOD' missing and package '$PKG' is not installed - installing it (repair attempt ${CHASE_TRIES[$MOD]} of 2 for '$MOD')"
    pip install "$PKG" || { log "FAIL: could not install $PKG for module $MOD"; exit 60; }
  fi
  if [ "$attempt" = "15" ]; then log "FAIL: import probe still failing after 15 repairs"; printf '%s\n' "$PROBE_OUT" | tail -15; exit 60; fi
done

# ---------------------------------------------------------------- Phase H prep census gate (r33.6)
# ROOT FIX for job 035's failure class (see the header and
# DIAGNOSIS-035.md): the Phase C census stands BEFORE Phase E and
# this prep - the phases whose installers populate the broad
# dependency set. On job 034's fresh-venv run it measured the
# small fresh venv (whole) and passed; Phase E's egore installer
# then landed the broad set WITHOUT its full closure (open3d
# without dash, requests without urllib3, rembg whose onnxruntime
# backend ships only in its [cpu] extra), nothing re-measured the
# venv afterwards, and the holes ambushed job 035's census. This
# gate re-runs the census AFTER every installer in the run has
# run, with the same two heal classes (file-damage --no-deps
# force-reinstall, then the dependency rung), and exits 60 loud
# if anything is still broken - a generation never starts on a
# venv that merely looked whole back in Phase C.
banner "Phase H prep census (r33.6): re-measure the whole venv AFTER every installer has run"
GATE_FAIL="$MARK/census-gate-fail.tsv"
GATE_EXCL="$MARK/census-gate-excl.tsv"
GATE_STILL="$MARK/census-gate-still.tsv"
GATE_STILL2="$MARK/census-gate-still2.tsv"
: > "$GATE_FAIL"; : > "$GATE_EXCL"; : > "$GATE_STILL"; : > "$GATE_STILL2"
if ! python3 "$MARK/venv_census.py" probe-all "$GATE_FAIL" "$GATE_EXCL"; then
  log "FAIL: the r33.6 post-prep census probe itself errored - refusing to generate on an unmeasured venv."
  exit 60
fi
GATE_NFAIL=$(grep -c . "$GATE_FAIL" || true)
if [ "$GATE_NFAIL" -eq 0 ]; then
  log "r33.6 post-prep census: clean - every healable package imports whole AFTER all installers ran. No reinstalls needed."
else
  log "r33.6 post-prep census: $GATE_NFAIL package(s) fail AFTER the installers ran - healing in place (file-damage class first: force-reinstall at the installed version, --no-deps)."
  while IFS=$'\t' read -r PKGNAME PKGVER; do
    [ -n "$PKGNAME" ] || continue
    if with_heartbeat "r33.6 gate heal $PKGNAME" pip install --force-reinstall --no-deps "$PKGNAME==$PKGVER" < /dev/null; then
      log "r33.6 post-prep census: reinstalled $PKGNAME==$PKGVER - re-probe below decides whether it healed."
    else
      log "WARN: r33.6 post-prep census could not reinstall $PKGNAME==$PKGVER (pip output above); the dependency rung and the re-probe below decide."
    fi
  done < "$GATE_FAIL"
  if ! python3 "$MARK/venv_census.py" probe-list "$GATE_FAIL" "$GATE_STILL"; then
    log "FAIL: the r33.6 post-prep census re-probe itself errored - refusing to generate."
    exit 60
  fi
  GATE_FINAL="$GATE_STILL"
  if [ -s "$GATE_STILL" ]; then
    log "r33.6 gate deps-heal: survivors of the file-damage heal get the dependency-class rung (same ladder as the Phase C census rung: named-module installs cap 2, r33.7 whole-closure pass for chains/extras-gated, force-reinstall-with-deps last)."
    while IFS=$'\t' read -r PKGNAME PKGVER; do
      [ -n "$PKGNAME" ] || continue
      HAMMERED=0
      CLOSURED=0
      NAMED_TRIES=0
      FIRSTMISS=""
      CLOSURE_EXTRAS=""
      case "$PKGNAME" in
        rembg) CLOSURE_EXTRAS="cpu" ;;
      esac
      while :; do
        MISSFILE="$MARK/.census-miss.txt"
        census_pkg_missing "$PKGNAME" > "$MISSFILE"
        MISSRC=$?
        MISS=$(head -n 1 "$MISSFILE" | sed 's/[^A-Za-z0-9_.]//g')
        rm -f "$MISSFILE"
        if [ "$MISSRC" -eq 1 ]; then
          log "r33.6 gate deps-heal: '$PKGNAME' imports whole now - the dependency rung is done with it (the re-probe below is the verdict)."
          break
        fi
        if [ "$CLOSURED" -eq 0 ] && { [ -n "$CLOSURE_EXTRAS" ] || { [ "$MISSRC" -eq 0 ] && [ -n "$MISS" ] && [ -n "$FIRSTMISS" ] && [ "$MISS" != "$FIRSTMISS" ]; }; }; then
          CLOSURED=1
          if [ -n "$CLOSURE_EXTRAS" ]; then
            CLOSURE_TARGET="$PKGNAME[$CLOSURE_EXTRAS]"
            CLOSURE_WHY="'$PKGNAME' is in the extras-gated table ([$CLOSURE_EXTRAS]) - its missing pieces ship behind the extra, so chasing named modules can never finish it"
          else
            CLOSURE_TARGET="$PKGNAME"
            CLOSURE_WHY="'$PKGNAME' named a SECOND distinct missing module ('$FIRSTMISS', now '$MISS') - the chain signature; hand-chasing modules reinvents pip's resolver one link at a time (job 036: rembg onnxruntime -> pymatting -> scikit-image, cap spent mid-chain)"
          fi
          log "[r33.7 closure-heal] $CLOSURE_WHY - running ONE resolver pass: pip install \"$CLOSURE_TARGET\" (plain install, NOT force-reinstall: the package's own files are whole, its closure is incomplete; a plain install fills the missing members and leaves satisfied and pinned distributions alone)."
          CLOUT="$MARK/.closure-out.txt"
          : > "$CLOUT"
          with_heartbeat "r33.7 closure $PKGNAME" bash -c 'set -o pipefail; pip install "$1" 2>&1 | tee "$2"' _ "$CLOSURE_TARGET" "$CLOUT" < /dev/null \
            || log "WARN: [r33.7 closure-heal] the resolver pass for $CLOSURE_TARGET reported problems (continuing; the re-probe decides)."
          CLOSEDIST=$(grep -E "^Installing collected packages:" "$CLOUT" | tail -n 1 | sed "s/^Installing collected packages: //")
          rm -f "$CLOUT"
          log "[r33.7 closure-heal] '$PKGNAME' closure pass - distributions pip installed or changed: ${CLOSEDIST:-<none - pip printed no 'Installing collected packages' line>}"
          if torch_probe; then
            log "[r33.7 closure-heal] torch probe after the '$PKGNAME' closure pass: clean - the pinned stack stood (any drift would be named in the changed-distributions line above, never silent)."
          else
            log "FAIL: [r33.7 closure-heal] the torch probe FAILS after the '$PKGNAME' closure pass - the closure pass is the suspect (distributions it changed: ${CLOSEDIST:-unknown}). Refusing to continue on a disturbed pinned stack."
            exit 60
          fi
          continue
        fi
        if [ "$MISSRC" -eq 0 ] && [ -n "$MISS" ]; then
          if [ "$NAMED_TRIES" -ge 2 ]; then
            log "r33.6 gate deps-heal: '$PKGNAME' - both named-module attempts are spent and it still names missing module '$MISS'. Leaving it for the verdict below."
            break
          fi
          NAMED_TRIES=$((NAMED_TRIES + 1))
          if [ -z "$FIRSTMISS" ]; then FIRSTMISS="$MISS"; fi
          case "$MISS" in
            PIL) DEPSDIST=Pillow ;;
            cv2) DEPSDIST=opencv-python ;;
            yaml) DEPSDIST=PyYAML ;;
            sklearn) DEPSDIST=scikit-learn ;;
            *) DEPSDIST="$MISS" ;;
          esac
          if python3 -c "import importlib.metadata as m, sys; m.version(sys.argv[1])" "$DEPSDIST" >/dev/null 2>&1; then
            DEPSVER=$(python3 -c "import importlib.metadata as m, sys; print(m.version(sys.argv[1]))" "$DEPSDIST")
            log "r33.6 gate deps-heal: '$PKGNAME' names missing module '$MISS', but distribution '$DEPSDIST' IS installed - the damage is inside it; force-reinstalling '$DEPSDIST==$DEPSVER' (--no-deps: its own files are the problem). Attempt $NAMED_TRIES of 2 for '$PKGNAME'."
            with_heartbeat "r33.6 gate deps $PKGNAME" pip install --force-reinstall --no-deps "$DEPSDIST==$DEPSVER" < /dev/null \
              || log "WARN: r33.6 gate deps-heal force-reinstall of $DEPSDIST reported problems (continuing; the re-probe decides)."
          else
            log "r33.6 gate deps-heal: '$PKGNAME' names missing module '$MISS' and distribution '$DEPSDIST' is not installed - installing it WITH its dependencies. Attempt $NAMED_TRIES of 2 for '$PKGNAME'."
            with_heartbeat "r33.6 gate deps $PKGNAME" pip install "$DEPSDIST" < /dev/null \
              || log "WARN: r33.6 gate deps-heal install of $DEPSDIST reported problems (continuing; the re-probe decides)."
          fi
        elif [ "$HAMMERED" -eq 0 ]; then
          HAMMERED=1
          log "r33.6 gate deps-heal: '$PKGNAME' names no single missing module - force-reinstalling '$PKGNAME==$PKGVER' WITH dependencies so pip resolves its whole declared closure (last-resort step for '$PKGNAME')."
          with_heartbeat "r33.6 gate deps $PKGNAME" pip install --force-reinstall "$PKGNAME==$PKGVER" < /dev/null \
            || log "WARN: r33.6 gate deps-heal force-reinstall of $PKGNAME reported problems (continuing; the re-probe decides)."
        else
          log "r33.6 gate deps-heal: '$PKGNAME' - the dependency ladder is spent (named-module cap, closure pass, force-reinstall) and it still names no single missing module. Leaving it for the verdict below."
          break
        fi
      done
    done < "$GATE_STILL"
    if ! python3 "$MARK/venv_census.py" probe-list "$GATE_STILL" "$GATE_STILL2"; then
      log "FAIL: the r33.6 gate deps-heal re-probe itself errored - refusing to generate."
      exit 60
    fi
    GATE_FINAL="$GATE_STILL2"
  fi
  GATE_NSTILL=$(grep -c . "$GATE_FINAL" || true)
  log "r33.6 post-prep census totals: $GATE_NFAIL package(s) failed after the installers, $((GATE_NFAIL - GATE_NSTILL)) healed and re-probed whole, $GATE_NSTILL still broken."
  if [ -s "$GATE_FINAL" ]; then
    log "FAIL: these packages STILL do not import after the post-prep census heals (both classes):"
    while IFS=$'\t' read -r PKGNAME PKGVER; do
      [ -n "$PKGNAME" ] || continue
      log "  still broken: $PKGNAME==$PKGVER"
    done < "$GATE_FINAL"
    log "Refusing to generate: the venv is known-broken AFTER every installer ran - starting a generation on it would waste the run."
    exit 60
  fi
  log "r33.6 post-prep census: every post-installer failure healed and re-probed whole - the venv is measured clean at the generation gate."
fi

# ---------------------------------------------------------------- Phase R1 gate (r35)
# Before ANY generation arm runs, the R1 gate stage
# (inside/r1-stage.sh) must prove the R1 HIP fill worker on this
# stack: live limits logged, hipcc compile, load probe, the three
# Phase 0 calibration probes, and the frozen corpus equivalence
# sample against the incumbent worker. The stage is NOT
# marker-gated - it re-proves on every run. Any failure exits 65
# here and generation does not run; the incumbent code is
# untouched either way (and the previous drop's files remain as
# .prev). On success the stage writes the worker path, and
# generate.py's one command pointer (the R1_FILL_WORKER branch
# in _run_fill_subprocess) targets the R1 worker for every chunk
# fill in Phases H/I/J/J2.
# r35.7 decoupling: RESUME_045 alone now means only the
# resume RUN SHAPE (Phase I canvas skip by verification below).
# The proof skip additionally requires PROOF_SKIP=1, because job
# 046 ships a CHANGED worker binary - calibration + corpus
# equivalence run in full even though the canvas stands.
if [ "${RESUME_045:-0}" = "1" ] && [ "${PROOF_SKIP:-0}" = "1" ]; then
  export R1_RESUME_SKIP_PROOF=1
  log "[r35-r1] RESUME: R1_RESUME_SKIP_PROOF=1 exported - the gate stage will SKIP calibration + corpus equivalence with banners (standing proof: job 044); compile + load probe still run in full."
fi
if ! bash "$PKG_WSL/inside/r1-stage.sh" "$PKG_WSL" "$ROOT"; then
  log "FAIL: [r35-r1] the R1 gate stage failed (see the [r35-r1] lines above) - refusing to generate on an unproven fill worker. The incumbent remains the standing path in the .prev drop; nothing was deleted."
  exit 65
fi
export R1_FILL_WORKER="$(cat "$ROOT/r1/worker-path.txt")"
log "[r35-r1] gate green - R1_FILL_WORKER=$R1_FILL_WORKER (generate.py's fill-chunk pointer targets the R1 worker for this run)"
if [ -f "$ROOT/r1/batch-flag.txt" ]; then
  export R1_FILL_BATCH=1
  log "[r35.7-batch] batch gate green - R1_FILL_BATCH=1 exported (generate.py fills each arm's whole chunk plan through ONE worker process; the per-chunk ladder remains for any chunk the batch cannot fill)"
fi

fan_hook on
banner "Phase H: SMOKE generation - Cory's canvas front at 512, fixed seed 0"
log "Author's warm-pipeline reference on an R9700: ~56 s, ~3.4 GB peak VRAM."
log "First-ever run also pays the one-time ~400 s cold pipeline load plus the ~2 GB conditioner fetch."
env $RUN_ENV python3 "$GEN" --input "$ROOT/canvas-front.webp" --output "$OUT/canvas-smoke-512.glb" --mode 512 --seed 0 \
  || { log "FAIL: smoke generation"; exit 60; }
python3 "$CHK" "$OUT/canvas-smoke-512.glb" || { log "FAIL: smoke GLB failed the structural check"; exit 60; }
log "Phase H done: smoke GLB exists and is structurally sane."
fan_hook off

# ---------------------------------------------------------------- Phase I: the real test (canvas @1024_cascade)
if [ "${RESUME_045:-0}" = "1" ]; then
  banner "Phase I: SKIPPED (resume) - canvas generation declared complete by reference to job 044"
  log "RESUME: verifying the standing canvas output before skipping: $OUT/canvas-1024cascade.glb (expected: 9139964 bytes, md5 5542882d18e7b7e582c30ee505c5d18e - the copy Cory pulled from this package's output on Oct 27 and Joe hash-verified; job 044 Phase I logged bounds gate PASS + a structurally sane GLB at 2026-10-26 16:38:02)."
  _GLB="$OUT/canvas-1024cascade.glb"
  _SIZE=$(stat -L -c %s "$_GLB" 2>/dev/null || echo missing)
  _MD5=$(md5sum "$_GLB" 2>/dev/null | awk '{print $1}')
  log "RESUME: standing canvas GLB on disk: size=$_SIZE md5=$_MD5"
  if [ "$_SIZE" = "9139964" ] && [ "$_MD5" = "5542882d18e7b7e582c30ee505c5d18e" ]; then
    log "RESUME: standing canvas GLB VERIFIED - Phase I output stands as job 044 left it; NOT regenerated, NOT overwritten."
  else
    log "FAIL: RESUME premise broken - the standing canvas GLB is missing or does not match job 044's verified size/md5. Refusing to proceed on a false premise; nothing was overwritten. (Exit 71 = resume premise failure.)"
    exit 71
  fi
else
  banner "Phase I: TEST generation - Cory's canvas front at 1024_cascade, fixed seed 0"
  log "Author's warm reference: ~81 s, ~3.4 GB peak VRAM on the R9700."
  if env $RUN_ENV python3 "$GEN" --input "$ROOT/canvas-front.webp" --output "$OUT/canvas-1024cascade.glb" --mode 1024_cascade --seed 0; then
    python3 "$CHK" "$OUT/canvas-1024cascade.glb" || { log "FAIL: 1024_cascade GLB failed the structural check"; exit 70; }
    log "Phase I done at full 1024_cascade."
  else
    log "1024_cascade failed - retrying ONCE at 512 and marking the result as a DOWNGRADE."
    cp "$OUT/canvas-smoke-512.glb" "$OUT/canvas-1024cascade.glb" 2>/dev/null || true
    echo "1024_cascade failed on this run; canvas-1024cascade.glb is the 512 smoke result, NOT a 1024 result." > "$OUT/DOWNGRADED.txt"
    [ -f "$OUT/canvas-smoke-512.glb" ] || { log "FAIL: no usable generation at any setting"; exit 70; }
  fi
fi

# ---------------------------------------------------------------- Phase J: subject generation (optional)
# If input/subject-front.<png|jpg|jpeg|webp> exists in the package, generate
# THAT subject at 1024_cascade after the canvas test pair. Subject failures
# are warnings only - the canvas verdict stands on its own.
SUBJ=""
# r18: Cory's label convention - the FRONT view input is subject-FRONT.png
# (a cutout with our own alpha, crop upscaled to exactly 1024). The old
# lowercase name still works as a fallback.
if [ -f "$PKG_WSL/input/subject-FRONT.png" ]; then SUBJ="$PKG_WSL/input/subject-FRONT.png"; fi
if [ -z "$SUBJ" ]; then
  for ext in png jpg jpeg webp; do
    if [ -f "$PKG_WSL/input/subject-front.$ext" ]; then SUBJ="$PKG_WSL/input/subject-front.$ext"; break; fi
  done
fi
if [ -n "$SUBJ" ]; then
  fan_hook on
  if [ "${RESUME_045:-0}" = "1" ]; then
    log "RESUME ENTRY: Phase J is this run's re-entry point. Job 044's arm-2 chunk artifacts do NOT persist on disk (generate.py holds sampling/extraction/chunks in memory; its only .npz writes are the fill seam's transient temp files, deleted per attempt) - this arm RE-DERIVES sampling + extraction + chunking at fixed seed 0. Census bar: it must reproduce job 044's arm-2 numbers exactly - 1602346 voxels, 34 extraction chunks, merged 1602346 verts genuine / 3034500 faces genuine. The fill then runs BATCHED on the R1 worker through one worker process (r35.7), still narrated per chunk by the [r35.6-fill] heartbeat lines, printed live by the batch hooks."
  fi
  banner "Phase J: SUBJECT generation - $(basename "$SUBJ") at 1024_cascade, fixed seed 0"
  if env $RUN_ENV python3 "$GEN" --input "$SUBJ" --output "$OUT/subject-1024cascade.glb" --mode 1024_cascade --seed 0 --texture-size 4096 --decimation-target 1000000; then
    python3 "$CHK" "$OUT/subject-1024cascade.glb" || log "WARN: subject GLB failed the structural check (kept for inspection)"
    log "Phase J done: subject generated."
  else
    log "WARN: subject generation failed - canvas results unaffected."
  fi
  fan_hook off
else
  log "Phase J skipped: no input/subject-front.* in the package."
fi

# ---------------------------------------------------------------- Phase J2: r30 control arm (marker-gated)
# One subject run at NON-cascade 1024, same seed, profiling forced on.
# Separates "cascade assembly caps faces at 2^20" from "single-pass
# extraction caps too" (HUNT-NOTE-3 verdict table). INERT unless the
# marker input/PROBE-CONTROL.local exists - normal runs never pay
# for this arm.
if [ "${SKIP_CONTROL:-0}" = "1" ]; then
  log "Phase J2: SKIPPED (r35.8 small test - SKIP_CONTROL=1): the control arm answered its question in job 046; this run tests the cascade arm's r34 export only."
fi
if [ "${SKIP_CONTROL:-0}" != "1" ] && [ -f "$PKG_WSL/input/PROBE-CONTROL.local" ] && [ -n "$SUBJ" ]; then
  banner "Phase J2: CONTROL generation - subject at 1024 (non-cascade), fixed seed 0, profiling on"
  if env $RUN_ENV TRELLIS_PROFILE=1 python3 "$GEN" --input "$SUBJ" --output "$OUT/subject-1024control.glb" --mode 1024 --seed 0 --texture-size 4096 --decimation-target 1000000; then
    python3 "$CHK" "$OUT/subject-1024control.glb" || log "WARN: control GLB failed the structural check (kept for inspection)"
    log "Phase J2 done: control generated."
  else
    log "WARN: control generation failed - main results unaffected."
  fi
fi

# ---------------------------------------------------------------- Phase K2: CuMesh probe + build-our-own patched CuMesh
# r22 (2026-10-24): r21's probe caught the murder on camera - S1 (post-init)
# counts look right but ~51% of face rows are zero-filled (the 2^20 cliff),
# S3 simplify collapses to 989 faces, S4 cleanup keeps a 635-vert husk
# (x <= 0 half). r21's patch arm found NO CuMesh source tree on disk (the
# Phase E build tree is gone), so r22 brings its own supply: clone the
# visualbruno fork (the one the issue-#42 A/B test proved with PR #35),
# apply PR #35 with fail-loud marker gates, build it into a STAGING dir,
# run the stage probe against the staged build via PYTHONPATH, and only
# if the probe walks the mesh through whole (S1 degenerate_faces ~ 0)
# swap the staged package over the installed one (original backed up to
# cumesh-pre-pr35-backup beside it). Any failure leaves the working
# install untouched.
if [ "${SKIP_K2:-0}" = "1" ]; then
  log "Phase K2: SKIPPED (r35.8 small test - SKIP_K2=1): the CuMesh probe/build is unrelated to the bounds question."
else
banner "Phase K2: CuMesh stage probe (baseline, unpatched)"
env $RUN_ENV python3 "$GEN" --input "$ROOT/canvas-front.webp" --output "$OUT/probe-unpatched.glb" --mode 1024_cascade --seed 0 --probe-cumesh \
  || log "WARN: baseline probe run failed (its [probe] lines above say where)"
CM_SRC="$ROOT/cumesh-src"
CM_STAGE="$ROOT/cumesh-staging"
rm -rf "$CM_SRC" "$CM_STAGE"
log "Phase K2: cloning the CuMesh fork (visualbruno; fallback JeffreyXiang)..."
if git clone --depth 1 https://github.com/visualbruno/CuMesh.git "$CM_SRC" 2>/dev/null || git clone --depth 1 https://github.com/JeffreyXiang/CuMesh.git "$CM_SRC"; then
  if (cd "$CM_SRC" && patch -p1 < "$PKG_WSL/inside/cumesh-pr35.diff") \
     && grep -q "kChunk" "$CM_SRC/src/io.cu" && grep -q "hipMemcpy2D fails" "$CM_SRC/src/io.cu"; then
    log "Phase K2: PR #35 applied, markers verified."
    # r24 (2026-10-25): the PR #35 tree does not compile under HIP as-is:
    # its clean_up source uses ::cuda::std::tuple (the libcu++ spelling),
    # which the HIP/rocThrust headers do not provide - r22's build died
    # at clean_up.hip:234 with "no member named 'cuda'". Substitute the
    # thrust spelling in the clean_up source(s) before building. If the
    # pattern is absent, WARN loudly and build anyway (upstream may have
    # fixed it) - never fail silently in either direction. The install
    # itself is untouched by this; only the freshly cloned source tree
    # in $CM_SRC is edited.
    _TUPLE_FILES=$(grep -rl '::cuda::std::tuple' "$CM_SRC/src" 2>/dev/null | grep -i 'clean_up' || true)
    if [ -n "$_TUPLE_FILES" ]; then
      for _tf in $_TUPLE_FILES; do
        sed -i 's/::cuda::std::tuple/thrust::tuple/g' "$_tf"
        log "Phase K2: r24 tuple fix applied to $_tf (::cuda::std::tuple -> thrust::tuple)"
        # r27 (2026-10-25): the r24 substitution alone still failed to
        # build - clean_up.hip:234:33: "error: no template named
        # 'tuple' in namespace 'thrust'". The NAME was fixed but the
        # header that declares thrust::tuple was never included. Add
        # #include <thrust/tuple.h> right after the file's first
        # #include line, wherever the substitution landed and the
        # include is not already present; then verify it is there and
        # WARN if it could not be placed - never fail silently in
        # either direction; the build below is the final judge.
        if grep -q 'thrust::tuple' "$_tf" && ! grep -q '#include <thrust/tuple.h>' "$_tf"; then
          sed -i '0,/^#include.*/s//&\n#include <thrust\/tuple.h>/' "$_tf"
        fi
        if grep -q 'thrust::tuple' "$_tf" && grep -q '#include <thrust/tuple.h>' "$_tf"; then
          log "Phase K2: r27 include fix verified in $_tf (#include <thrust/tuple.h> present for thrust::tuple)"
        else
          log "Phase K2: WARNING - r27 include fix could NOT be verified in $_tf (no #include line to anchor to, or the substitution left no thrust::tuple use); if the build dies at clean_up.hip on the tuple error, this is the spot."
        fi
      done
    else
      log "Phase K2: WARNING - r24 tuple fix found no '::cuda::std::tuple' in the cloned clean_up sources; building as-is. If the build dies at clean_up with 'no member named cuda', this is the spot."
    fi
    log "Phase K2: building patched CuMesh into staging (the long, hot compile; heartbeat on)..."
    if with_heartbeat "patched cumesh build" pip install --no-build-isolation --no-deps --target "$CM_STAGE" "$CM_SRC"; then
      log "Phase K2: staged build OK. Probing against the STAGED cumesh (PYTHONPATH)..."
      if env $RUN_ENV PYTHONPATH="$CM_STAGE" python3 "$GEN" --input "$ROOT/canvas-front.webp" --output "$OUT/probe-patched.glb" --mode 1024_cascade --seed 0 --probe-cumesh; then
        SITE_CM=$(python3 -c "import cumesh, os; print(os.path.dirname(cumesh.__file__))" 2>/dev/null)
        STAGED_CM=$(ls -d "$CM_STAGE"/cumesh 2>/dev/null)
        if [ -n "$SITE_CM" ] && [ -n "$STAGED_CM" ] && [ -d "$SITE_CM" ]; then
          rm -rf "$SITE_CM.pre-pr35-backup"
          cp -r "$SITE_CM" "$SITE_CM.pre-pr35-backup"
          rm -rf "$SITE_CM"
          cp -r "$STAGED_CM" "$SITE_CM"
          log "Phase K2: PATCHED CuMesh SWAPPED IN at $SITE_CM (original backed up at $SITE_CM.pre-pr35-backup). Read the patched probe's [probe] lines above for the stage verdict."
        else
          log "Phase K2: could not locate the installed/staged cumesh dirs for the swap - staged build left at $CM_STAGE, install untouched."
        fi
      else
        log "Phase K2: patched probe run failed - NO swap. Install untouched; staged build left at $CM_STAGE."
      fi
    else
      log "Phase K2: patched CuMesh BUILD FAILED - install untouched. (Compile log tail is above.)"
    fi
  else
    log "Phase K2: PR #35 DID NOT APPLY CLEANLY to the cloned source - arm aborted, install untouched."
  fi
else
  log "Phase K2: CuMesh clone failed (network?) - arm aborted, install untouched."
fi
fi

banner "ALL PHASES COMPLETE"
log "Outputs in the package output folder: $(ls -la "$OUT")"
exit 0
