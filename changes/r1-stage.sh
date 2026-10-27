#!/usr/bin/env bash
# r1-stage.sh - the r35 R1 GATE STAGE, run by build-inside.sh after
# the post-prep census gate and BEFORE any generation arm (Phases
# H/I/J/J2). Version: 2026-10-27-r35.7-r1-fill-batch.
#
# The stage answers one question with evidence: may this run's
# chunk fills run on the R1 HIP fill worker instead of the
# incumbent CuMesh worker? Gates, in order (ANY failure exits 65
# and generation does NOT run; the incumbent code is never touched
# by this stage - it stays in place as the fallback, and the
# previous drop remains in the package as generate.py.prev /
# build-inside.sh.prev via the update mechanism):
#
#   0. LIVE LIMITS CAPTURE (logging only, never gated): the WSL
#      memory/swap/CPU limits this run actually booted under,
#      read from /proc/meminfo + nproc + free -g. (The .wslconfig
#      that sets them must be written before WSL boots - it is
#      Cory's optional pre-step via the standalone tuning kit,
#      never something this drop can or does change mid-flight.)
#   a. COMPILE: the CPU logic tests must pass (the shared brain),
#      then hipcc builds the HIP worker for gfx1201 (ROCm 7.2.x).
#      The shipped source must contain exactly ONE hipMemcpy call
#      site (the <=65,536-row chunked transfer of the R1 design -
#      the 2^20 cliff impossible by construction).
#   b. LOAD PROBE (BUILD-ON-PC section 4): the binary refuses a
#      missing input loudly and writes nothing; it fills the
#      small_hole calibration fixture exactly; the equivalence
#      harness's check battery passes all 5 fixtures.
#   c. CALIBRATION (r1_calibration.py): incumbent-vs-itself
#      determinism on 25 real corpus chunks; the crafted
#      B>0/M=0 code-9 shape probe; the threshold-edge probe.
#   d. CORPUS EQUIVALENCE SAMPLE (r1_equiv_check.py): R1 vs
#      incumbent, per-chunk, over the frozen predeclared sample
#      (inside/r1-corpus/sample-subject.json).
#   e. BATCH GATE (r35.7): every fixture filled through the
#      worker's --batch mode in ONE process must produce
#      byte-identical output to the single-shot seam, and a
#      poisoned manifest (good / nonexistent / good) must exit
#      10 with the earlier item kept and the later item
#      untouched. Only then does the stage write
#      batch-flag.txt, and build-inside.sh exports
#      R1_FILL_BATCH=1 for the generation arms.
#
# RESUME SWITCH (r35.6, job 045): when the caller exports
# R1_RESUME_SKIP_PROOF=1, sections (c) and (d) are SKIPPED
# with explicit banners - their proof stands by reference
# to job 044 (Oct 26, 2026), which ran them green on
# byte-identical fill sources (fill_logic.cpp sha256
# 9743e000..., fill_hip.hip.cpp sha256 9dcc851a...).
# Sections 0/a/b are NOT skipped: limits, compile and the
# load probe re-prove on every run, resume or not. With
# the variable unset the stage behaves exactly as the r35
# original. (r35.7: section (e) is NOT skipped either - the
# batch plumbing belongs to the binary this run just built,
# so it re-proves every run. Job 046 does not use the resume
# switch at all: the worker changed, so (c) and (d) run.)
#
# On success the stage writes the worker binary path to
# $ROOT/r1/worker-path.txt; build-inside.sh reads it, exports
# R1_FILL_WORKER, and only then do the generation arms start.
#
# Args: $1 = package path (WSL form), $2 = inside root (/root/trellis-amd-port)

PKG_WSL="${1:?package path (WSL form) required}"
ROOT="${2:?inside root required}"
VENV="$ROOT/venv"
PY="$VENV/bin/python"
GEN="$PKG_WSL/inside/generate.py"
SRC="$PKG_WSL/inside/r1-src"
HAR="$PKG_WSL/inside/r1-harness"
COR="$PKG_WSL/inside/r1-corpus"
PRB="$PKG_WSL/inside/r1-probes"
WORK="$ROOT/r1"
BIN="$WORK/fill_worker"
HIPCC=/opt/rocm/bin/hipcc

mkdir -p "$WORK"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }
fail() { log "[r35-r1] FAIL: $*"; log "[r35-r1] generation does NOT run. The incumbent worker code is untouched in the package (and the previous drop beside it as .prev); nothing was deleted."; exit 65; }
banner() { echo; echo "=================================================================="; log "INSIDE-WSL: $*"; echo "=================================================================="; }

banner "Phase R1: R1 fill worker gate (compile -> probe -> calibration -> equivalence)"
R1_T0=$SECONDS
if [ "${R1_RESUME_SKIP_PROOF:-0}" = "1" ]; then
  log "[r35-r1] RESUME MODE (R1_RESUME_SKIP_PROOF=1): sections (c) calibration and (d) corpus equivalence will be SKIPPED with banners - standing proof by reference to job 044 on byte-identical fill sources. Sections 0/a/b run in full."
fi

# ------------------------------------------------- 0. live limits (logging only)
log "[r35-r1] live WSL limits this run booted under (recorded for grading; NOT a gate - the values are whatever .wslconfig was in place before WSL started, and this stage cannot change them):"
_MEM_KB=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
_SWAP_KB=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
log "[r35-r1]   MemTotal: ${_MEM_KB} kB (total memory visible to WSL)"
log "[r35-r1]   SwapTotal: ${_SWAP_KB} kB (swap visible to WSL)"
log "[r35-r1]   processors visible (nproc): $(nproc)"
while IFS= read -r _free_line; do
  log "[r35-r1]   free -g: $_free_line"
done <<EOF_FREE
$(free -g)
EOF_FREE

# ------------------------------------------------- a. compile
log "[r35-r1] (a) compile: toolchain check"
[ -x "$HIPCC" ] || fail "hipcc not found at $HIPCC - the R1 worker cannot be built on this stack."
HIPCC_VER=$("$HIPCC" --version 2>&1 | head -n 5)
log "[r35-r1] hipcc --version: $(echo "$HIPCC_VER" | head -n 1)"
echo "$HIPCC_VER" | grep -q "7\.2\." || fail "hipcc does not report ROCm 7.2.x - the R1 draft was written against 7.2.x and a version change is a recorded event, not a detail to absorb silently. Full --version output: $HIPCC_VER"
command -v g++ >/dev/null 2>&1 || fail "g++ not found - needed for the CPU logic tests."
[ -f /usr/include/zlib.h ] || fail "zlib header missing (/usr/include/zlib.h) - the worker links -lz."
[ -x "$PY" ] || fail "venv python not found at $PY."

_MCP=$(grep -c "hipMemcpy(" "$SRC/fill_hip.hip.cpp" || true)
[ "$_MCP" = "1" ] || fail "fill_hip.hip.cpp must contain exactly ONE hipMemcpy( call site (the chunked transfer); found $_MCP - the shipped source is not the reviewed draft."
log "[r35-r1] transfer discipline: exactly one hipMemcpy( call site in fill_hip.hip.cpp (chunked_transfer, <= 65,536 rows per call) - as designed."

log "[r35-r1] (a) compile: CPU logic tests (the shared brain both workers run)"
g++ -std=c++17 -O2 -Wall -Wextra -I"$SRC" "$SRC/tests/test_fill_logic.cpp" "$SRC/fill_logic.cpp" -o "$WORK/test_fill_logic" \
  || fail "CPU logic test build failed (g++)."
"$WORK/test_fill_logic" > "$WORK/test_fill_logic.out" 2>&1 \
  || { cat "$WORK/test_fill_logic.out"; fail "CPU logic tests FAILED - the fill brain is not proven; the HIP wrapper is not even attempted."; }
grep -q "ALL TESTS PASS" "$WORK/test_fill_logic.out" \
  || { cat "$WORK/test_fill_logic.out"; fail "CPU logic tests did not print ALL TESTS PASS."; }
log "[r35-r1] CPU logic tests: ALL TESTS PASS (output in \$ROOT/r1/test_fill_logic.out)."

log "[r35-r1] (a) compile: HIP worker for gfx1201"
"$HIPCC" -std=c++17 -O2 --offload-arch=gfx1201 -DR1_USE_HIP -I"$SRC" \
  "$SRC/fill_worker.cpp" "$SRC/fill_logic.cpp" "$SRC/fill_hip.hip.cpp" \
  -lz -o "$BIN" \
  || fail "hipcc build of the R1 worker FAILED - the draft does not compile on this stack as-is. The exact compiler output is above; file it as the next R1 attempt. The incumbent is untouched."
[ -x "$BIN" ] || fail "hipcc reported success but $BIN is not an executable file."
log "[r35-r1] HIP worker built: $BIN"

# ------------------------------------------------- b. load probe
log "[r35-r1] (b) load probe (BUILD-ON-PC section 4)"
rm -f "$WORK/never.npz"
"$BIN" /nonexistent.npz "$WORK/never.npz" 0.03 >/dev/null 2>&1
_RC=$?
[ "$_RC" != "0" ] || fail "load probe: the worker exited 0 on a nonexistent input - its failure discipline is broken."
[ ! -e "$WORK/never.npz" ] || fail "load probe: the worker wrote an output file for a nonexistent input."
log "[r35-r1] load probe (a): missing input refused (exit $_RC), no output file written - exit discipline holds."

"$BIN" "$HAR/fixtures/chunks/small_hole.npz" "$WORK/probe_small.npz" 5.0 \
  || fail "load probe: the worker failed on the small_hole calibration fixture."
"$PY" - "$HAR/fixtures/chunks/small_hole.npz" "$WORK/probe_small.npz" <<'EOF_PROBE'
import sys
import numpy as np
i = np.load(sys.argv[1])
o = np.load(sys.argv[2])
assert np.array_equal(o["verts"][:8], i["verts"]), "prefix verts not bit-exact"
assert np.array_equal(o["faces"][:8], i["faces"]), "prefix faces not bit-exact"
assert o["verts"].shape == (9, 3) and o["faces"].shape == (12, 3), \
    f"expected 1 appended centroid + 4 appended fans, got {o['verts'].shape} / {o['faces'].shape}"
assert o["verts"][8].tolist() == [0.0, 0.0, 0.0], "appended centroid is not the hole centroid"
print("probe OK: 1 centroid + 4 fans, prefix bit-exact")
EOF_PROBE
[ $? -eq 0 ] || fail "load probe: small_hole output failed the exact assertions (see above)."
log "[r35-r1] load probe (b): small_hole filled exactly (1 centroid + 4 fans, prefix bit-exact)."

"$PY" "$HAR/run_harness.py" check --worker "$BIN" --corpus "$HAR/fixtures/chunks" \
  --max-hole-perimeter 5.0 --generate-py "$GEN" --timeout 900 \
  --out "$WORK/harness-fixtures.json" \
  || fail "load probe: the harness check battery did not pass all 5 calibration fixtures for the HIP worker (verdict JSON: \$ROOT/r1/harness-fixtures.json)."
log "[r35-r1] load probe (c): harness check battery - 5/5 fixtures PASS for the HIP worker."

# ------------------------------------------------- c. calibration
if [ "${R1_RESUME_SKIP_PROOF:-0}" = "1" ]; then
  log "[r35-r1] (c) calibration: SKIPPED (resume) - standing proof: job 044 (2026-10-26) calibration PASSED on byte-identical fill sources: probe 1 ZERO R1 failures on all 25 chunks (1/4/11/12/24 included), probe 2 clean no-fill exit 0 bit-identical, probe 3 PASS. NOT re-run in this resume run."
else
  log "[r35-r1] (c) calibration probes (determinism / code-9 shape / threshold edge)"
  "$PY" "$HAR/r1_calibration.py" \
    --generate-py "$GEN" --r1-worker "$BIN" \
    --corpus-dir "$COR/subject" --sample-file "$COR/sample-subject.json" \
    --probes-dir "$PRB" --work-dir "$WORK" --threshold 0.03 --timeout 900 \
    || fail "calibration probes FAILED (see the [r35-r1] calibration lines above) - generation does NOT run with R1."
fi

# ------------------------------------------------- d. equivalence sample
if [ "${R1_RESUME_SKIP_PROOF:-0}" = "1" ]; then
  log "[r35-r1] (d) corpus equivalence sample: SKIPPED (resume) - standing proof: job 044 (2026-10-26) corpus equivalence sample PASSED on byte-identical fill sources (frozen predeclared list, R1 vs incumbent per-chunk identical). NOT re-run in this resume run."
else
  log "[r35-r1] (d) corpus equivalence sample (frozen predeclared list)"
  "$PY" "$HAR/r1_equiv_check.py" \
    --generate-py "$GEN" --r1-worker "$BIN" \
    --corpus-dir "$COR/subject" --sample-file "$COR/sample-subject.json" \
    --work-dir "$WORK" --threshold 0.03 --timeout 900 \
    || fail "corpus equivalence sample FAILED (see the [r35-r1] equivalence lines above) - generation does NOT run with R1."
fi

# ------------------------------------------------- e. batch gate (r35.7)
log "[r35-r1] (e) batch gate: one process, every fixture, byte-identical to the single-shot seam"
BG="$WORK/batchgate"
# Scrub any earlier run's gate evidence in this $ROOT first
# (santa review, Oct 27): a stale batch-flag.txt or batch
# output must never let this run's comparisons pass on old
# files.
rm -rf "$BG"
rm -f "$WORK/batch-flag.txt"
mkdir -p "$BG"
: > "$BG/manifest.txt"
for _fx in "$HAR"/fixtures/chunks/*.npz; do
  _name=$(basename "$_fx")
  "$BIN" "$_fx" "$BG/single-$_name" 5.0 \
    || fail "batch gate: single-shot fill failed on fixture $_name - cannot compare batch output."
  printf '%s\t%s\t%s\n' "$_fx" "$BG/batch-$_name" "5.0" >> "$BG/manifest.txt"
done
"$BIN" --batch "$BG/manifest.txt" > "$BG/batch.stdout" 2> "$BG/batch.stderr" \
  || { cat "$BG/batch.stderr"; fail "batch gate: --batch exited nonzero on an all-good fixture manifest."; }
for _fx in "$HAR"/fixtures/chunks/*.npz; do
  _name=$(basename "$_fx")
  cmp -s "$BG/single-$_name" "$BG/batch-$_name" \
    || fail "batch gate: fixture $_name batch output differs from single-shot - batch mode is NOT proven identical."
done
log "[r35-r1] batch gate (a): every fixture output byte-identical, batch vs single-shot."
_first=$(basename "$(ls "$HAR"/fixtures/chunks/*.npz | head -n 1)")
_second=$(basename "$(ls "$HAR"/fixtures/chunks/*.npz | head -n 2 | tail -n 1)")
printf '%s\t%s\t%s\n' "$HAR/fixtures/chunks/$_first" "$BG/poison-0.npz" "5.0" > "$BG/manifest-poison.txt"
printf '%s\t%s\t%s\n' "/nonexistent.npz" "$BG/poison-1.npz" "5.0" >> "$BG/manifest-poison.txt"
printf '%s\t%s\t%s\n' "$HAR/fixtures/chunks/$_second" "$BG/poison-2.npz" "5.0" >> "$BG/manifest-poison.txt"
"$BIN" --batch "$BG/manifest-poison.txt" > "$BG/poison.stdout" 2> "$BG/poison.stderr"
_RC=$?
[ "$_RC" = "10" ] || { cat "$BG/poison.stderr"; fail "batch gate: poisoned manifest exited $_RC, expected 10 (exit at the first failed item)."; }
[ -e "$BG/poison-0.npz" ] || fail "batch gate: poisoned manifest lost the item BEFORE the failure."
[ ! -e "$BG/poison-1.npz" ] || fail "batch gate: poisoned manifest wrote an output for the failed item."
[ ! -e "$BG/poison-2.npz" ] || fail "batch gate: poisoned manifest continued PAST the failed item - exit-at-first-failure discipline is broken."
log "[r35-r1] batch gate (b): poisoned manifest exits 10 at the failed item; earlier output kept, later items untouched."
echo "1" > "$WORK/batch-flag.txt"

# ------------------------------------------------- green
echo "$BIN" > "$WORK/worker-path.txt"
log "[r35-r1] batch gate green - batch-flag.txt written; build-inside.sh will export R1_FILL_BATCH=1 so generation fills through ONE worker process (r35.7)."
if [ "${R1_RESUME_SKIP_PROOF:-0}" = "1" ]; then
  log "[r35-r1] GATE GREEN in $((SECONDS - R1_T0))s (RESUME): compile + load probe pass THIS run; calibration + corpus equivalence SKIPPED, standing by reference to job 044. Generation arms proceed with R1 as the fill worker ($BIN); the incumbent code remains in the package, untouched, as the fallback."
else
  log "[r35-r1] GATE GREEN in $((SECONDS - R1_T0))s: compile + load probe + calibration + equivalence all pass. Generation arms proceed with R1 as the fill worker ($BIN); the incumbent code remains in the package, untouched, as the fallback."
fi
exit 0
