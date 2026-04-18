#!/usr/bin/env bash
# view_trace.sh — parse a torch compiler log with tlparse and serve it locally
# Usage: ./view_trace.sh [LOG_FILE] [PORT]
#
# On your LOCAL machine, run:
#   ssh -L <PORT>:localhost:<PORT> <user>@<hpc-host>
# then open http://localhost:<PORT> in your browser.

set -euo pipefail

TRACE_DIR="$(cd "$(dirname "$0")" && pwd)"
TLPARSE=$(which tlparse 2>/dev/null || echo "")
DEFAULT_PORT=8765

# ── helpers ──────────────────────────────────────────────────────────────────
die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "  $*"; }

# ── pick tlparse ──────────────────────────────────────────────────────────────
[[ -z "$TLPARSE" ]] && die "tlparse not found in PATH. Activate the right conda env first."

# ── pick log file ─────────────────────────────────────────────────────────────
LOGS=("$TRACE_DIR"/*.log)
if [[ ${#LOGS[@]} -eq 0 ]]; then
    die "No .log files found in $TRACE_DIR"
fi

if [[ $# -ge 1 && -f "$1" ]]; then
    LOG_FILE="$1"
    shift
elif [[ $# -ge 1 && -f "$TRACE_DIR/$1" ]]; then
    LOG_FILE="$TRACE_DIR/$1"
    shift
else
    echo ""
    echo "Available trace logs:"
    for i in "${!LOGS[@]}"; do
        size=$(du -sh "${LOGS[$i]}" | cut -f1)
        printf "  [%d] %-30s  %s\n" "$((i+1))" "$(basename "${LOGS[$i]}")" "$size"
    done
    echo "  [a] All logs (separate output dirs per log)"
    echo ""
    read -rp "Choose [1-${#LOGS[@]}/a]: " choice

    if [[ "$choice" == "a" || "$choice" == "A" ]]; then
        LOG_FILE="__ALL__"
    elif [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#LOGS[@]} )); then
        LOG_FILE="${LOGS[$((choice-1))]}"
    else
        die "Invalid choice."
    fi
fi

# ── pick port ─────────────────────────────────────────────────────────────────
PORT="${1:-$DEFAULT_PORT}"
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    die "Port must be a number, got: $PORT"
fi

# ── run tlparse ───────────────────────────────────────────────────────────────
parse_one() {
    local log="$1"
    local stem
    stem=$(basename "$log" .log)
    local out_dir="$TRACE_DIR/tl_out_${stem}"

    if [[ -d "$out_dir" ]]; then
        echo ""
        echo "Skipping: $(basename "$log")  (already parsed → $out_dir)"
        return
    fi

    echo ""
    echo "Parsing: $(basename "$log")"
    echo "Output:  $out_dir"

    "$TLPARSE" --no-browser -o "$out_dir" "$log"
    echo "Done: $out_dir"
}

if [[ "$LOG_FILE" == "__ALL__" ]]; then
    for log in "${LOGS[@]}"; do
        parse_one "$log"
    done
    SERVE_DIR="$TRACE_DIR"
    echo ""
    echo "Serving root trace dir (navigate into tl_out_* subdirs)"
else
    STEM=$(basename "$LOG_FILE" .log)
    OUT_DIR="$TRACE_DIR/tl_out_${STEM}"
    parse_one "$LOG_FILE"
    SERVE_DIR="$OUT_DIR"
fi

# ── serve ─────────────────────────────────────────────────────────────────────
echo ""
echo "┌─────────────────────────────────────────────────────────────────┐"
echo "│  Serving at http://localhost:${PORT}                               │"
echo "│                                                                 │"
echo "│  On your LOCAL machine run:                                     │"
printf "│    ssh -L %d:localhost:%d <user>@<hpc-host>                     │\n" "$PORT" "$PORT"
echo "│  then open:  http://localhost:${PORT}                              │"
echo "│                                                                 │"
echo "│  Press Ctrl+C to stop.                                         │"
echo "└─────────────────────────────────────────────────────────────────┘"
echo ""

cd "$SERVE_DIR"
python3 -m http.server "$PORT"
