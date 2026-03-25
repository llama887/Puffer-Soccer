#!/bin/bash

set -euo pipefail

# Watch a Slurm job until it starts running and, optionally, emits stdout.
#
# This helper exists because launch failures can happen very early, before anyone
# notices that the job left the queue. It prints the current scheduler state,
# shows newly written stderr and stdout lines from the repo-local log files, and
# exits successfully after the job reaches the running state and, when requested,
# after the stdout log has at least one line. If the job fails, is cancelled,
# times out, or disappears from the queue before satisfying the requested
# condition, the script exits non-zero so the caller can treat that as a launch
# failure.

if [ "$#" -ne 1 ]; then
    printf 'Usage: %s <job_id>\n' "$0" >&2
    exit 2
fi

JOB_ID="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
ERR_FILE=""
OUT_FILE=""
ERR_LINES=0
OUT_LINES=0
POLL_SECONDS="${WATCH_SBATCH_POLL_SECONDS:-5}"
WAIT_FOR_STDOUT_LINES="${WATCH_SBATCH_WAIT_FOR_STDOUT_LINES:-0}"

print_new_lines() {
    local file_path="$1"
    local previous_lines="$2"
    local label="$3"
    local current_lines

    if [ ! -f "$file_path" ]; then
        printf '%s' "$previous_lines"
        return
    fi

    current_lines="$(wc -l < "$file_path")"
    if [ "$current_lines" -gt "$previous_lines" ]; then
        printf '[%s] %s\n' "$label" "$file_path" >&2
        sed -n "$((previous_lines + 1)),$((current_lines))p" "$file_path" >&2
    fi

    printf '%s' "$current_lines"
}

resolve_log_paths() {
    local discovered_err
    local discovered_out

    discovered_err="$(ls -1t "$LOG_DIR"/puffer_soccer_auto-"$JOB_ID".err 2>/dev/null | head -n 1 || true)"
    discovered_out="$(ls -1t "$LOG_DIR"/puffer_soccer_auto-"$JOB_ID".out 2>/dev/null | head -n 1 || true)"

    if [ -n "$discovered_err" ]; then
        ERR_FILE="$discovered_err"
    fi

    if [ -n "$discovered_out" ]; then
        OUT_FILE="$discovered_out"
    fi
}

printf 'Watching job %s\n' "$JOB_ID"

while true; do
    resolve_log_paths

    if squeue_output="$(squeue -h -j "$JOB_ID" -o '%T|%R' 2>/dev/null)"; then
        if [ -n "$squeue_output" ]; then
            job_state="${squeue_output%%|*}"
            job_reason="${squeue_output#*|}"
            printf 'state=%s reason=%s\n' "$job_state" "$job_reason"

            if [ -n "$ERR_FILE" ]; then
                ERR_LINES="$(print_new_lines "$ERR_FILE" "$ERR_LINES" "stderr")"
            fi

            if [ -n "$OUT_FILE" ]; then
                OUT_LINES="$(print_new_lines "$OUT_FILE" "$OUT_LINES" "stdout")"
            fi

            if [ "$job_state" = "RUNNING" ]; then
                if [ "$WAIT_FOR_STDOUT_LINES" = "1" ]; then
                    if [ -n "$OUT_FILE" ] && [ "$OUT_LINES" -gt 0 ]; then
                        printf 'Job %s is running and stdout is active\n' "$JOB_ID"
                        exit 0
                    fi
                else
                    printf 'Job %s is running\n' "$JOB_ID"
                    exit 0
                fi
            fi

            sleep "$POLL_SECONDS"
            continue
        fi
    fi

    sacct_output="$(sacct -n -j "$JOB_ID" --format=State,ExitCode -P | head -n 1 || true)"
    if [ -n "$sacct_output" ]; then
        job_state="${sacct_output%%|*}"
        exit_code="${sacct_output#*|}"

        if [ -n "$ERR_FILE" ]; then
            ERR_LINES="$(print_new_lines "$ERR_FILE" "$ERR_LINES" "stderr")"
        fi

        if [ -n "$OUT_FILE" ]; then
            OUT_LINES="$(print_new_lines "$OUT_FILE" "$OUT_LINES" "stdout")"
        fi

        printf 'final_state=%s exit_code=%s\n' "$job_state" "$exit_code" >&2
        exit 1
    fi

    printf 'Job %s is no longer visible in squeue and not yet present in sacct\n' "$JOB_ID" >&2
    exit 1
done
