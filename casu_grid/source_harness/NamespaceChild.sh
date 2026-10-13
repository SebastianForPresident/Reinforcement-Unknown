#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${1:?runtime root is required}"
LAUNCHER="${2:?contained launcher is required}"
shift 2
MASKED_HOME="$RUNTIME_ROOT/masked-home"
MASKED_DRI="$RUNTIME_ROOT/masked-dri"
SECCOMP_EXEC="$RUNTIME_ROOT/launcher/seccomp-exec"

[[ -d "$MASKED_HOME" ]] || { echo "error: masked home is missing: $MASKED_HOME" >&2; exit 1; }
[[ -d "$MASKED_DRI" ]] || { echo "error: masked DRM directory is missing: $MASKED_DRI" >&2; exit 1; }
[[ "$LAUNCHER" = "$RUNTIME_ROOT"/* ]] || { echo "error: launcher escaped runtime root" >&2; exit 2; }
[[ -x "$SECCOMP_EXEC" ]] || { echo "error: seccomp launcher is missing: $SECCOMP_EXEC" >&2; exit 1; }

# All mount changes occur after unshare and therefore cannot alter the host
# namespace. The bind mount masks the host's /home Btrfs subvolume while
# retaining the path for programs that probe it.
if [[ "${PPO_COUNTERFACTUAL_MOUNTS_READY:-0}" != "1" ]]; then
    mount --make-rprivate /
    mount --bind "$MASKED_HOME" /home
    if [[ "${PPO_COUNTERFACTUAL_MASK_DRI:-1}" == "1" && -d /dev/dri ]]; then
        mount --bind "$MASKED_DRI" /dev/dri
    fi
fi

export HOME="$RUNTIME_ROOT/user-home"
export XDG_CONFIG_HOME="$RUNTIME_ROOT/xdg-config"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg-cache"
export XDG_DATA_HOME="$RUNTIME_ROOT/xdg-data"
export XDG_RUNTIME_DIR="$RUNTIME_ROOT/xdg-runtime"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
unset OLDPWD PYTHONPATH VIRTUAL_ENV CONDA_PREFIX PIP_PREFIX
cd -- "$RUNTIME_ROOT/game"
export PWD

# No inherited environment value may retain the development mount. Values are
# scrubbed before the runtime launch and only variable names are recorded, so
# the audit cannot leak unrelated environment contents.
SCRUB_LOG="$RUNTIME_ROOT/user-home/environment_scrub.log"
: > "$SCRUB_LOG"
for name in $(compgen -e); do
    value="${!name-}"
    if [[ "$value" == *"/home"* || "$value" == *"Reinforcement-Unknown"* ]]; then
        printf '%s\n' "$name" >> "$SCRUB_LOG"
        unset "$name"
    fi
done

# Forward only the explicit test-only source instrumentation allowlist.
for name in PPO_DIAG_PLAYER_TICKS PPO_DIAG_PLAYER_TICKS_OUTPUT PPO_DIAG_PLAYER_POST_PHYSICS PPO_DIAG_PLAYER_POST_PHYSICS_OUTPUT PPO_DIAG_PLAYER_PREFIXED PPO_DIAG_PLAYER_PREFIXED_OUTPUT PPO_DIAG_OCC_FULL PPO_DIAG_OCC_FULL_AT PPO_DIAG_OCC_FULL_OUTPUT PPO_DIAG_OCC_REGION PPO_DIAG_OCC_EVERY PPO_DIAG_OCC_OUTPUT PPO_DIAG_PLAYER_FIXTURE_GRID_X PPO_DIAG_PLAYER_FIXTURE_GRID_Y PPO_DIAG_PLAYER_FIXTURE_TERRAIN; do
    if [[ -n "${!name+x}" ]]; then export "$name"; fi
done

# The host Xwayland connection is an external Unix stream socket and cannot be
# part of a self-contained CRIU process tree. For the privileged counterfactual
# run, start a private X server here, after entering the experiment PID/cgroup
# tree, so Wine clients and their display server are checkpointed together.
if [[ "${PPO_COUNTERFACTUAL_XVFB:-0}" == "1" ]]; then
    XVFB_BIN=/usr/bin/Xvfb
    XDPYINFO_BIN=/usr/bin/xdpyinfo
    XAUTH_BIN=/usr/bin/xauth
    DISPLAY=:99
    XAUTHORITY="$RUNTIME_ROOT/user-home/.Xauthority-cb1"
    XVFB_LOG="$RUNTIME_ROOT/user-home/diagnostics/xvfb.log"
    X11_SOCKET=/tmp/.X11-unix/X99
    X11_LOCK=/tmp/.X99-lock

    [[ -x "$XVFB_BIN" && -x "$XDPYINFO_BIN" && -x "$XAUTH_BIN" ]] || {
        echo "error: private Xvfb dependencies are missing" >&2
        exit 1
    }
    [[ ! -e "$X11_SOCKET" && ! -L "$X11_SOCKET" && ! -e "$X11_LOCK" && ! -L "$X11_LOCK" ]] || {
        echo "error: private X display :99 is already occupied" >&2
        exit 1
    }

    mkdir -p -- "$(dirname -- "$XVFB_LOG")"
    PREVIOUS_UMASK="$(umask)"
    umask 077
    : > "$XAUTHORITY"
    chmod 0600 "$XAUTHORITY"
    COOKIE="$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')"
    [[ "$COOKIE" =~ ^[0-9a-f]{32}$ ]] || {
        echo "error: failed to generate Xauthority cookie" >&2
        exit 1
    }
    "$XAUTH_BIN" -f "$XAUTHORITY" add "$DISPLAY" . "$COOKIE"
    unset COOKIE
    umask "$PREVIOUS_UMASK"

    "$XVFB_BIN" "$DISPLAY" -auth "$XAUTHORITY" \
        -screen 0 1920x1034x24 -dpi 96 -nolisten tcp -noreset \
        >"$XVFB_LOG" 2>&1 &
    XVFB_PID=$!
    XVFB_READY=0
    for _ in $(seq 1 100); do
        if ! kill -0 "$XVFB_PID" 2>/dev/null; then
            wait "$XVFB_PID" || true
            echo "error: private Xvfb exited during startup" >&2
            sed -n '1,120p' "$XVFB_LOG" >&2
            exit 1
        fi
        if DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" \
            "$XDPYINFO_BIN" -display "$DISPLAY" >/dev/null 2>&1; then
            XVFB_READY=1
            break
        fi
        sleep 0.1
    done
    if [[ "$XVFB_READY" != "1" ]]; then
        kill "$XVFB_PID" 2>/dev/null || true
        wait "$XVFB_PID" 2>/dev/null || true
        echo "error: private Xvfb did not become ready" >&2
        sed -n '1,120p' "$XVFB_LOG" >&2
        exit 1
    fi

    export DISPLAY XAUTHORITY
    unset WAYLAND_DISPLAY
    echo "Private checkpoint-contained X display ready: $DISPLAY (1920x1034x24 work area, 96 dpi)"
fi

exec "$SECCOMP_EXEC" "$LAUNCHER" "$@"
