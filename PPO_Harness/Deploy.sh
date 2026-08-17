#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
STEAM_ROOT="/home/sebastian/.local/share/Steam"
GAME_APP_ID="4576510"
COMPAT_DATA_PATH="$STEAM_ROOT/steamapps/compatdata/$GAME_APP_ID"
PROTON="${STEAM_ROOT}/steamapps/common/Proton - Experimental/proton"

DLL_SOURCE="$SCRIPT_DIR/bin/Debug/netstandard2.1/PPO_Harness.dll"
DLL_TARGET="$PROJECT_ROOT/BepInEx/plugins/PPO_Harness.dll"
GAME_EXE="$PROJECT_ROOT/CasualtiesUnknown.exe"

CLIPBOARD_VALUE="H4sIAAAAAAAACoWUTW4bMQyF7zLroEiaoovcIGcIsqAkaoaw/kJRddyidw/HWXg8huSFN+MP5CPfE9/+Ta+C8Wl6mVqyC5WCbno4f/s5vXgIFf8/XBgDFUPO4jBVktOFfPrxuOVWJrYgVAIhd7G1nDCUe+VWhpJlVP4CPV5DEA1hkkDzIhvoqg5FDBRp8//vXZUqwEJprq2s2utG0xb7LDNQ6gqOKGByoBoZBLvYghC015DxDFYaYxn1MwHRDctQ8miFcqoFtxb3MLtAsgNVZJhCgBUd9o05u5Q56qB/79Mmp1bVIQMcM3etVpMiJWCcsb8TEAF7cBBhHsijRDNE1GldXeDQJ6uuuNiTDTjeICbbouF1exZKv5xwPmzjJdyunxqr4d/q/Wcf85ktnrX1mZCPqw34R5/HoGOgj0autLpoJvuYo+pp1kRGLdfHUj6rYtRJ6Tt5g8OiuVMnBANUUXfT6AjpNXDI+4A+P9/cDKW0ZISY21bp7XFRUB9Y9l7PUTd0aymnnp6GCQ7ZHgrpj9FS1an7JJxQU+7I0+00v3bN18tlUI6ICfVALR8NrtKznwljQdWpLt2ZKutRHe/HoWnzMXNwO7PfvwCoHYQSPQYAAA=="

if ! command -v dotnet >/dev/null 2>&1; then
    echo "error: dotnet was not found in PATH" >&2
    exit 1
fi

for required_path in "$PROTON" "$COMPAT_DATA_PATH" "$GAME_EXE" "$PROJECT_ROOT/BepInEx/plugins"; do
    if [[ ! -e "$required_path" ]]; then
        echo "error: required path does not exist: $required_path" >&2
        exit 1
    fi
done

echo "Building PPO_Harness..."
dotnet build "$SCRIPT_DIR/PPO_Harness.csproj" -c Debug

if [[ ! -f "$DLL_SOURCE" ]]; then
    echo "error: build did not produce $DLL_SOURCE" >&2
    exit 1
fi

echo "Copying PPO_Harness.dll into this workspace..."
cp -- "$DLL_SOURCE" "$DLL_TARGET"

if command -v wl-copy >/dev/null 2>&1; then
    printf '%s' "$CLIPBOARD_VALUE" | wl-copy
    echo "Clipboard updated with wl-copy."
elif command -v xclip >/dev/null 2>&1; then
    printf '%s' "$CLIPBOARD_VALUE" | xclip -selection clipboard
    echo "Clipboard updated with xclip."
elif command -v xsel >/dev/null 2>&1; then
    printf '%s' "$CLIPBOARD_VALUE" | xsel --clipboard --input
    echo "Clipboard updated with xsel."
else
    echo "Clipboard tool not found; skipping clipboard update."
fi

echo "Launching workspace build through Proton Experimental..."
cd -- "$PROJECT_ROOT"
PPO_TIME_SCALE="${PPO_TIME_SCALE:-10}" \
PPO_FORCE_BODY_UPDATE="${PPO_FORCE_BODY_UPDATE:-1}" \
STEAM_COMPAT_CLIENT_INSTALL_PATH="$STEAM_ROOT" \
STEAM_COMPAT_DATA_PATH="$COMPAT_DATA_PATH" \
STEAM_COMPAT_APP_ID="$GAME_APP_ID" \
WINEDLLOVERRIDES="winhttp=n,b" \
"$PROTON" run "$GAME_EXE"
