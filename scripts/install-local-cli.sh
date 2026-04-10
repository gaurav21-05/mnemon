#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${HOME}/.local/bin"
mkdir -p "$TARGET_DIR"
ln -sfn "$REPO_ROOT/bin/mnemon" "$TARGET_DIR/mnemon"
ln -sfn "$REPO_ROOT/bin/mnemon-daemon" "$TARGET_DIR/mnemon-daemon"

echo "Installed launchers:"
echo "  $TARGET_DIR/mnemon -> $REPO_ROOT/bin/mnemon"
echo "  $TARGET_DIR/mnemon-daemon -> $REPO_ROOT/bin/mnemon-daemon"
echo
echo "Try:"
echo "  mnemon --help"
echo "  mnemon-daemon --help"
