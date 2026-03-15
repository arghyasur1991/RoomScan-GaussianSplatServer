#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "[build] Building React app..."
cd "$ROOT_DIR/web"
npm ci --legacy-peer-deps
npm run build

echo "[build] Copying dist to server/static..."
rm -rf "$ROOT_DIR/server/static"
cp -r "$ROOT_DIR/web/dist" "$ROOT_DIR/server/static"

echo "[build] Done! Run the server with:"
echo "  cd $ROOT_DIR/server && uvicorn main:app --host 0.0.0.0 --port 8420"
