#!/bin/bash
# Shortscut VPS - Deploy server-side face tracking
#
# Prerequisites: vps/server.py, vps/face_tracking.py and
# vps/requirements.txt on this machine are already up to date (git pull,
# or scp'd manually) before running this script.
#
# Usage (on the VPS): bash deploy_facetrack.sh
set -e

SERVER_PY=$(find /root /opt /srv /home -name "server.py" -path "*shortscut*" 2>/dev/null | head -1)
if [ -z "$SERVER_PY" ]; then
    SERVER_PY="/root/shortscut/server.py"
fi
SERVER_DIR="$(dirname "$SERVER_PY")"
echo "Server dir: $SERVER_DIR"

if [ ! -f "$SERVER_DIR/face_tracking.py" ]; then
    echo "ERROR: face_tracking.py not found in $SERVER_DIR — sync the repo first (git pull, or scp vps/server.py vps/face_tracking.py vps/requirements.txt)."
    exit 1
fi

echo "Installing face-tracking Python deps (opencv-python-headless, mediapipe)..."
pip3 install -r "$SERVER_DIR/requirements.txt"

echo "Restarting server..."
pkill -f "python3.*server.py.*3458" 2>/dev/null || true
sleep 2

cd "$SERVER_DIR"
nohup python3 "$SERVER_PY" > /var/log/shortscut-server.log 2>&1 &
sleep 2

echo "=== Health check ==="
curl -s http://localhost:3458/health
echo ""
echo "Done. Watch /var/log/shortscut-server.log during the next job for 'face crop mode = tracking/dual/single/center' lines to confirm it's working."
