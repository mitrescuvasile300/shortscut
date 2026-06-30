#!/usr/bin/env python3
"""
Shortscut VPS Deploy Script — Cross-platform (Windows / macOS / Linux)
=====================================================================
Connects to VPS via SSH, deploys the updated server.py with /transcript
endpoint, restarts the server, and tests it.

Usage:
    python deploy_transcript.py
    python deploy_transcript.py --host 76.13.133.153 --user root
    python deploy_transcript.py --key ~/.ssh/id_ed25519
    python deploy_transcript.py --password

Requirements:
    pip install paramiko
"""

import argparse
import sys
import time
import json

try:
    import paramiko
except ImportError:
    print("Installing paramiko...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "paramiko"])
    import paramiko

try:
    import urllib.request
except ImportError:
    pass

# ── Config ──────────────────────────────────────────────────────────
VPS_HOST = "76.13.133.153"
VPS_USER = "root"
VPS_PORT = 22
SERVER_PORT = 3458
API_KEY = "shortcut-vps-2026"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/mitrescuvasile300/shortscut/main/vps/server.py"


def connect_ssh(host, user, port, key_path=None, password=None):
    """Connect to VPS via SSH."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs = {
        "hostname": host,
        "port": port,
        "username": user,
        "timeout": 15,
    }

    if key_path:
        print(f"  Connecting with key: {key_path}")
        connect_kwargs["key_filename"] = key_path
    elif password:
        print(f"  Connecting with password...")
        connect_kwargs["password"] = password
    else:
        # Try default keys
        print(f"  Connecting with default SSH keys...")
        connect_kwargs["look_for_keys"] = True

    client.connect(**connect_kwargs)
    return client


def run_remote(client, cmd, timeout=60):
    """Run a command on VPS and return (stdout, stderr, exit_code)."""
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return out, err, exit_code


def print_step(n, msg):
    print(f"\n{'='*60}")
    print(f"  Step {n}: {msg}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Deploy Shortscut VPS /transcript endpoint")
    parser.add_argument("--host", default=VPS_HOST, help=f"VPS hostname (default: {VPS_HOST})")
    parser.add_argument("--user", default=VPS_USER, help=f"SSH user (default: {VPS_USER})")
    parser.add_argument("--port", type=int, default=VPS_PORT, help=f"SSH port (default: {VPS_PORT})")
    parser.add_argument("--key", default=None, help="Path to SSH private key")
    parser.add_argument("--password", action="store_true", help="Use password authentication (will prompt)")
    args = parser.parse_args()

    password = None
    if args.password:
        import getpass
        password = getpass.getpass(f"SSH password for {args.user}@{args.host}: ")

    print(f"\n🚀 Shortscut VPS Deploy — /transcript endpoint")
    print(f"   Target: {args.user}@{args.host}:{args.port}")

    # ── Step 1: Connect ──
    print_step(1, "Connecting to VPS via SSH")
    try:
        client = connect_ssh(args.host, args.user, args.port, args.key, password)
        print("  ✅ Connected!")
    except Exception as e:
        print(f"  ❌ SSH connection failed: {e}")
        print(f"\n  Tips:")
        print(f"    --key path/to/key    Use a specific SSH key")
        print(f"    --password           Use password auth")
        sys.exit(1)

    try:
        # ── Step 2: Find current server ──
        print_step(2, "Finding current server.py")
        out, _, _ = run_remote(client, "find /root /opt /srv /home -name 'server.py' -path '*shortscut*' 2>/dev/null | head -5")
        if out:
            server_paths = out.strip().split("\n")
            server_py = server_paths[0]
            print(f"  Found: {server_py}")
        else:
            server_py = "/root/shortscut/server.py"
            print(f"  Not found, using default: {server_py}")

        # ── Step 3: Backup ──
        print_step(3, "Backing up current server.py")
        run_remote(client, f"cp '{server_py}' '{server_py}.bak.$(date +%Y%m%d_%H%M%S)' 2>/dev/null")
        print("  ✅ Backup created")

        # ── Step 4: Check if /transcript already exists ──
        print_step(4, "Checking if /transcript endpoint already exists")
        out, _, rc = run_remote(client, f"grep -c '/transcript' '{server_py}' 2>/dev/null || echo 0")
        if out.strip() not in ("0", ""):
            try:
                count = int(out.strip())
                if count > 0:
                    print(f"  ⚠️  /transcript already in server.py ({count} occurrences)")
                    print("  Replacing with latest version from GitHub...")
            except ValueError:
                pass

        # ── Step 5: Download latest server.py from GitHub ──
        print_step(5, "Downloading latest server.py from GitHub")
        server_dir = "/".join(server_py.replace("\\", "/").split("/")[:-1])
        dl_cmd = f"curl -sL '{GITHUB_RAW_URL}' -o '{server_py}' && echo 'OK' || echo 'FAIL'"
        out, err, rc = run_remote(client, dl_cmd, timeout=30)
        if "OK" in out:
            print(f"  ✅ Downloaded to {server_py}")
        else:
            print(f"  ❌ Download failed: {err}")
            sys.exit(1)

        # ── Step 6: Restart server ──
        print_step(6, "Restarting VPS server")
        run_remote(client, f"pkill -f 'python3.*server.py' 2>/dev/null; sleep 1")
        print("  Stopped old server")

        # Start new server
        start_cmd = f"cd '{server_dir}' && nohup python3 '{server_py}' > /var/log/shortscut-server.log 2>&1 & echo $!"
        out, err, rc = run_remote(client, start_cmd)
        pid = out.strip()
        print(f"  Started new server (PID: {pid})")
        time.sleep(3)

        # Verify it's running
        out, _, _ = run_remote(client, f"kill -0 {pid} 2>/dev/null && echo 'RUNNING' || echo 'DEAD'")
        if "RUNNING" in out:
            print("  ✅ Server is running")
        else:
            print("  ❌ Server died! Check logs:")
            out, _, _ = run_remote(client, "tail -20 /var/log/shortscut-server.log")
            print(out)
            sys.exit(1)

        # ── Step 7: Test /health ──
        print_step(7, "Testing /health endpoint")
        out, _, rc = run_remote(client, f"curl -s http://localhost:{SERVER_PORT}/health")
        try:
            health = json.loads(out)
            print(f"  Status: {health.get('status', '?')}")
            print(f"  Server: {health.get('server', '?')}")
            print("  ✅ Health OK")
        except (json.JSONDecodeError, TypeError):
            print(f"  Response: {out}")
            print("  ⚠️  Unexpected response")

        # ── Step 8: Test /transcript ──
        print_step(8, "Testing /transcript endpoint (Rick Astley)")
        test_cmd = (
            f"curl -s -X POST http://localhost:{SERVER_PORT}/transcript "
            f"-H 'Content-Type: application/json' "
            f"-H 'X-API-Key: {API_KEY}' "
            f"-d '{{\"video_url\":\"https://www.youtube.com/watch?v=dQw4w9WgXcQ\",\"lang\":\"en\"}}'"
        )
        out, _, rc = run_remote(client, test_cmd, timeout=120)
        try:
            result = json.loads(out)
            if result.get("success"):
                print(f"  ✅ SUCCESS! {result.get('segment_count', '?')} segments extracted")
                print(f"  Format: {result.get('format', '?')}, Language: {result.get('language', '?')}")
                if result.get("segments"):
                    first = result["segments"][0]
                    print(f"  First segment: [{first['start']:.1f}s] {first['text'][:60]}...")
            else:
                print(f"  ❌ Error: {result.get('error', 'unknown')}")
        except (json.JSONDecodeError, TypeError):
            print(f"  Response: {out[:300]}")

        print(f"\n{'='*60}")
        print(f"  🎉 Deploy complete!")
        print(f"  Server running at http://{args.host}:{SERVER_PORT}")
        print(f"  Endpoints: /health, /process, /transcript, /download/{{id}}")
        print(f"{'='*60}\n")

    finally:
        client.close()


if __name__ == "__main__":
    main()
