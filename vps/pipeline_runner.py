"""
Run the *exact* local ShortsCut script (shortscut_pipeline.py) on the VPS.

The web app's "download the .sh script" flow runs:

    python3 shortscut_pipeline.py URL --api-key K --language L \
        --num-shorts N --min-duration A --max-duration B

This module runs the very same file with the very same arguments (plus
--cookies, because the VPS IP is blocked by YouTube without them, and
--output-dir so each job is isolated), as a subprocess, and exposes its
progress/results to server.py. Nothing about the processing itself is
re-implemented here — download, Whisper, AI analysis, face tracking, silence
removal, subtitles and encoding all come from shortscut_pipeline.py.

Job directory layout (WORK_DIR/pipeline_<id>/):
    cookies.txt      private cookies for this job (deleted when done)
    log.txt          combined stdout/stderr of the script
    clips.json       written by the script after AI analysis
    transcript.json  written by the script after Whisper
    NN_Title.mp4     final shorts
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "shortscut_pipeline.py"

PIPELINE_TTL = 3 * 3600  # keep finished jobs (and their mp4s) for 3h

# Step markers printed by shortscut_pipeline.py → job status used by the app
_STEP_MARKERS = [
    ("Step 1/5", "downloading"),
    ("Step 2/5", "transcribing"),
    ("Step 3/5", "analyzing"),
    ("Step 4/5", "generating"),   # face detection
    ("Step 5/5", "generating"),   # cutting shorts
]

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _write_cookies(cookies_text: str, dest: Path) -> str | None:
    """Accept Netscape cookies.txt or a raw 'a=b; c=d' Cookie header."""
    if not cookies_text or not cookies_text.strip():
        return None
    text = cookies_text.strip()
    if "# Netscape HTTP Cookie File" in text or "\t" in text:
        content = text if text.startswith("#") else "# Netscape HTTP Cookie File\n" + text
    else:
        lines = ["# Netscape HTTP Cookie File"]
        for pair in text.split(";"):
            pair = pair.strip()
            if "=" not in pair:
                continue
            name, value = pair.split("=", 1)
            lines.append(f".youtube.com\tTRUE\t/\tTRUE\t2147483647\t{name.strip()}\t{value.strip()}")
        content = "\n".join(lines)
    path = dest / "cookies.txt"
    path.write_text(content + "\n", encoding="utf-8")
    return str(path)


def start(work_dir: Path, *, youtube_url: str, api_key: str, language: str,
          num_shorts: int, min_duration: int, max_duration: int,
          cookies_text: str | None = None, gpt_model: str | None = None) -> str:
    """Spawn the script. Returns the pipeline id."""
    if not SCRIPT.exists():
        raise RuntimeError(f"{SCRIPT} missing on the VPS")
    pid = uuid.uuid4().hex[:12]
    job_dir = work_dir / f"pipeline_{pid}"
    job_dir.mkdir(parents=True, exist_ok=True)
    cookies_path = _write_cookies(cookies_text or "", job_dir)

    cmd = [
        sys.executable, str(SCRIPT), youtube_url,
        "--api-key", api_key,
        "--language", language or "en",
        "--num-shorts", str(int(num_shorts)),
        "--min-duration", str(int(min_duration)),
        "--max-duration", str(int(max_duration)),
        "--output-dir", str(job_dir),
    ]
    if cookies_path:
        cmd += ["--cookies", cookies_path]

    log_f = open(job_dir / "log.txt", "w", encoding="utf-8")
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    if gpt_model and re.fullmatch(r"[A-Za-z0-9._-]{1,64}", gpt_model):
        env["SHORTSCUT_GPT_MODEL"] = gpt_model  # script default: gpt-5.6-sol
    # Run through a tiny shell wrapper so the exit code lands on disk even if
    # server.py is restarted mid-job; start_new_session detaches the job from
    # the server's process group (systemd KillMode=process leaves it alive).
    wrapper = f"{shlex.join(cmd)}; echo $? > exit_code"
    proc = subprocess.Popen(["bash", "-c", wrapper], stdout=log_f, stderr=subprocess.STDOUT,
                            cwd=str(job_dir), env=env, start_new_session=True)
    created = time.time()
    (job_dir / "meta.json").write_text(json.dumps({
        "id": pid, "os_pid": proc.pid, "created": created, "cookies": cookies_path,
    }), encoding="utf-8")
    with _lock:
        _jobs[pid] = {
            "id": pid, "dir": job_dir, "proc": proc, "log_f": log_f, "os_pid": proc.pid,
            "created": created, "finished": None, "cookies": cookies_path,
        }
    threading.Thread(target=_wait, args=(pid,), daemon=True).start()
    return pid


def _pid_alive(os_pid: int) -> bool:
    try:
        os.kill(os_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # zombie check
    try:
        state = Path(f"/proc/{os_pid}/stat").read_text().split(")")[-1].split()[0]
        return state != "Z"
    except Exception:
        return True


def _exit_code(job: dict) -> int | None:
    """Exit code of the job, or None while it is still running."""
    proc = job.get("proc")
    if proc is not None:
        rc = proc.poll()
        if rc is not None:
            return rc
        return None
    ec = job["dir"] / "exit_code"
    if ec.exists():
        try:
            return int(ec.read_text().strip() or 1)
        except ValueError:
            return 1
    if job.get("os_pid") and _pid_alive(job["os_pid"]):
        return None
    return 137  # process gone without exit code → killed


def recover(work_dir: Path) -> int:
    """Re-attach to jobs started by a previous server process (after restart)."""
    n = 0
    for meta_path in sorted(work_dir.glob("pipeline_*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        pid = meta.get("id")
        if not pid or pid in _jobs:
            continue
        job_dir = meta_path.parent
        rc = None
        if (job_dir / "exit_code").exists():
            rc = 1
        elif not (meta.get("os_pid") and _pid_alive(meta["os_pid"])):
            rc = 137
        job = {
            "id": pid, "dir": job_dir, "proc": None, "log_f": None, "os_pid": meta.get("os_pid"),
            "created": meta.get("created", job_dir.stat().st_mtime),
            "finished": None, "cookies": meta.get("cookies"),
        }
        if rc is not None:
            job["finished"] = (job_dir / "exit_code").stat().st_mtime if (job_dir / "exit_code").exists() \
                else max((p.stat().st_mtime for p in job_dir.iterdir()), default=time.time())
            if not (job_dir / "exit_code").exists():
                (job_dir / "exit_code").write_text("137\n")
            _finalize(job)
        with _lock:
            _jobs[pid] = job
        if rc is None:
            threading.Thread(target=_wait, args=(pid,), daemon=True).start()
        n += 1
    return n


def _finalize(job: dict):
    if job.get("cookies") and os.path.exists(job["cookies"]):
        os.unlink(job["cookies"])
    # the script leaves the big source video behind; we only need the shorts
    for f in job["dir"].glob("source.*"):
        f.unlink(missing_ok=True)
    for f in job["dir"].glob("audio*.mp3"):
        f.unlink(missing_ok=True)


def _wait(pid: str):
    job = _jobs[pid]
    if job.get("proc") is not None:
        job["proc"].wait()
        job["log_f"].close()
    else:  # recovered job: poll the detached process
        while _exit_code(job) is None:
            time.sleep(5)
    job["finished"] = time.time()
    _finalize(job)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def status(pid: str) -> dict | None:
    job = _jobs.get(pid)
    if not job:
        return None
    d: Path = job["dir"]
    log_text = ""
    try:
        log_text = (d / "log.txt").read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        pass

    step = "downloading"
    for marker, st in _STEP_MARKERS:
        if marker in log_text:
            step = st

    m = re.search(r"Title:\s*(.+)", log_text)
    title = m.group(1).strip() if m else None

    clips = _read_json(d / "clips.json")
    outputs = sorted(
        [p for p in d.glob("*.mp4") if re.match(r"\d{2}_", p.name)],
        key=lambda p: p.name,
    )
    outputs_info = [{
        "index": int(p.name[:2]) - 1,
        "name": p.name,
        "size": p.stat().st_size,
        "download_url": f"/pipeline/{pid}/file/{p.name}",
    } for p in outputs if p.stat().st_size > 0]

    rc = _exit_code(job)
    if rc is None:
        state = "running"
        error = None
    elif rc == 0 and outputs_info:
        state = "completed"
        error = None
    else:
        state = "failed"
        # last meaningful lines from the script (it prints ❌ on fatal errors)
        lines = [l.strip() for l in log_text.splitlines() if l.strip()]
        err_lines = [l for l in lines if "❌" in l or "Error" in l or "error" in l]
        error = " | ".join((err_lines or lines)[-4:]) or f"script exited with code {rc}"
        if rc == 137 and not err_lines:
            error = "Procesul a fost oprit (VPS repornit / memorie insuficientă). " + error

    return {
        "id": pid,
        "state": state,
        "step": step,
        "video_title": title,
        "clips": clips,
        "outputs": outputs_info,
        "error": error,
        "log_tail": "\n".join(log_text.splitlines()[-25:]),
        "elapsed": round((job["finished"] or time.time()) - job["created"]),
    }


def file_path(pid: str, name: str) -> Path | None:
    job = _jobs.get(pid)
    if not job or "/" in name or ".." in name:
        return None
    p = job["dir"] / name
    return p if p.exists() and p.suffix == ".mp4" else None


def cleanup(now: float | None = None):
    now = now or time.time()
    with _lock:
        stale = [pid for pid, j in _jobs.items()
                 if j["finished"] and now - j["finished"] > PIPELINE_TTL]
        for pid in stale:
            j = _jobs.pop(pid)
            shutil.rmtree(j["dir"], ignore_errors=True)


