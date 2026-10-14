"""Fault-injection driver for the XY1 trainer: start a run in its own process group, wait for N updates, then either send
SIGINT to the group (what the workbench GUI does) or SIGKILL one shard worker; report exit code, markers and leftovers.

usage: resume_fault_test.py {sigint|killworker} RUN_DIR --python PY [--after-updates 3]
"""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("sigint", "killworker"))
ap.add_argument("run_dir", type=Path)
ap.add_argument("--python", required=True)
ap.add_argument("--after-updates", type=int, default=3)
ap.add_argument("--envs", type=int, default=16)
ap.add_argument("--timeout", type=float, default=900)
a = ap.parse_args()

cfg = "casu_grid/workbench/experiments/xy1_point_navigation_mini.py"
cmd = [a.python, "-m", "casu_grid.workbench.train", "--config", cfg, "--num-envs", str(a.envs), "--terrain-mode", "independent",
       "--seed", "7", "--updates", "5000", "--run-dir", str(a.run_dir)]
log = open(str(a.run_dir) + ".log", "wb")
proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
pgid = os.getpgid(proc.pid)
metrics = a.run_dir / "metrics.jsonl"
t0 = time.time()
while time.time() - t0 < a.timeout:
    if proc.poll() is not None:
        print("run ended before the fault (rc=%s)" % proc.returncode); sys.exit(2)
    if metrics.exists() and sum(1 for _ in open(metrics)) >= a.after_updates:
        break
    time.sleep(0.2)
else:
    os.killpg(pgid, signal.SIGKILL); print("timeout waiting for updates"); sys.exit(3)

def workers():
    out = subprocess.run(["pgrep", "-f", "^[^ ]*python3? -m casu_grid[.]isaac_shard_worker"], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out]

if a.mode == "sigint":
    print("SIGINT to group", pgid)
    os.killpg(pgid, signal.SIGINT)
else:
    w = workers()
    print("SIGKILL worker", w[0], "of", w)
    os.kill(w[0], signal.SIGKILL)
t1 = time.time()
try:
    rc = proc.wait(timeout=180)
except subprocess.TimeoutExpired:
    rc = None; print("trainer did not exit within 180 s -> killing group"); os.killpg(pgid, signal.SIGKILL)
print("trainer rc =", rc, "exit latency %.1f s" % (time.time() - t1))
time.sleep(3)
print("files:", sorted(p.name for p in a.run_dir.iterdir()))
fj = a.run_dir / "failure.json"
if fj.exists(): print("failure.json:", fj.read_text()[:500].replace("\n", " "))
print("metrics lines:", sum(1 for _ in open(metrics)))
left = workers()
print("leftover shard workers:", left)
for p in left: os.kill(p, signal.SIGKILL)
