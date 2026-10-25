#!/usr/bin/env python
"""Run a list of shell jobs: GPU jobs one per free XPU, CPU jobs in a bounded pool.

    python scripts/job_queue.py jobs.txt --gpus 0,1,2,4,5,6,7 --cpu-slots 6

Each line of the jobs file is ``gpu|cpu <name> <command ...>``.  Every job's output goes to
``outputs/logs/jobs/<name>.log``; a summary (exit code, wall time) is appended to
``outputs/logs/jobs/summary.tsv``.  Jobs whose log already ends with ``JOB OK`` are skipped,
so the queue can be re-run after a failure.  A job is claimed by atomically creating
``<name>.lock``, so several queue processes (e.g. on different GPU sets) can share one jobs
file without running anything twice; delete the lock of a failed job to retry it.
"""
import argparse
import os
import subprocess
import threading
import time
import queue

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "outputs", "logs", "jobs")


def worker(q, env_extra, lock):
    while True:
        try:
            name, cmd = q.get_nowait()
        except queue.Empty:
            return
        log = os.path.join(LOGS, f"{name}.log")
        if os.path.isfile(log) and open(log).read().rstrip().endswith("JOB OK"):
            continue
        try:                                                  # claim the job atomically
            os.close(os.open(os.path.join(LOGS, f"{name}.lock"), os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        env = {**os.environ, **env_extra, "OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "16"}
        t0 = time.time()
        with open(log, "w") as f:
            f.write(f"$ {cmd}\n# env {env_extra}\n")
            f.flush()
            rc = subprocess.call(cmd, shell=True, cwd=ROOT, env=env, stdout=f,
                                 stderr=subprocess.STDOUT)
            f.write(f"\n# exit {rc} after {time.time() - t0:.0f} s\n")
            f.write("JOB OK\n" if rc == 0 else "JOB FAILED\n")
        with lock:
            with open(os.path.join(LOGS, "summary.tsv"), "a") as s:
                s.write(f"{name}\t{rc}\t{time.time() - t0:.0f}\t{env_extra}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--gpus", default="0,1,2,4,5,6,7")
    ap.add_argument("--cpu-slots", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    gq, cq = queue.Queue(), queue.Queue()
    for line in open(a.jobs):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kind, name, cmd = line.split(maxsplit=2)
        (gq if kind == "gpu" else cq).put((name, cmd))
    lock = threading.Lock()
    th = [threading.Thread(target=worker, args=(gq, {"COSMOS_DEVICE": f"xpu:{g}"}, lock))
          for g in a.gpus.split(",") if g]
    th += [threading.Thread(target=worker, args=(cq, {"COSMOS_DEVICE": "cpu"}, lock))
           for _ in range(a.cpu_slots)]
    for t in th:
        t.start()
    for t in th:
        t.join()


if __name__ == "__main__":
    main()
