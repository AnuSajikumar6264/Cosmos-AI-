"""Verify that a pushed commit on GitHub reproduces the local tree byte for byte.

    python scripts/verify_github_upload.py --clone-dir /tmp/cosmos_verify [--ref <branch>]

Clones the remote into a scratch directory, runs ``git lfs pull``, then compares every file
in the pushed commit against the local repository: presence, size, and SHA-256.  Files
carried in Git LFS are checked against the pointer OID, which *is* the SHA-256 of the
content, so a file that came back as an unresolved pointer fails rather than passing quietly.

Exit status is 0 only if every file matches.  Nothing is written inside the project.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LFS = shutil.which("git-lfs") or "/data/anupa/COSMOS_work/.tools/bin/git-lfs"


def run(cmd, cwd=None, check=True):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode:
        print(f"FAILED: {' '.join(cmd)}\n{r.stderr[-2000:]}")
        raise SystemExit(2)
    return r.stdout


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def lfs_oids(repo):
    """{path: (oid, size)} for every LFS-tracked file in the checked-out ref."""
    out = {}
    for line in run([LFS, "ls-files", "-l"], cwd=repo).splitlines():
        # "<oid> <*|-> <path>"
        parts = line.split(" ", 2)
        if len(parts) == 3:
            out[parts[2].strip()] = parts[0].strip()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone-dir", default="/tmp/cosmos_verify")
    ap.add_argument("--ref", default=None, help="branch to verify (default: current local branch)")
    ap.add_argument("--remote", default=None, help="clone URL (default: this repo's origin)")
    ap.add_argument("--keep", action="store_true", help="keep the clone for inspection")
    a = ap.parse_args(argv)

    ref = a.ref or run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT).strip()
    remote = a.remote or run(["git", "remote", "get-url", "origin"], cwd=ROOT).strip()
    local_sha = run(["git", "rev-parse", "HEAD"], cwd=ROOT).strip()
    print(f"local  {ROOT}\n       branch {ref} at {local_sha}")
    print(f"remote {remote}")

    if os.path.isdir(a.clone_dir):
        shutil.rmtree(a.clone_dir)
    os.makedirs(os.path.dirname(a.clone_dir) or "/tmp", exist_ok=True)

    print("\ncloning...")
    run(["git", "clone", "--branch", ref, remote, a.clone_dir])
    print("git lfs pull...")
    env = dict(os.environ, PATH=os.path.dirname(LFS) + os.pathsep + os.environ.get("PATH", ""))
    r = subprocess.run([LFS, "pull"], cwd=a.clone_dir, capture_output=True, text=True, env=env)
    if r.returncode:
        print(f"  git lfs pull returned {r.returncode}: {r.stderr[-1500:]}")

    remote_sha = run(["git", "rev-parse", "HEAD"], cwd=a.clone_dir).strip()
    print(f"\ncommit  local {local_sha}\n        remote {remote_sha}"
          f"  -> {'SAME' if local_sha == remote_sha else 'DIFFERENT'}")

    local_files = set(run(["git", "ls-files"], cwd=ROOT).splitlines())
    remote_files = set(run(["git", "ls-files"], cwd=a.clone_dir).splitlines())
    missing = sorted(local_files - remote_files)
    extra = sorted(remote_files - local_files)
    print(f"\nfiles   local {len(local_files)}  remote {len(remote_files)}"
          f"  missing {len(missing)}  extra {len(extra)}")
    for p in missing[:20]:
        print(f"  MISSING  {p}")
    for p in extra[:20]:
        print(f"  EXTRA    {p}")

    oids = lfs_oids(a.clone_dir)
    print(f"\nLFS-tracked in clone: {len(oids)} files")

    bad, pointer, checked, total_bytes = [], [], 0, 0
    for rel in sorted(local_files & remote_files):
        lp, rp = os.path.join(ROOT, rel), os.path.join(a.clone_dir, rel)
        if not os.path.isfile(lp) or not os.path.isfile(rp):
            continue
        ls, rs = os.path.getsize(lp), os.path.getsize(rp)
        if rel in oids:
            # an unresolved pointer is ~130 bytes of text, never the real content
            if rs != ls:
                pointer.append((rel, ls, rs))
                continue
            if sha256(rp) != oids[rel]:
                bad.append((rel, "LFS content does not match pointer OID"))
                continue
        else:
            if ls != rs:
                bad.append((rel, f"size {ls} vs {rs}"))
                continue
            if sha256(lp) != sha256(rp):
                bad.append((rel, "sha256 differs"))
                continue
        checked += 1
        total_bytes += rs

    print(f"verified {checked} files, {total_bytes/1024**3:.3f} GiB, byte-identical")
    for rel, ls, rs in pointer[:20]:
        print(f"  POINTER ONLY  {rel}  local {ls} vs remote {rs} bytes")
    for rel, why in bad[:20]:
        print(f"  MISMATCH      {rel}: {why}")

    ok = not missing and not bad and not pointer and local_sha == remote_sha
    print("\nRESULT:", "PASS — remote reproduces the local tree" if ok else "FAIL — see above")
    if not a.keep:
        shutil.rmtree(a.clone_dir, ignore_errors=True)
    else:
        print(f"clone kept at {a.clone_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
