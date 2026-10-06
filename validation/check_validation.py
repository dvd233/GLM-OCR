#!/usr/bin/env python3
"""Bound, no-secret comparison of two immutable GLM-OCR source trees."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

CONTROL = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("GITHUB_WORKSPACE", Path.cwd()))
LOGS = ROOT / "validation-logs"
LOGS.mkdir(exist_ok=True)
MANIFEST = json.loads((CONTROL / "manifest.json").read_text())
STATUS = {name: {"state": "not_run"} for name in (
    "base-install", "base-native", "candidate-install", "candidate-native", "precommit-all"
)}


def save_status():
    (LOGS / "status.json").write_text(json.dumps(STATUS, indent=2) + "\n")


def record(name, command, cwd):
    """Preserve actual exit status and output; never turn a failure green."""
    started = time.time()
    STATUS[name] = {"state": "running", "started_at_unix": started}
    save_status()
    print(f"\n== {name}: {' '.join(command)} ==", flush=True)
    with (LOGS / f"{name}.log").open("w") as stream:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        code = process.wait()
    STATUS[name] = {"state": "completed", "exit_code": code, "seconds": round(time.time() - started, 2)}
    save_status()
    return code


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight():
    for name in ("validation_repository", "candidate_repository"):
        value = MANIFEST[name]
        if not re.fullmatch(r"dvd233/[A-Za-z0-9_.-]+", value):
            raise ValueError(f"Unbound or unapproved {name}: {value}")
    for name in ("base_commit", "base_tree", "candidate_commit", "candidate_tree"):
        if not re.fullmatch(r"[0-9a-f]{40}", MANIFEST[name]):
            raise ValueError(f"Unbound {name}; Root must bind immutable SHA before publication")
    if os.environ.get("GITHUB_REPOSITORY") != MANIFEST["validation_repository"]:
        raise ValueError("Workflow repository differs from reviewed static manifest")
    expected_ref = "refs/heads/" + MANIFEST["validation_branch"]
    if os.environ.get("GITHUB_REF") != expected_ref:
        raise ValueError("Workflow branch differs from reviewed static manifest")
    if os.environ.get("GITHUB_EVENT_NAME") != "push":
        raise ValueError("Only the fixed validation-branch push is permitted")
    for key in ("ZHIPU_API_KEY", "GLMOCR_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(key):
            raise ValueError(f"Unexpected nonempty {key}; no credentials are permitted")
    (LOGS / "manifest.json").write_text(json.dumps(MANIFEST, indent=2) + "\n")
    STATUS["preflight"] = {"exit_code": 0}
    save_status()


def verify_sources():
    base, candidate = ROOT / "baseline", ROOT / "candidate"
    for name, source in (("base", base), ("candidate", candidate)):
        if git(source, "rev-parse", "HEAD") != MANIFEST[f"{name}_commit"]:
            raise ValueError(f"{name} commit mismatch")
        if git(source, "rev-parse", "HEAD^{tree}") != MANIFEST[f"{name}_tree"]:
            raise ValueError(f"{name} complete tree mismatch")
        if git(source, "status", "--porcelain"):
            raise ValueError(f"{name} checkout is dirty before validation")
        for fixture in MANIFEST["fixtures"]:
            path = source / fixture["path"]
            if sha256(path) != fixture["sha256"]:
                raise ValueError(f"{name} fixture mismatch: {fixture['path']}")
        for item in MANIFEST["changed_files"]:
            key = "baseline_sha256" if name == "base" else "fixed_sha256"
            if sha256(source / item["path"]) != item[key]:
                raise ValueError(f"{name} frozen source mismatch: {item['path']}")
    parents = [line[7:] for line in git(candidate, "cat-file", "-p", "HEAD").splitlines()
               if line.startswith("parent ")]
    if parents != [MANIFEST["candidate_parent"]]:
        raise ValueError("Candidate must have the fixed baseline as its sole parent")
    def tree_entries(source):
        lines = git(source, "ls-tree", "-r", "HEAD").splitlines()
        return {line.split("\t", 1)[1]: line.split("\t", 1)[0] for line in lines}
    old, new = tree_entries(base), tree_entries(candidate)
    changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    expected = sorted(item["path"] for item in MANIFEST["changed_files"])
    if changed != expected:
        raise ValueError(f"Unexpected source changes: {changed}")
    STATUS["source_binding"] = {"exit_code": 0, "changed_files": changed}
    save_status()
    return base, candidate


def strict_test_count(name, expected):
    text = (LOGS / f"{name}.log").read_text()
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    summaries = re.findall(r"^=+ (.*?) =+$", text, flags=re.MULTILINE)
    good = bool(summaries and re.fullmatch(
        rf"{expected} passed(?:, \d+ warnings?)? in .+", summaries[-1]
    ))
    STATUS[name]["expected_passed"] = expected
    STATUS[name]["zero_skip_exact_count"] = good
    save_status()
    return STATUS[name]["exit_code"] == 0 and good


def dependency_snapshot(name, cwd):
    if record(name, ["python", "-m", "pip", "freeze", "--all"], cwd):
        return None
    lines = (LOGS / f"{name}.log").read_text().splitlines()
    return sorted(line for line in lines if not (
        line.startswith("# Editable") or line.startswith("-e ")
        or re.match(r"(?i)^glmocr(?:==| @ )", line)
    ))


def assert_freeze(candidate):
    if git(candidate, "status", "--porcelain"):
        return False
    return all(sha256(candidate / item["path"]) == item["fixed_sha256"]
               for item in MANIFEST["changed_files"])


def run():
    preflight()
    if sys.argv[1:] == ["--preflight"]:
        return 0
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Native validation requires Python 3.12")
    base, candidate = verify_sources()
    record("runner-disk-before", ["df", "-h"], ROOT)
    record("runner-memory", ["free", "-h"], ROOT)
    record("python-version", ["python", "--version"], ROOT)
    if record("upgrade-pip", ["python", "-m", "pip", "install", "--upgrade", "pip"], base):
        return 1
    if record("base-install", ["pip", "install", "-e", ".[dev,server,layout]"], base):
        record("runner-disk-install-failure", ["df", "-h"], ROOT)
        return 1
    initial = dependency_snapshot("base-pip-freeze", base)
    checks = [initial is not None]
    checks.append(record("base-pip-check", ["python", "-m", "pip", "check"], base) == 0)
    imports = (
        "import cv2, torch, torchvision, transformers, flask; "
        "from transformers import PPDocLayoutV3ForObjectDetection, PPDocLayoutV3ImageProcessor; "
        "from importlib.metadata import version; "
        "print({name: version(name) for name in "
        "['opencv-python-headless','torch','torchvision','transformers','flask']})"
    )
    checks.append(record("layout-imports", ["python", "-c", imports], base) == 0)
    record("base-native", ["pytest", "glmocr/tests/test_unit.py", "-v", "--tb=short"], base)
    checks.append(strict_test_count("base-native", MANIFEST["expected_passed"]["base"]))
    if record("candidate-install", ["pip", "install", "-e", ".[dev,server,layout]"], candidate):
        record("runner-disk-install-failure", ["df", "-h"], ROOT)
        return 1
    updated = dependency_snapshot("candidate-pip-freeze", candidate)
    same_dependencies = initial is not None and initial == updated
    STATUS["dependency_parity"] = {"exit_code": 0 if same_dependencies else 1}
    checks.append(same_dependencies)
    checks.append(record("candidate-pip-check", ["python", "-m", "pip", "check"], candidate) == 0)
    record("candidate-native", ["pytest", "glmocr/tests/test_unit.py", "-v", "--tb=short"], candidate)
    checks.append(strict_test_count("candidate-native", MANIFEST["expected_passed"]["candidate"]))
    checks.append(record("precommit-install", ["pip", "install", "pre-commit"], candidate) == 0)
    checks.append(record("precommit-all", ["pre-commit", "run", "-a", "--show-diff-on-failure"], candidate) == 0)
    checks.append(record("diff-check", ["git", "diff", "--check"], candidate) == 0)
    record("post-check-diff", ["git", "diff", "--binary", "HEAD"], candidate)
    record("post-check-status", ["git", "status", "--porcelain"], candidate)
    unchanged = assert_freeze(candidate)
    STATUS["source_unchanged_after_checks"] = {"exit_code": 0 if unchanged else 1}
    checks.append(unchanged)
    record("runner-disk-after", ["df", "-h"], ROOT)
    STATUS["overall"] = {"exit_code": 0 if all(checks) else 1}
    save_status()
    return STATUS["overall"]["exit_code"]


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception as exc:
        STATUS["harness_error"] = {"exit_code": 1, "type": type(exc).__name__, "message": str(exc)}
        save_status()
        print(f"Validation stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
