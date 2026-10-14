"""Verify the published submission on Kaggle only; never train or rescore test."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/conanWinner/K4-Track-4-Day10-DoanQuangThang-2A202602395-VisionLanguageModel.git"
ROOT = Path("/kaggle/working/github-submission")
OUT = Path("/kaggle/working/submission-check.json")
EXPECTED = os.environ["DAY10_EXPECTED_SHA"]
result = {"expected_sha": EXPECTED, "repo": REPO}


def run(args, cwd=None):
    p = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    print(p.stdout, flush=True)
    assert p.returncode == 0, (args, p.returncode)
    return p.stdout


try:
    assert str(Path.cwd()).startswith("/kaggle/"), "Execute on Kaggle only"
    run(["git", "clone", REPO, str(ROOT)])
    run(["git", "checkout", "--detach", EXPECTED], ROOT)
    result["checked_sha"] = run(["git", "rev-parse", "HEAD"], ROOT).strip()
    assert result["checked_sha"] == EXPECTED
    result["commit_count"] = int(run(["git", "rev-list", "--count", "HEAD"], ROOT))
    assert result["commit_count"] >= 5
    assert not run(["git", "diff", "c2f3997", "HEAD", "--", "scripts", "tests"], ROOT).strip()
    manifest = json.loads((ROOT / "report/source-manifest.json").read_text())
    for name, digest in manifest.items():
        if name.startswith(("scripts/", "tests/", "student/")):
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    sources = [p.parent for p in Path("/kaggle/input").rglob("labels-YOLO")
               if p.is_dir() and (p.parent / "images").is_dir()]
    assert len(sources) == 1, sources
    os.environ["D10_RAW"] = str(sources[0])
    run([sys.executable, "scripts/prepare_data.py", "--source", str(sources[0])], ROOT)
    result["tests"] = run([sys.executable, "-m", "pytest", "-q", "tests",
                           "student/test_student.py", "--deselect",
                           "tests/test_configs_and_docs.py::test_report_template_has_student_markers_and_ten_questions"], ROOT)
    result["checker"] = run([sys.executable, "scripts/check_submission.py"], ROOT)
    assert "0 FAIL, 0 WARN" in result["checker"]
    evidence = {p.stem: json.loads(p.read_text()) for p in (ROOT / "report/evidence").glob("*.json")}
    assert set(evidence) == {"e0", "e1", "e2", "e3", "e4_data_size_0.1", "e4_data_size_1.0"}
    for name, d in evidence.items():
        assert d["n_images"] == 311 and d["missing_predictions"] == 0
        assert d["dataset_version"] == "d10-21e4070301"
        if name != "e0":
            assert d["meta"]["optimizer_steps"] == 300
    for setup in ("e1", "e2", "e3", "e4_data_size_0.1"):
        losses = [json.loads(s) for s in (ROOT / f"report/training/{setup}/loss.jsonl").read_text().splitlines()]
        assert len(losses) == 300 and [x["step"] for x in losses] == list(range(1, 301))
    assert evidence["e1"]["parse_valid_rate"] >= .95
    assert evidence["e1"]["micro"]["f1"] - evidence["e0"]["micro"]["f1"] >= .10
    report = (ROOT / "REPORT.md").read_text()
    assert len(re.findall(r"^### Câu \d+", report, re.M)) == 10 and "<điền" not in report
    for relative in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report):
        assert (ROOT / relative).is_file(), relative
    assert len(list((ROOT / "report/figures").glob("prediction-*.png"))) >= 6
    assert not run(["git", "status", "--porcelain"], ROOT).strip()
    result["status"] = "PASS"
    print("SUBMISSION_READY: 0 FAIL, 0 WARN", flush=True)
except Exception as exc:
    result["status"] = "FAIL"
    result["error"] = repr(exc)
    raise
finally:
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
