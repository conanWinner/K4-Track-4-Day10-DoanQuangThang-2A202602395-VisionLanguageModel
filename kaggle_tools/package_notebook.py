"""Package the existing lab template for Kaggle; never execute lab code locally."""
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "lab_day10.ipynb"
backup = ROOT / "lab_day10.colab-original.ipynb"
original = path.read_bytes()
if backup.exists():
    raise SystemExit("Backup already exists; inspect before repackaging.")
with backup.open("xb") as stream:
    stream.write(original)
nb = json.loads(original)
files = [ROOT / name for name in (
    "README.md", "GUIDE.md", "RUBRIC.md", "REPORT.md", "requirements-colab.txt", "pytest.ini"
)]
for folder in ("scripts", "tests", "configs"):
    files.extend(p for p in sorted((ROOT / folder).iterdir()) if p.suffix in (".py", ".yaml"))
payload = {}
manifest = {}
for p in files:
    relative = p.relative_to(ROOT).as_posix()
    raw = p.read_bytes()
    payload[relative] = base64.b64encode(raw).decode("ascii")
    manifest[relative] = hashlib.sha256(raw).hexdigest()

def source(cell, text):
    cell["source"] = text.splitlines(keepends=True)

setup = '''# Kaggle setup: restore only allowlisted source files, without credentials.
import base64, hashlib, json, os, subprocess, sys
from pathlib import Path
root = Path("/kaggle/working/day10-source")
root.mkdir(exist_ok=False)
payload = json.loads(PAYLOAD_JSON)
manifest = json.loads(MANIFEST_JSON)
for relative, encoded in payload.items():
    raw = base64.b64decode(encoded)
    assert hashlib.sha256(raw).hexdigest() == manifest[relative], relative
    dest = root / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
(root / "source-manifest.json").write_text(json.dumps(manifest, indent=2))
os.chdir(root)
subprocess.run(["nvidia-smi", "-L"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-colab.txt"], check=True)
'''.replace("PAYLOAD_JSON", repr(json.dumps(payload))).replace("MANIFEST_JSON", repr(json.dumps(manifest)))
for cell in nb["cells"]:
    text = "".join(cell["source"])
    if cell["cell_type"] == "markdown" and "## 0. Môi trường" in text:
        source(cell, "## 0. Môi trường (Kaggle GPU)\n")
    elif cell["cell_type"] == "code" and "pip -q install" in text:
        source(cell, setup)
    elif cell["cell_type"] == "code" and "from google.colab" in text:
        source(cell, '# Dataset is attached with kernel-metadata.json; no Kaggle key is needed here.\nprint("Dataset mount:", list(Path("/kaggle/input").iterdir()))\n')
    elif cell["cell_type"] == "code" and "prepare_data.py --kaggle" in text:
        source(cell, '''# Use the attached dataset and the unchanged instructor preparation script.
candidates = [p for p in Path("/kaggle/input").rglob("images")
              if p.is_dir() and (p.parent / "labels-YOLO").is_dir()]
if len(candidates) != 1:
    raise RuntimeError(f"Expected one dataset root, found: {candidates}")
subprocess.run([sys.executable, "scripts/prepare_data.py", "--source", str(candidates[0].parent)], check=True)
''')
    elif cell["cell_type"] == "code" and "!python scripts/score.py" in text:
        source(cell, '''# Fail explicitly if E0 predictions have not been implemented/generated.
subprocess.run([sys.executable, "scripts/score.py", "--pred", "work/e0.jsonl",
                "--name", "E0", "--out", "report/evidence/e0.json"], check=True)
''')
nb["metadata"].pop("colab", None)
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
(ROOT / "kaggle-source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
metadata = {
    "id": "thngonquang/track4-day10-qwen3-vl-road-damage",
    "title": "Track4 Day10 Qwen3 VL Road Damage",
    "code_file": "lab_day10.ipynb", "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": ["lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"],
    "competition_sources": [], "kernel_sources": []
}
with (ROOT / "kernel-metadata.json").open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(metadata, indent=2) + "\n")
print(f"Packaged {len(payload)} source files; TODO cells preserved; backup: {backup}")
