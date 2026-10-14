"""Author the executable Kaggle notebook locally; does not execute lab code."""
import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
mode = sys.argv[1] if len(sys.argv) > 1 else "preflight"
assert mode in {"preflight", "full"}
files = [ROOT / name for name in ("README.md", "GUIDE.md", "RUBRIC.md", "REPORT.md",
                                  "requirements-colab.txt", "pytest.ini")]
for folder in ("scripts", "tests", "configs", "student"):
    files.extend(p for p in sorted((ROOT / folder).iterdir()) if p.suffix in {".py", ".yaml"})
payload, manifest = {}, {}
for path in files:
    name = path.relative_to(ROOT).as_posix()
    raw = path.read_bytes()
    payload[name] = base64.b64encode(raw).decode()
    manifest[name] = hashlib.sha256(raw).hexdigest()
config = {"mode": mode, "max_new_tokens": 512, "optimizer_steps": 300,
          "batch_size": 1, "grad_accum": 8, "seed": 42,
          "note": "batch 1/accum 8 retains effective batch 8; completion cap 512 shared by all setups"}
bootstrap = '''import base64, hashlib, json, os, subprocess, sys
from pathlib import Path
root = Path('/kaggle/working/day10-source')
root.mkdir(exist_ok=False)
payload = json.loads(PAYLOAD)
manifest = json.loads(MANIFEST)
for relative, value in payload.items():
    raw = base64.b64decode(value)
    assert hashlib.sha256(raw).hexdigest() == manifest[relative]
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
(root / 'source-manifest.json').write_text(json.dumps(manifest, indent=2))
(root / 'run-config.json').write_text(CONFIG)
os.chdir(root)
os.environ['HF_HOME'] = '/kaggle/temp/day10-hf-cache'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
deps = '/kaggle/temp/day10-deps'
# Install into a fresh directory; preserve Kaggle's existing packages.
subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', deps, '--no-deps',
    'transformers==5.19.0', 'peft==0.21.2', 'trl==1.9.2', 'accelerate==1.15.0',
    'bitsandbytes==0.50.2', 'huggingface-hub==1.33.0', 'torchao==0.18.0'], check=True)
os.environ['PYTHONPATH'] = deps + ':' + str(root)
subprocess.run(['nvidia-smi', '-L'], check=True)
'''.replace("PAYLOAD", repr(json.dumps(payload))).replace("MANIFEST", repr(json.dumps(manifest))).replace("CONFIG", repr(json.dumps(config)))

def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(True)}

cells = [markdown('# Lab Day10 — Qwen3-VL road damage on Kaggle\n'
                  'Student code is in student/. Instructor scripts/tests are unchanged.\n'),
         markdown('## 0. Môi trường và nguồn local\n'), code(bootstrap)]
for number, description in enumerate(("Dữ liệu: split theo session, test công cụ đề", "Nhãn YOLO → JSON 0–1000",
    "E0: greedy zero-shot, lưu raw", "Chat template, mask loss và collator", "Setup LoRA E1/E2/E3 và projector float32",
    "Train: 300 optimizer steps, checkpoint", "Reload, val và test với scorer gốc", "E4: 10%/100%, báo cáo và audit"), 1):
    cells.append(markdown(f'## {number}. {description}\n'
        'Triển khai trong student/pipeline.py và các module student/ được đóng gói ở cell 0.\n'))
cells.append(code('subprocess.run([sys.executable, "-u", "-m", "student.pipeline"], check=True)\n'))
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
# Include a source snapshot with the same executable cells for the original notebook checks.
# Snapshot is made before inserting itself, preventing recursive expansion.
snapshot = json.dumps(nb, ensure_ascii=False, indent=1).encode()
payload['lab_day10.ipynb'] = base64.b64encode(snapshot).decode()
manifest['lab_day10.ipynb'] = hashlib.sha256(snapshot).hexdigest()
cells[2] = code(bootstrap.replace(repr(json.dumps({k: v for k, v in payload.items() if k != 'lab_day10.ipynb'})),
                                repr(json.dumps(payload))).replace(
                         repr(json.dumps({k: v for k, v in manifest.items() if k != 'lab_day10.ipynb'})),
                         repr(json.dumps(manifest))))
nb_path = ROOT / 'lab_day10.ipynb'
backup = ROOT / f'kaggle_tools/before-{mode}-notebook.ipynb'
if not backup.exists():
    with backup.open('xb') as stream:
        stream.write(nb_path.read_bytes())
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
(ROOT / 'kaggle-source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Authored', mode, 'notebook, source files:', len(files))
