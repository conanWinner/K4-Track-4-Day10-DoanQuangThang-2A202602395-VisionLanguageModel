"""Run only on Kaggle: preflight or the full mandatory experiment sequence."""
import gc
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import torch
from PIL import ImageDraw

from scripts.score import evaluate
from student.data import PROMPT, collate, encode_example, image_for, objects_to_answer, read_rows, subset
from student.infer import predict
from student.model import MODEL_ID, MODEL_REVISION, apply_setup, architecture, load_base, load_checkpoint, load_processor
from student.train import train

ROOT = Path.cwd()
DATA = ROOT / "data"
WORK = ROOT / "work"
EV = ROOT / "report/evidence"
VAL = ROOT / "report/validation"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare():
    roots = [p.parent for p in Path("/kaggle/input").rglob("images")
             if p.is_dir() and (p.parent / "labels-YOLO").is_dir()]
    assert len(roots) == 1, roots
    env = dict(os.environ, D10_RAW=str(roots[0]))
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "student/test_student.py"],
                   env=env, check=True)
    subprocess.run([sys.executable, "scripts/prepare_data.py", "--source", str(roots[0])], check=True)
    splits = {name: read_rows(DATA / f"splits/{name}.jsonl") for name in ("train", "val", "test")}
    sessions = {name: {r["session"] for r in rows} for name, rows in splits.items()}
    assert not sessions["train"] & sessions["val"]
    assert not sessions["train"] & sessions["test"]
    assert not sessions["val"] & sessions["test"]
    meta = json.loads((DATA / "splits/meta.json").read_text())
    assert meta["dataset_version"] == "d10-21e4070301", meta
    assert [len(splits[n]) for n in ("train", "val", "test")] == [1422, 276, 311]
    for rows in splits.values():
        for row in rows:
            objects_to_answer(row["objects"])
    invalid = {name: [{"image_id": row["image_id"], "object": obj} for row in rows
                     for obj in row["objects"] if obj["yolo"][2] <= 0 or obj["yolo"][3] <= 0]
               for name, rows in splits.items()}
    write_json(ROOT / "report/degenerate-labels.json", invalid)
    for i, row in enumerate(splits["train"][:6]):
        image = image_for(row, DATA)
        draw = ImageDraw.Draw(image)
        for obj in json.loads(objects_to_answer(row["objects"])):
            box = obj["bbox_2d"]
            xy = [box[0] * image.width / 1000, box[1] * image.height / 1000,
                  box[2] * image.width / 1000, box[3] * image.height / 1000]
            draw.rectangle(xy, outline="lime", width=2)
            draw.text((xy[0], xy[1]), obj["label"], fill="lime")
        (ROOT / "report/figures").mkdir(parents=True, exist_ok=True)
        image.save(ROOT / f"report/figures/labels-{i}.png")
    write_json(ROOT / "report/data-audit.json", meta)
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "transformers", "peft", "accelerate", "bitsandbytes", "trl", "torchao")}
    write_json(ROOT / "report/environment.json", dict(versions=versions, gpu=torch.cuda.get_device_name(),
                cuda=torch.version.cuda, python=sys.version, prompt=PROMPT, model_id=MODEL_ID,
                model_revision=MODEL_REVISION))
    return splits


def memory_release():
    gc.collect()
    torch.cuda.empty_cache()


def validate_encoding(processor, splits):
    examples = [encode_example(row, DATA, processor) for row in splits["train"][:2]]
    image_tokens = int((examples[0]["input_ids"] == processor.image_token_id).sum())
    decoded = processor.tokenizer.decode(examples[0]["labels"][examples[0]["labels"] != -100],
                                         skip_special_tokens=False)
    assert image_tokens == 220, image_tokens
    assert decoded.startswith(objects_to_answer(splits["train"][0]["objects"]))
    batch = collate(examples, processor.tokenizer.pad_token_id)
    assert batch["input_ids"].shape == batch["labels"].shape
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == -100)
    write_json(ROOT / "report/encoding-audit.json", dict(image_tokens=image_tokens,
                 supervised_tokens=int((examples[0]["labels"] != -100).sum()),
                 decoded_supervised=decoded, image_grid_thw=examples[0]["image_grid_thw"].tolist()))


def score_file(name, predictions, split="test", meta=None, output=None):
    output = output or (EV / f"{name.lower()}.json" if split == "test" else VAL / f"{name.lower()}.json")
    command = [sys.executable, "scripts/score.py", "--pred", str(predictions), "--name", name,
               "--split", split, "--out", str(output)]
    for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
        if meta is not None:
            command += ["--meta", f"{key}={meta[key]}"]
    if meta:
        for key in ("optimizer_steps", "train_images", "setup", "batch_size", "grad_accum"):
            command += ["--meta", f"{key}={meta[key]}"]
    subprocess.run(command, check=True)
    return json.loads(Path(output).read_text())


def preflight(splits, config):
    processor = load_processor()
    validate_encoding(processor, splits)
    results = {}
    for setup in ("E1", "E2", "E3"):
        seed_all()
        base = load_base()
        arch = architecture(base)
        write_json(ROOT / "report/architecture.json", arch)
        model, info = apply_setup(base, setup)
        del base
        # Two optimizer steps, on train only; these are not final experiment results.
        run = train(model, processor, splits["train"][:16], DATA, WORK / f"smoke-{setup}", info,
                    steps=2, batch_size=1, accum=8)
        del model
        memory_release()
        model = load_checkpoint(load_base(), WORK / f"smoke-{setup}/final")
        output = WORK / f"smoke-{setup}-val.jsonl"
        raw = predict(model, processor, splits["val"][:2], DATA, output, batch_size=2,
                      max_new_tokens=config["max_new_tokens"])
        result = evaluate(splits["val"][:2], {r["image_id"]: r["raw"] for r in raw}, n_boot=100)
        results[setup] = {"train": run, "validation": result, "raw": raw}
        del model
        memory_release()
    assert results["E2"]["train"]["trainable_params"] - results["E1"]["train"]["trainable_params"] == results["E2"]["train"]["projector_trainable_params"]
    write_json(ROOT / "report/preflight.json", results)
    print("PREFLIGHT_COMPLETE", flush=True)


def trained_experiment(setup, name, splits, processor, config, fraction=1.0):
    seed_all()
    model, info = apply_setup(load_base(), setup)
    rows = splits["train"] if fraction == 1 else subset(splits["train"], fraction)
    info.update(name=name, train_fraction=fraction, max_new_tokens=config["max_new_tokens"])
    folder = WORK / name.lower()
    run = train(model, processor, rows, DATA, folder, info, steps=300, batch_size=1, accum=8)
    del model
    memory_release()
    model = load_checkpoint(load_base(), folder / "final")
    val_path = WORK / f"{name.lower()}-val.jsonl"
    predict(model, processor, splits["val"], DATA, val_path, max_new_tokens=config["max_new_tokens"])
    validation = score_file(name, val_path, split="val", meta=run)
    test_path = WORK / f"{name.lower()}.jsonl"
    predict(model, processor, splits["test"], DATA, test_path, max_new_tokens=config["max_new_tokens"])
    score_file(name, test_path, meta=run)
    del model
    memory_release()
    return validation


def full(splits, config):
    processor = load_processor()
    validate_encoding(processor, splits)
    seed_all()
    model = load_base()
    write_json(ROOT / "report/architecture.json", architecture(model))
    predict(model, processor, splits["val"][:40], DATA, WORK / "e0-val-debug.jsonl",
            max_new_tokens=config["max_new_tokens"])
    predict(model, processor, splits["test"], DATA, WORK / "e0.jsonl",
            max_new_tokens=config["max_new_tokens"])
    score_file("E0", WORK / "e0.jsonl")
    del model
    memory_release()
    validations = {}
    for setup in ("E1", "E2", "E3"):
        validations[setup] = trained_experiment(setup, setup, splits, processor, config)
    # Select before any E4 test generation, using validation exclusively.
    best = max(validations, key=lambda name: validations[name]["micro"]["f1"])
    write_json(ROOT / "report/e4-selection.json", dict(selected=best, basis="validation micro F1",
               validation_f1={name: value["micro"]["f1"] for name, value in validations.items()}))
    trained_experiment(best, "E4_data_size_0.1", splits, processor, config, fraction=.1)
    baseline_run = json.loads((WORK / f"{best.lower()}/run.json").read_text())
    score_file("E4_data_size_1.0", WORK / f"{best.lower()}.jsonl", meta=baseline_run)
    from student.analyze import analyze
    analyze(ROOT)
    subprocess.run([sys.executable, "scripts/check_submission.py"], check=True)
    print("MANDATORY_EXPERIMENTS_COMPLETE", flush=True)


def compact_artifacts():
    # Always export small diagnostics, even after failure; weights stay on Kaggle.
    target = Path("/kaggle/working/day10-review.zip")
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in (ROOT / "report", WORK, DATA / "splits"):
            if folder.exists():
                for path in folder.rglob("*"):
                    if path.is_file() and path.suffix in {".json", ".jsonl", ".png", ".md", ".csv"}:
                        archive.write(path, path.relative_to(ROOT))
        if (ROOT / "REPORT.md").exists():
            archive.write(ROOT / "REPORT.md", "REPORT.md")
        if (ROOT / "source-manifest.json").exists():
            archive.write(ROOT / "source-manifest.json", "source-manifest.json")


def main():
    config = json.loads((ROOT / "run-config.json").read_text())
    write_json(ROOT / "report/run-config.json", config)
    try:
        splits = prepare()
        if config["mode"] == "preflight":
            preflight(splits, config)
        else:
            full(splits, config)
    except BaseException:
        write_json(ROOT / "report/failure.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        compact_artifacts()


if __name__ == "__main__":
    main()
