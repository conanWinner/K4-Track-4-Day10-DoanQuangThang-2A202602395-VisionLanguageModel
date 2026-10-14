import json
import math
import random
import time
from pathlib import Path

import torch

from student.data import EncodedData, collate
from student.model import save_checkpoint


def train(model, processor, rows, data_dir, folder, info, steps=300, batch_size=1, accum=8,
          lr=2e-4, seed=42):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    data = EncodedData(rows, data_dir, processor)
    (folder / "train-ids.json").write_text(json.dumps([r["image_id"] for r in rows]))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0, foreach=False)
    warmup = max(1, math.ceil(steps * .05))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    rng = random.Random(seed)
    order, position = [], 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step, attempts = 0, 0
    with (folder / "loss.jsonl").open("x", buffering=1) as log:
        while step < steps:
            attempts += 1
            if attempts > steps + 30:
                raise RuntimeError("Too many skipped optimizer steps from non-finite gradients")
            loss_sum = 0.0
            for micro in range(accum):
                indices = []
                for _ in range(batch_size):
                    if position >= len(order):
                        order = list(range(len(rows)))
                        rng.shuffle(order)
                        position = 0
                    indices.append(order[position])
                    position += 1
                batch = {k: v.to("cuda") for k, v in collate([data.get(i) for i in indices],
                                         processor.tokenizer.pad_token_id).items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}, microbatch {micro}")
                loss_sum += loss.detach().item() / accum
                scaler.scale(loss / accum).backward()
                del batch, loss
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < previous_scale
            optimizer.zero_grad(set_to_none=True)
            if skipped:
                print("SKIPPED scaled step", step, "scale", previous_scale, flush=True)
                continue
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite unscaled gradient")
            step += 1
            row = {"step": step, "loss": loss_sum, "grad_norm": norm.item(),
                   "lr": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - started}
            log.write(json.dumps(row) + "\n")
            scheduler.step()
            if step <= 3 or step % 10 == 0:
                print("TRAIN", info["setup"], row, flush=True)
            if step % 100 == 0:
                checkpoint = folder / f"step-{step}"
                save_checkpoint(model, processor, checkpoint, dict(info, optimizer_steps=step))
                # Keep the full resume state outside exported output to avoid multi-GB artifacts.
                resume = Path("/kaggle/temp/day10-resume") / folder.name / f"step-{step}"
                resume.mkdir(parents=True, exist_ok=False)
                torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict(), "rng": rng.getstate(), "order": order,
                            "position": position, "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all()}, resume / "training-state.pt")
    torch.cuda.synchronize()
    run = dict(info, optimizer_steps=step, attempts=attempts, seed=seed, batch_size=batch_size,
               grad_accum=accum, learning_rate=lr, train_images=len(rows),
               peak_vram_gb=torch.cuda.max_memory_allocated() / 1024 ** 3,
               train_minutes=(time.monotonic() - started) / 60)
    save_checkpoint(model, processor, folder / "final", run)
    (folder / "run.json").write_text(json.dumps(run, indent=2))
    print("TRAIN_DONE", {key: run[key] for key in ("setup", "optimizer_steps", "trainable_params",
                         "peak_vram_gb", "train_minutes", "train_images")}, flush=True)
    return run
