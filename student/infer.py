import json
import time
from pathlib import Path

import torch

from student.data import image_for, user_message


@torch.inference_mode()
def predict(model, processor, rows, data_dir, out, batch_size=4, max_new_tokens=512):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    processor.tokenizer.padding_side = "left"
    model.eval()
    model.config.use_cache = True
    start = time.monotonic()
    outputs = []
    with out.open("x", buffering=1) as stream:
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            messages = [[user_message(image_for(row, data_dir))] for row in batch]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                         return_dict=True, return_tensors="pt", processor_kwargs={"padding": True}).to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                           use_cache=True)
            raw = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            for row, text in zip(batch, raw):
                item = {"image_id": row["image_id"], "raw": text}
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                outputs.append(item)
            if offset % 40 == 0 or offset + len(batch) == len(rows):
                print("PREDICT", out.name, offset + len(batch), "/", len(rows),
                      round(time.monotonic() - start, 1), "seconds", flush=True)
            del inputs, generated
    return outputs
