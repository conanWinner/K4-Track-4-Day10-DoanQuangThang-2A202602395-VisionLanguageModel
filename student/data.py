import json
import random
from pathlib import Path

import torch
from PIL import Image

PROMPT = ('Detect all visible road damage in this image. Use only the labels "pothole", '
          '"crack", and "manhole". Return only a JSON list, with no explanation or markdown. '
          'Each object must have "label" and "bbox_2d": [x1,y1,x2,y2]. '
          'Coordinates must be integers normalized to 0-1000 relative to image width and height. '
          'Order objects from top to bottom, then left to right. Return [] if none are visible.')


def objects_to_answer(objects):
    result = []
    for obj in objects:
        xc, yc, w, h = obj["yolo"]
        if w <= 0 or h <= 0:
            # Source annotation noise: keep original GT for evaluation, omit impossible SFT targets.
            continue
        box = [max(0, min(1000, round(v * 1000))) for v in
               (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)]
        if box[2] <= box[0]:
            box[0] = min(box[0], 999)
            box[2] = box[0] + 1
        if box[3] <= box[1]:
            box[1] = min(box[1], 999)
            box[3] = box[1] + 1
        result.append({"label": obj["label"], "bbox_2d": box})
    result.sort(key=lambda x: (x["bbox_2d"][1], x["bbox_2d"][0]))
    return json.dumps(result, separators=(",", ":"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def image_for(row, data_dir):
    with Image.open(Path(data_dir) / row["image_path"]) as image:
        return image.convert("RGB")


def user_message(image):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": PROMPT}]}


def encode_example(row, data_dir, processor):
    message = user_message(image_for(row, data_dir))
    answer = objects_to_answer(row["objects"])
    prompt = processor.apply_chat_template([message], tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt")
    full = processor.apply_chat_template([message, {"role": "assistant", "content": answer}],
                                         tokenize=True, add_generation_prompt=False,
                                         return_dict=True, return_tensors="pt")
    length = prompt["input_ids"].shape[1]
    assert torch.equal(full["input_ids"][:, :length], prompt["input_ids"]), "Prompt not prefix"
    labels = full["input_ids"].clone()
    labels[:, :length] = -100
    full["labels"] = labels
    assert (labels != -100).any()
    assert "mm_token_type_ids" in full, "M-RoPE needs multimodal token types"
    return dict(full)


def collate(examples, pad_id):
    result = {}
    length = max(x["input_ids"].shape[1] for x in examples)
    sequence_keys = {"input_ids", "attention_mask", "mm_token_type_ids", "labels"}
    for key in examples[0]:
        if key in sequence_keys:
            fill = -100 if key == "labels" else pad_id if key == "input_ids" else 0
            result[key] = torch.cat([torch.nn.functional.pad(x[key], (0, length - x[key].shape[1]),
                                                            value=fill) for x in examples])
        elif key in {"pixel_values", "image_grid_thw"}:
            result[key] = torch.cat([x[key] for x in examples], dim=0)
        else:
            raise ValueError(f"Unhandled processor key {key}")
    return result


class EncodedData:
    def __init__(self, rows, data_dir, processor):
        self.rows, self.data_dir, self.processor = rows, data_dir, processor
        self.cache = {}

    def get(self, index):
        if index not in self.cache:
            self.cache[index] = encode_example(self.rows[index], self.data_dir, self.processor)
        return self.cache[index]


def subset(rows, fraction, seed=42):
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[i] for i in indices[:max(1, round(len(rows) * fraction))]]
