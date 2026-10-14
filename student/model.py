import hashlib
import json
import math
import re
from pathlib import Path

import bitsandbytes as bnb
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
LLM_TARGET = r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
VIT_TARGET = r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))"


def projector_parameter(name):
    return "visual.merger." in name or "visual.deepstack_merger_list." in name


def load_base():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            # Transformers 5.19 matches a full-name prefix, not an interior path component.
            llm_int8_skip_modules=["model.visual", "lm_head"]),
    )
    assert not any(isinstance(m, bnb.nn.Linear4bit) for m in model.model.visual.modules())
    assert not isinstance(model.lm_head, bnb.nn.Linear4bit)
    assert sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules()) == 252
    return model


def load_processor():
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    processor.tokenizer.padding_side = "left"
    return processor


def architecture(model):
    counts = {"language_model": 0, "vision_backbone": 0, "projector": 0, "other": 0}
    for name, p in model.named_parameters():
        shape = p.quant_state.shape if isinstance(p, bnb.nn.Params4bit) and p.quant_state else p.shape
        size = math.prod(shape)
        block = "projector" if projector_parameter(name) else (
            "vision_backbone" if "visual." in name else
            "language_model" if "language_model." in name else "other")
        counts[block] += size
    counts["total"] = sum(counts.values())
    return counts


def apply_setup(model, setup, rank=16):
    for p in model.parameters():
        p.requires_grad_(False)
    model.enable_input_require_grads()
    pattern = LLM_TARGET if setup != "E3" else f"(?:{LLM_TARGET}|{VIT_TARGET})"
    matches = [name for name, module in model.named_modules() if re.fullmatch(pattern, name)]
    expected = 348 if setup == "E3" else 252
    assert len(matches) == expected, (setup, len(matches), matches[:5])
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                                            target_modules=pattern, task_type="CAUSAL_LM"))
    for name, p in model.named_parameters():
        if setup in {"E2", "E3"} and projector_parameter(name):
            p.data = p.data.float()
            p.requires_grad_(True)
        if p.requires_grad:
            assert "lora_" in name or (setup in {"E2", "E3"} and projector_parameter(name)), name
            # GradScaler unscale requires fp32 trainable gradients.
            p.data = p.data.float()
    trainables = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    info = {"setup": setup, "rank": rank, "matched_modules": matches,
            "trainable_params": sum(trainables.values()),
            "projector_trainable_params": sum(n for name, n in trainables.items() if projector_parameter(name)),
            "trainable_parameter_names": list(trainables)}
    print("SETUP", setup, info["trainable_params"], "modules", len(matches), flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    return model, info


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(model, processor, folder, info):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(folder, safe_serialization=True)
    processor.save_pretrained(folder)
    state = {name: p.detach().cpu().clone() for name, p in model.named_parameters()
             if p.requires_grad and projector_parameter(name)}
    if state:
        torch.save(state, folder / "projector.pt")
    saved = dict(info, projector_sha256=tensor_hash(state) if state else None)
    (folder / "checkpoint-info.json").write_text(json.dumps(saved, indent=2))
    return saved


def load_checkpoint(base, folder):
    folder = Path(folder)
    model = PeftModel.from_pretrained(base, folder, is_trainable=False)
    info = json.loads((folder / "checkpoint-info.json").read_text())
    if (folder / "projector.pt").exists():
        state = torch.load(folder / "projector.pt", map_location="cpu", weights_only=True)
        assert tensor_hash(state) == info["projector_sha256"]
        params = dict(model.named_parameters())
        for name, value in state.items():
            assert name in params and params[name].shape == value.shape, name
            params[name].data = value.to(device=params[name].device, dtype=torch.float32)
        assert tensor_hash({name: params[name] for name in state}) == info["projector_sha256"]
    model.config.use_cache = True
    model.config.text_config.use_cache = True
    model.eval()
    return model
