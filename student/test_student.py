import json

import pytest
import torch

from student.data import collate, objects_to_answer, subset
from student import analyze  # import validates report-generation source on the Kaggle runtime


def test_target_conversion_order_and_clipping():
    answer = objects_to_answer([{"label": "crack", "yolo": [.2, .8, .1, .1]},
                                {"label": "pothole", "yolo": [.5, .5, .2, .2]}])
    objects = json.loads(answer)
    assert objects[0] == {"label": "pothole", "bbox_2d": [400, 400, 600, 600]}
    assert objects_to_answer([]) == "[]"
    assert json.loads(objects_to_answer([{"label": "pothole", "yolo": [0, 0, .2, .2]}]))[0]["bbox_2d"] == [0, 0, 100, 100]


def test_collator_preserves_visual_rows_and_masks_padding():
    a = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]]),
         "mm_token_type_ids": torch.tensor([[0, 1]]), "labels": torch.tensor([[-100, 2]]),
         "pixel_values": torch.ones(3, 5), "image_grid_thw": torch.tensor([[1, 2, 6]])}
    b = {**a, "input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]]),
         "mm_token_type_ids": torch.tensor([[0, 1, 0]]), "labels": torch.tensor([[-100, 2, 3]])}
    result = collate([a, b], 0)
    assert result["labels"].tolist() == [[-100, 2, -100], [-100, 2, 3]]
    assert result["pixel_values"].shape == (6, 5)
    assert result["image_grid_thw"].shape == (2, 3)
    assert result["attention_mask"].tolist() == [[1, 1, 0], [1, 1, 1]]


def test_subset_is_repeatable_and_does_not_mutate():
    rows = list(range(1422))
    small = subset(rows, .1)
    assert len(small) == 142 and len(set(small)) == 142
    assert small == subset(rows, .1)
    assert rows == list(range(1422))


def test_impossible_source_box_is_not_a_training_target():
    assert objects_to_answer([{"label": "crack", "yolo": [.5, .5, .1, 0]}]) == "[]"
    assert json.loads(objects_to_answer([{"label": "crack", "yolo": [.5, .5, .0001, .0001]}]))[0]["bbox_2d"] == [500, 500, 501, 501]
