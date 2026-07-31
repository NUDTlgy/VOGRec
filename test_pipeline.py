from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from vogrec.config import ModelConfig
from vogrec.data import load_interactions, prepare_data
from vogrec.graph import HONBuilder
from vogrec.model import VOGRec
from vogrec.train_utils import (
    bpr_loss,
    build_tensorized_bpr_data,
    lookup_target_state_ids,
)


def _dense_interactions() -> pd.DataFrame:
    rows = []
    timestamp = 0
    for user in range(6):
        for offset in range(6):
            rows.append(
                {
                    "user_raw": f"u{user}",
                    "item_raw": f"i{(user + offset) % 6}",
                    "timestamp": timestamp,
                }
            )
            timestamp += 1
    return pd.DataFrame(rows)


def test_generic_loader_dedup_and_iterative_five_core(tmp_path: Path) -> None:
    frame = _dense_interactions()
    duplicate = frame.iloc[[0]].copy()
    duplicate["timestamp"] = 10_000
    source = pd.concat([frame, duplicate], ignore_index=True)
    path = tmp_path / "events.csv"
    source.to_csv(path, index=False)

    loaded = load_interactions(
        path,
        user_col="user_raw",
        item_col="item_raw",
        timestamp_col="timestamp",
    )
    assert len(loaded) == 36
    assert loaded.attrs["duplicates_removed"] == 1
    data = prepare_data(loaded, 5, 5)
    assert data.num_users == 6
    assert data.num_items == 6
    assert all(len(sequence) == 4 for sequence in data.train_sequences.values())
    assert len(data.validation) == len(data.test) == 6


def test_json_aliases_are_format_generic(tmp_path: Path) -> None:
    path = tmp_path / "reviews.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "reviewerID": f"u{index}",
                    "asin": f"i{index}",
                    "unixReviewTime": index,
                }
            )
            for index in range(3)
        ),
        encoding="utf-8",
    )
    loaded = load_interactions(path)
    assert list(loaded.columns) == ["user_raw", "item_raw", "timestamp"]
    assert len(loaded) == 3


def test_movielens_delimited_file_interface(tmp_path: Path) -> None:
    path = tmp_path / "ratings.dat"
    path.write_text(
        "1::10::5::100\n2::11::4::101\n",
        encoding="utf-8",
    )
    loaded = load_interactions(
        path,
        delimiter="::",
        has_header=False,
        column_names=["user_id", "item_id", "rating", "timestamp"],
    )
    assert loaded["user_raw"].tolist() == [1, 2]
    assert loaded["item_raw"].tolist() == [10, 11]


def _state_graph(max_order: int = 3):
    sequences = {
        0: [0, 1, 2, 3, 4],
        1: [0, 1, 2, 3, 5],
        2: [6, 1, 2, 3, 4],
        3: [6, 1, 2, 3, 5],
    }
    return HONBuilder(max_order=max_order, min_support=2).build(sequences, 7)


def test_variable_order_graph_and_target_resolution() -> None:
    graph = _state_graph()
    assert (1, 2) in graph.state_to_id
    assert (1, 2, 3) in graph.state_to_id
    child = graph.state_to_id[(1, 2, 3)]
    assert graph.parent_ids[child].item() == graph.state_to_id[(2, 3)]
    assert 0.0 < graph.reliability[child].item() < 1.0
    candidates = torch.tensor([[3, 4, 5]])
    resolved = lookup_target_state_ids(graph, [[0, 1, 2]], candidates)
    assert resolved[0, 0].item() == graph.state_to_id[(1, 2, 3)]
    assert resolved[0, 1].item() == graph.state_to_id[(4,)]

    source, weights = graph.edge_index[0], graph.edge_weight
    for source_id in source.unique():
        assert weights[source == source_id].sum().item() == pytest.approx(1.0)


def test_model_forward_backward_without_item_bias() -> None:
    graph = _state_graph(max_order=2)
    model = VOGRec(
        num_users=4,
        num_items=7,
        state_graph=graph,
        config=ModelConfig(
            embedding_dim=8,
            state_gnn_layers=1,
            max_order=2,
            adapter_dim=4,
            dropout=0.0,
        ),
    )
    assert not hasattr(model, "item_bias")
    tensorized = build_tensorized_bpr_data(
        {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}, graph
    )
    indices = torch.tensor([1, 4])
    negative_items = torch.tensor([4, 5])
    negative_targets = lookup_target_state_ids(
        graph, [[0, 1], [0, 1]], negative_items
    )
    outputs = model(
        tensorized.users[indices],
        tensorized.valid_state_ids[indices],
        tensorized.state_mask[indices],
        tensorized.positive_items[indices],
        negative_items,
        tensorized.positive_target_state_ids[indices],
        negative_targets,
    )
    loss = bpr_loss(outputs["positive_scores"], outputs["negative_scores"])
    loss = loss + 0.01 * model.backoff_loss(outputs["state_embeddings"])
    loss.backward()
    assert torch.isfinite(loss)


def test_kmax_three_target_lookup_and_forward() -> None:
    graph = _state_graph(max_order=3)
    model = VOGRec(
        num_users=4,
        num_items=7,
        state_graph=graph,
        config=ModelConfig(
            embedding_dim=8,
            state_gnn_layers=1,
            max_order=3,
            adapter_dim=4,
            dropout=0.0,
        ),
    )
    tensorized = build_tensorized_bpr_data(
        {0: [0, 1, 2, 3], 1: [6, 1, 2, 3]}, graph
    )
    negative_items = torch.tensor([4, 5, 4, 5, 4, 5])
    negative_targets = lookup_target_state_ids(
        graph,
        [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
        negative_items,
    )
    outputs = model(
        tensorized.users,
        tensorized.valid_state_ids,
        tensorized.state_mask,
        tensorized.positive_items,
        negative_items,
        tensorized.positive_target_state_ids,
        negative_targets,
    )
    assert outputs["positive_scores"].shape == outputs["negative_scores"].shape


def test_one_epoch_training_smoke(tmp_path: Path) -> None:
    data_path = tmp_path / "events.csv"
    _dense_interactions().to_csv(data_path, index=False)
    output_dir = tmp_path / "run"
    repository = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(repository / "train.py"),
        "--data",
        str(data_path),
        "--user-col",
        "user_raw",
        "--item-col",
        "item_raw",
        "--timestamp-col",
        "timestamp",
        "--output-dir",
        str(output_dir),
        "--epochs",
        "1",
        "--batch-size",
        "8",
        "--embedding-dim",
        "8",
        "--adapter-dim",
        "4",
        "--state-gnn-layers",
        "1",
        "--min-state-support",
        "1",
        "--validation-every",
        "1",
        "--full-ranking-batch-size",
        "3",
        "--device",
        "cpu",
    ]
    completed = subprocess.run(
        command,
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    results = json.loads((output_dir / "final_results.json").read_text())
    assert results["selection"] == "validation_HR@10"
    assert (output_dir / "best.pt").exists()
