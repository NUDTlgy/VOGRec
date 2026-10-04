from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from .config import ModelConfig, TrainConfig
from .data import load_interactions, prepare_data
from .evaluation import full_ranking_metrics
from .graph import HONBuilder
from .model import VOGRec
from .train_utils import (
    bpr_loss,
    build_seen_item_mask,
    build_tensorized_bpr_data,
    lookup_target_state_ids_from_context,
    sample_unseen_items,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VOGRec with the paper's full-ranking protocol"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="runs/vogrec")
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--user-col", default="user_id")
    parser.add_argument("--item-col", default="item_id")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--delimiter")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--column-names", nargs="+")
    parser.add_argument("--rating-col")
    parser.add_argument("--min-rating", type=float)
    parser.add_argument("--max-interactions", type=int)
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--max-user-sequence-length", type=int)
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--min-item-interactions", type=int, default=5)

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--adapter-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--state-gnn-layers", type=int, default=2)
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--min-state-support", type=int, default=2)
    parser.add_argument("--smoothing-alpha", type=float, default=0.5)
    parser.add_argument("--evidence-temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--backoff-weight", type=float, default=0.01)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--validation-switch-epoch", type=int, default=400)
    parser.add_argument("--validation-after-every", type=int, default=25)
    parser.add_argument("--full-ranking-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validation_every < 1:
        raise ValueError("validation-every must be positive")
    if args.validation_switch_epoch < 1:
        raise ValueError("validation-switch-epoch must be positive")
    if args.validation_after_every < 1:
        raise ValueError("validation-after-every must be positive")
    if args.max_order < 1:
        raise ValueError("max-order must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = ModelConfig(
        embedding_dim=args.embedding_dim,
        state_gnn_layers=args.state_gnn_layers,
        max_order=args.max_order,
        adapter_dim=args.adapter_dim,
        dropout=args.dropout,
    )
    train_config = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        min_state_support=args.min_state_support,
        smoothing_alpha=args.smoothing_alpha,
        evidence_temperature=args.evidence_temperature,
        backoff_weight=args.backoff_weight,
        validation_every=args.validation_every,
        validation_switch_epoch=args.validation_switch_epoch,
        validation_after_every=args.validation_after_every,
        full_ranking_batch_size=args.full_ranking_batch_size,
    )

    interactions = load_interactions(
        args.data,
        user_col=args.user_col,
        item_col=args.item_col,
        timestamp_col=args.timestamp_col,
        delimiter=args.delimiter,
        has_header=not args.no_header,
        column_names=args.column_names,
        rating_col=args.rating_col,
        min_rating=args.min_rating,
        deduplicate_user_items=not args.keep_duplicates,
        max_interactions=args.max_interactions,
    )
    loaded_interactions = interactions.attrs.get("loaded_interactions", len(interactions))
    duplicates_removed = interactions.attrs.get("duplicates_removed", 0)
    data = prepare_data(
        interactions,
        min_user_interactions=train_config.min_user_interactions,
        min_item_interactions=train_config.min_item_interactions,
        max_user_sequence_length=args.max_user_sequence_length,
    )
    state_graph = HONBuilder(
        max_order=model_config.max_order,
        min_support=train_config.min_state_support,
        alpha=train_config.smoothing_alpha,
        evidence_temperature=train_config.evidence_temperature,
    ).build(data.train_sequences, data.num_items)
    state_graph.to_json(output_dir / "state_graph.json")

    tensorized = build_tensorized_bpr_data(
        data.train_sequences, state_graph
    ).to(device)
    if len(tensorized) == 0:
        raise RuntimeError("No training examples remain after preprocessing")
    seen_mask = build_seen_item_mask(
        data.train_history, data.num_users, data.num_items
    ).to(device)
    train_generator = torch.Generator(device=device.type)
    train_generator.manual_seed(train_config.seed)

    model = VOGRec(
        data.num_users,
        data.num_items,
        state_graph,
        model_config,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    run_config = {
        "dataset": args.dataset_name,
        "split": "chronological_leave_two_out",
        "evaluation": "full_ranking",
        "checkpoint_selection": "validation_HR@10",
        "test_evaluations_during_training": 0,
        "deduplicate_user_items": not args.keep_duplicates,
        "loaded_interactions": int(loaded_interactions),
        "duplicates_removed": int(duplicates_removed),
        "interactions_after_iterative_5_core": len(data.interactions),
        "num_users": data.num_users,
        "num_items": data.num_items,
        "num_states": len(state_graph.states),
        "num_high_order_states": sum(
            len(state) > 1 for state in state_graph.states
        ),
        "num_state_edges": int(state_graph.edge_index.size(1)),
        "num_training_examples": len(tensorized),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_config, indent=2), flush=True)
    if args.dry_run:
        return

    best_validation_hr = -1.0
    training_curve = {}
    checkpoint_path = output_dir / "best.pt"
    for epoch in range(1, train_config.epochs + 1):
        started = time.perf_counter()
        model.train()
        permutation = torch.randperm(
            len(tensorized), device=device, generator=train_generator
        )
        running_loss = 0.0
        examples_seen = 0
        for start in range(0, len(tensorized), train_config.batch_size):
            indices = permutation[start : start + train_config.batch_size]
            users = tensorized.users[indices]
            positive_items = tensorized.positive_items[indices]
            negative_items = sample_unseen_items(
                users, seen_mask, train_generator
            )
            negative_targets = lookup_target_state_ids_from_context(
                state_graph,
                tensorized.prefix_contexts[indices],
                tensorized.prefix_lengths[indices],
                negative_items,
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                users,
                tensorized.valid_state_ids[indices],
                tensorized.state_mask[indices],
                positive_items,
                negative_items,
                tensorized.positive_target_state_ids[indices],
                negative_targets,
            )
            loss = bpr_loss(
                outputs["positive_scores"], outputs["negative_scores"]
            )
            loss = loss + train_config.backoff_weight * model.backoff_loss(
                outputs["state_embeddings"]
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_size = users.size(0)
            running_loss += float(loss.detach()) * batch_size
            examples_seen += batch_size

        mean_loss = running_loss / examples_seen
        elapsed = time.perf_counter() - started
        message = f"Epoch {epoch:03d} loss={mean_loss:.6f} time={elapsed:.2f}s"
        validation_interval = train_config.validation_every
        if epoch > train_config.validation_switch_epoch:
            validation_interval = train_config.validation_after_every
        validation_due = (
            epoch % validation_interval == 0
            or epoch == train_config.epochs
        )
        if validation_due:
            validation_metrics = full_ranking_metrics(
                model,
                data.validation,
                state_graph,
                data.validation_history,
                data.num_items,
                device,
                train_config.full_ranking_batch_size,
            )
            training_curve[str(epoch)] = {
                "loss": mean_loss,
                "seconds": elapsed,
                "validation": validation_metrics,
            }
            (output_dir / "training_curve.json").write_text(
                json.dumps(training_curve, indent=2), encoding="utf-8"
            )
            message += (
                f" val_HR@10={validation_metrics['HR@10']:.6f}"
                f" val_NDCG@10={validation_metrics['NDCG@10']:.6f}"
            )
            if validation_metrics["HR@10"] > best_validation_hr:
                best_validation_hr = validation_metrics["HR@10"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": asdict(model_config),
                        "train_config": asdict(train_config),
                        "best_epoch": epoch,
                        "best_validation": validation_metrics,
                    },
                    checkpoint_path,
                )
        print(message, flush=True)

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = full_ranking_metrics(
        model,
        data.test,
        state_graph,
        data.test_history,
        data.num_items,
        device,
        train_config.full_ranking_batch_size,
    )
    final_results = {
        "selection": "validation_HR@10",
        "best_epoch": checkpoint["best_epoch"],
        "validation": checkpoint["best_validation"],
        "test": test_metrics,
    }
    (output_dir / "final_results.json").write_text(
        json.dumps(final_results, indent=2), encoding="utf-8"
    )
    print("FINAL_RESULTS " + json.dumps(final_results), flush=True)


if __name__ == "__main__":
    main()

