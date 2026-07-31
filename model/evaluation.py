from __future__ import annotations

from typing import Dict, Sequence

import torch

from .data import EvalExample
from .graph import StateGraph
from .model import VOGRec
from .train_utils import lookup_target_state_ids


def build_state_batch(
    examples: Sequence[EvalExample],
    state_graph: StateGraph,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_lists = [state_graph.valid_states(example.prefix) for example in examples]
    max_states = max(len(values) for values in valid_lists)
    state_ids = torch.zeros(
        (len(examples), max_states), dtype=torch.long, device=device
    )
    state_mask = torch.zeros(
        (len(examples), max_states), dtype=torch.bool, device=device
    )
    for row, values in enumerate(valid_lists):
        state_ids[row, : len(values)] = torch.tensor(
            values, dtype=torch.long, device=device
        )
        state_mask[row, : len(values)] = True
    return state_ids, state_mask


def full_ranking_metrics(
    model: VOGRec,
    examples: Sequence[EvalExample],
    state_graph: StateGraph,
    user_history: Dict[int, set],
    num_items: int,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    if not examples:
        raise ValueError("Full-ranking evaluation requires at least one example")
    model.eval()
    ranks = []
    all_items = torch.arange(num_items, dtype=torch.long, device=device)
    with torch.no_grad():
        user_embeddings, item_embeddings, state_embeddings = model.encode_graphs()
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            users = torch.tensor(
                [example.user for example in chunk],
                dtype=torch.long,
                device=device,
            )
            state_ids, state_mask = build_state_batch(chunk, state_graph, device)
            candidates = all_items.unsqueeze(0).expand(len(chunk), -1)
            target_state_ids = lookup_target_state_ids(
                state_graph,
                [example.prefix for example in chunk],
                candidates,
            )
            user_vectors, source_states, routing = model.session_representations(
                users,
                state_ids,
                state_mask,
                user_embeddings,
                state_embeddings,
            )
            scores = model.score_items(
                user_vectors,
                source_states,
                routing,
                candidates,
                target_state_ids,
                item_embeddings,
                state_embeddings,
            )
            target_items = torch.tensor(
                [example.target for example in chunk],
                dtype=torch.long,
                device=device,
            )
            for row, example in enumerate(chunk):
                forbidden = set(user_history[example.user])
                forbidden.discard(example.target)
                if forbidden:
                    scores[row, torch.tensor(list(forbidden), device=device)] = -torch.inf
            target_scores = scores.gather(1, target_items.unsqueeze(1))
            ranks.append((scores > target_scores).sum(dim=1).cpu() + 1)

    rank_tensor = torch.cat(ranks).float()
    metrics: Dict[str, float] = {}
    for top_k in (5, 10, 20):
        hits = rank_tensor <= top_k
        metrics[f"HR@{top_k}"] = float(hits.float().mean())
        metrics[f"NDCG@{top_k}"] = float(
            torch.where(
                hits,
                1.0 / torch.log2(rank_tensor + 1.0),
                torch.zeros_like(rank_tensor),
            ).mean()
        )
    metrics["MRR"] = float((1.0 / rank_tensor).mean())
    return metrics
