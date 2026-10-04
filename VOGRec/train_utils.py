from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .data import EvalExample
from .graph import StateGraph
from .model import VOGRec


@dataclass
class TensorizedBPRData:
    users: torch.Tensor
    last_items: torch.Tensor
    prefix_contexts: torch.Tensor
    prefix_lengths: torch.Tensor
    positive_items: torch.Tensor
    valid_state_ids: torch.Tensor
    state_mask: torch.Tensor
    positive_target_state_ids: torch.Tensor

    def __len__(self) -> int:
        return int(self.users.numel())

    def to(self, device: torch.device) -> 'TensorizedBPRData':
        return TensorizedBPRData(
            users=self.users.to(device),
            last_items=self.last_items.to(device),
            prefix_contexts=self.prefix_contexts.to(device),
            prefix_lengths=self.prefix_lengths.to(device),
            positive_items=self.positive_items.to(device),
            valid_state_ids=self.valid_state_ids.to(device),
            state_mask=self.state_mask.to(device),
            positive_target_state_ids=self.positive_target_state_ids.to(device),
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tensorized_bpr_data(
    train_sequences: Mapping[int, Sequence[int]],
    state_graph: StateGraph,
) -> TensorizedBPRData:
    users: List[int] = []
    last_items: List[int] = []
    prefix_contexts: List[Tuple[int, ...]] = []
    prefix_lengths: List[int] = []
    positive_items: List[int] = []
    positive_target_state_ids: List[int] = []
    valid_state_ids: List[Tuple[int, ...]] = []
    state_masks: List[Tuple[bool, ...]] = []
    context_width = max(state_graph.max_order - 1, 0)

    for user, sequence in train_sequences.items():
        values = [int(item) for item in sequence]
        for target_index in range(1, len(values)):
            last_item = values[target_index - 1]
            row_state_ids = [0] * state_graph.max_order
            row_state_mask = [False] * state_graph.max_order
            for order in range(1, min(state_graph.max_order, target_index) + 1):
                state = tuple(values[target_index - order : target_index])
                state_id = state_graph.state_to_id.get(state)
                if state_id is not None:
                    row_state_ids[order - 1] = state_id
                    row_state_mask[order - 1] = True

            positive_item = values[target_index]
            positive_target = state_graph.state_to_id[(positive_item,)]
            for order in range(
                min(state_graph.max_order, target_index + 1),
                1,
                -1,
            ):
                target_state = tuple(
                    values[target_index - order + 1 : target_index]
                    + [positive_item]
                )
                target_state_id = state_graph.state_to_id.get(target_state)
                if target_state_id is not None:
                    positive_target = target_state_id
                    break

            recent_context = values[
                max(0, target_index - context_width) : target_index
            ]
            padded_context = [0] * context_width
            if recent_context:
                padded_context[-len(recent_context) :] = recent_context
            users.append(int(user))
            last_items.append(last_item)
            prefix_contexts.append(tuple(padded_context))
            prefix_lengths.append(len(recent_context))
            positive_items.append(positive_item)
            positive_target_state_ids.append(positive_target)
            valid_state_ids.append(tuple(row_state_ids))
            state_masks.append(tuple(row_state_mask))

    users_tensor = torch.tensor(users, dtype=torch.long)
    last_items_tensor = torch.tensor(last_items, dtype=torch.long)
    positive_items_tensor = torch.tensor(positive_items, dtype=torch.long)
    prefix_context_tensor = (
        torch.tensor(prefix_contexts, dtype=torch.long)
        if context_width
        else torch.empty((len(users), 0), dtype=torch.long)
    )
    return TensorizedBPRData(
        users=users_tensor,
        last_items=last_items_tensor,
        prefix_contexts=prefix_context_tensor,
        prefix_lengths=torch.tensor(prefix_lengths, dtype=torch.long),
        positive_items=positive_items_tensor,
        valid_state_ids=torch.tensor(valid_state_ids, dtype=torch.long),
        state_mask=torch.tensor(state_masks, dtype=torch.bool),
        positive_target_state_ids=torch.tensor(
            positive_target_state_ids,
            dtype=torch.long,
        ),
    )


def build_seen_item_mask(
    user_history: Mapping[int, Sequence[int] | set],
    num_users: int,
    num_items: int,
) -> torch.Tensor:
    seen_mask = torch.zeros((num_users, num_items), dtype=torch.bool)
    for user, items in user_history.items():
        item_list = list(items)
        if item_list:
            seen_mask[int(user), torch.tensor(item_list, dtype=torch.long)] = True
    return seen_mask


def sample_unseen_items(
    users: torch.Tensor,
    seen_mask: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    negatives = torch.randint(
        seen_mask.size(1),
        users.shape,
        device=users.device,
        generator=generator,
    )
    invalid = seen_mask[users, negatives]
    while invalid.any():
        negatives[invalid] = torch.randint(
            seen_mask.size(1),
            (int(invalid.sum().item()),),
            device=users.device,
            generator=generator,
        )
        invalid = seen_mask[users, negatives]
    return negatives


def build_collate_fn(
    state_graph: StateGraph,
) -> Callable[[Sequence[Tuple[int, Tuple[int, ...], int]]], Dict[str, object]]:
    def collate(batch: Sequence[Tuple[int, Tuple[int, ...], int]]) -> Dict[str, object]:
        users, prefixes, positives = zip(*batch)
        valid_lists = [state_graph.valid_states(prefix) for prefix in prefixes]
        max_states = max(len(values) for values in valid_lists)
        state_ids = torch.zeros((len(batch), max_states), dtype=torch.long)
        state_mask = torch.zeros((len(batch), max_states), dtype=torch.bool)
        for row, values in enumerate(valid_lists):
            state_ids[row, : len(values)] = torch.tensor(values, dtype=torch.long)
            state_mask[row, : len(values)] = True
        return {
            'users': torch.tensor(users, dtype=torch.long),
            'prefixes': prefixes,
            'positive_items': torch.tensor(positives, dtype=torch.long),
            'valid_state_ids': state_ids,
            'state_mask': state_mask,
        }

    return collate


def bpr_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(positive_scores - negative_scores).mean()


def lookup_target_state_ids(
    state_graph: StateGraph,
    prefixes: Sequence[Sequence[int]],
    item_ids: torch.Tensor,
) -> torch.Tensor:
    if item_ids.dim() not in (1, 2):
        raise ValueError('item_ids must have shape [batch] or [batch, candidates]')
    if item_ids.size(0) != len(prefixes):
        raise ValueError('item_ids batch dimension must match prefixes')
    context_width = max(state_graph.max_order - 1, 0)
    prefix_contexts = torch.zeros(
        (len(prefixes), context_width),
        dtype=torch.long,
        device=item_ids.device,
    )
    prefix_lengths = torch.zeros(
        len(prefixes),
        dtype=torch.long,
        device=item_ids.device,
    )
    for row, prefix in enumerate(prefixes):
        recent = [int(item) for item in prefix[-context_width:]] if context_width else []
        if recent:
            prefix_contexts[row, -len(recent) :] = torch.tensor(
                recent,
                dtype=torch.long,
                device=item_ids.device,
            )
        prefix_lengths[row] = len(recent)
    return lookup_target_state_ids_from_context(
        state_graph,
        prefix_contexts,
        prefix_lengths,
        item_ids,
    )


def lookup_target_state_ids_from_context(
    state_graph: StateGraph,
    prefix_contexts: torch.Tensor,
    prefix_lengths: torch.Tensor,
    item_ids: torch.Tensor,
) -> torch.Tensor:
    if item_ids.dim() not in (1, 2):
        raise ValueError('item_ids must have shape [batch] or [batch, candidates]')
    if item_ids.size(0) != prefix_contexts.size(0):
        raise ValueError('item_ids batch dimension must match prefix contexts')

    device = item_ids.device
    cache = getattr(state_graph, '_kmax_target_cache', None)
    if cache is None:
        cache = {}
        state_graph._kmax_target_cache = cache
    cache_key = str(device)
    cached = cache.get(cache_key)
    if cached is None:
        cached = {
            'base': state_graph.base_state_ids.to(device),
            'orders': {
                order: (
                    state_graph.target_keys_by_order[order].to(device),
                    state_graph.target_state_ids_by_order[order].to(device),
                )
                for order in state_graph.target_keys_by_order
            },
        }
        cache[cache_key] = cached

    num_items = int(state_graph.base_state_ids.numel())
    result = cached['base'][item_ids]
    for order in range(2, state_graph.max_order + 1):
        keys, state_ids = cached['orders'][order]
        if keys.numel() == 0:
            continue
        context_length = order - 1
        rows = torch.nonzero(
            prefix_lengths >= context_length,
            as_tuple=False,
        ).flatten()
        if rows.numel() == 0:
            continue
        contexts = prefix_contexts[rows, -context_length:]
        context_keys = torch.zeros(rows.numel(), dtype=torch.long, device=device)
        for column in range(context_length):
            context_keys = context_keys * num_items + contexts[:, column]
        selected_items = item_ids[rows]
        query_keys = (
            context_keys * num_items + selected_items
            if item_ids.dim() == 1
            else context_keys.unsqueeze(1) * num_items + selected_items
        )
        positions = torch.searchsorted(keys, query_keys)
        safe_positions = positions.clamp_max(keys.numel() - 1)
        matched = (positions < keys.numel()) & (keys[safe_positions] == query_keys)
        resolved = state_ids[safe_positions]
        result[rows] = torch.where(matched, resolved, result[rows])
    return result


def evaluate_leave_one_out(
    model: VOGRec,
    examples: Sequence[EvalExample],
    state_graph: StateGraph,
    user_history: Dict[int, set],
    num_items: int,
    num_negatives: int,
    top_k: int,
    seed: int,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    hits: List[float] = []
    ndcgs: List[float] = []
    prepared = []
    all_items = np.arange(num_items)
    for example in examples:
            forbidden = set(user_history[example.user])
            forbidden.discard(example.target)
            available = np.array([item for item in all_items if item not in forbidden and item != example.target])
            if available.size == 0:
                continue
            sample_size = min(num_negatives, available.size)
            negatives = rng.choice(available, size=sample_size, replace=False).tolist()
            prepared.append((
                example,
                [example.target] + [int(item) for item in negatives],
                state_graph.valid_states(example.prefix),
            ))

    with torch.no_grad():
        user_embeddings, item_embeddings, state_embeddings = model.encode_graphs()
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start : start + batch_size]
            max_states = max(len(valid) for _, _, valid in chunk)
            users = torch.tensor(
                [example.user for example, _, _ in chunk],
                dtype=torch.long,
                device=device,
            )
            state_ids = torch.zeros((len(chunk), max_states), dtype=torch.long, device=device)
            state_mask = torch.zeros((len(chunk), max_states), dtype=torch.bool, device=device)
            for row, (_, _, valid) in enumerate(chunk):
                state_ids[row, : len(valid)] = torch.tensor(valid, dtype=torch.long, device=device)
                state_mask[row, : len(valid)] = True
            candidate_tensor = torch.tensor(
                [candidates for _, candidates, _ in chunk],
                dtype=torch.long,
                device=device,
            )
            candidate_target_state_ids = lookup_target_state_ids(
                state_graph,
                [example.prefix for example, _, _ in chunk],
                candidate_tensor,
            ).to(device)
            user_vectors, short_terms, mixture = model.session_representations(
                users,
                state_ids,
                state_mask,
                user_embeddings,
                state_embeddings,
            )
            scores = model.score_items(
                user_vectors,
                short_terms,
                mixture,
                candidate_tensor,
                candidate_target_state_ids,
                item_embeddings,
                state_embeddings,
            )
            ranks = (scores[:, 1:] > scores[:, :1]).sum(dim=1) + 1
            for rank_value in ranks.tolist():
                rank = int(rank_value)
                hits.append(float(rank <= top_k))
                ndcgs.append(1.0 / math.log2(rank + 1.0) if rank <= top_k else 0.0)

    return {
        f'HR@{top_k}': float(np.mean(hits)) if hits else 0.0,
        f'NDCG@{top_k}': float(np.mean(ndcgs)) if ndcgs else 0.0,
    }
