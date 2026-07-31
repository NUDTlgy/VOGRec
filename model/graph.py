from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import exp, log
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import json
import torch


StateKey = Tuple[int, ...]


@dataclass
class StateGraph:
    states: List[StateKey]
    state_to_id: Dict[StateKey, int]
    base_items: torch.Tensor
    orders: torch.Tensor
    supports: torch.Tensor
    reliability: torch.Tensor
    out_degrees: torch.Tensor
    parent_ids: torch.Tensor
    context_items: torch.Tensor
    context_mask: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    target_lookup: torch.Tensor
    target_pair_keys: torch.Tensor
    target_pair_state_ids: torch.Tensor
    target_keys_by_order: Dict[int, torch.Tensor]
    target_state_ids_by_order: Dict[int, torch.Tensor]
    base_state_ids: torch.Tensor
    max_order: int

    def valid_states(self, prefix: Sequence[int]) -> List[int]:
        valid = []
        max_length = min(len(prefix), self.max_order)
        for length in range(1, max_length + 1):
            key = tuple(int(item) for item in prefix[-length:])
            state_id = self.state_to_id.get(key)
            if state_id is not None:
                valid.append(state_id)
        if not valid:
            raise KeyError(f'No first-order state for item {prefix[-1]}')
        return valid

    def longest_state(self, prefix: Sequence[int]) -> int:
        return self.valid_states(prefix)[-1]

    def target_state(self, prefix: Sequence[int], candidate_item: int) -> int:
        extended = tuple(int(item) for item in prefix) + (int(candidate_item),)
        return self.longest_state(extended)

    def lookup_target_state_ids(
        self,
        last_items: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.max_order != 2:
            raise ValueError('Sparse target lookup currently supports order two only')
        if last_items.shape != item_ids.shape:
            last_items, item_ids = torch.broadcast_tensors(last_items, item_ids)
        device = item_ids.device
        cache = getattr(self, '_lookup_cache', None)
        if cache is None:
            cache = {}
            self._lookup_cache = cache
        cache_key = str(device)
        cached = cache.get(cache_key)
        if cached is None:
            cached = (
                self.base_state_ids.to(device),
                self.target_pair_keys.to(device),
                self.target_pair_state_ids.to(device),
            )
            cache[cache_key] = cached
        base_state_ids, pair_keys, pair_state_ids = cached
        base_state_ids = base_state_ids[item_ids]
        if self.target_pair_keys.numel() == 0:
            return base_state_ids
        num_items = self.base_state_ids.numel()
        query_keys = last_items * num_items + item_ids
        positions = torch.searchsorted(pair_keys, query_keys)
        valid = positions < pair_keys.numel()
        safe_positions = positions.clamp_max(pair_keys.numel() - 1)
        valid = valid & (pair_keys[safe_positions] == query_keys)
        return torch.where(valid, pair_state_ids[safe_positions], base_state_ids)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        edges = []
        for index in range(self.edge_index.size(1)):
            edges.append(
                {
                    'source': int(self.edge_index[0, index]),
                    'target': int(self.edge_index[1, index]),
                    'weight': float(self.edge_weight[index]),
                }
            )
        payload = {
            'max_order': self.max_order,
            'states': [
                {
                    'id': state_id,
                    'items': list(state),
                    'name': format_state(state),
                    'base_item': int(self.base_items[state_id]),
                    'order': int(self.orders[state_id]),
                    'support': float(self.supports[state_id]),
                    'reliability': float(self.reliability[state_id]),
                    'out_degree': float(self.out_degrees[state_id]),
                    'parent_id': int(self.parent_ids[state_id]),
                }
                for state_id, state in enumerate(self.states)
            ],
            'edges': edges,
        }
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def format_state(state: StateKey) -> str:
    base = str(state[-1])
    if len(state) == 1:
        return base
    context = '.'.join(str(item) for item in reversed(state[:-1]))
    return f'{base}|{context}'


class HONBuilder:
    def __init__(
        self,
        max_order: int = 2,
        min_support: int = 5,
        alpha: float = 0.5,
        evidence_temperature: float = 1.0,
    ) -> None:
        if max_order < 1:
            raise ValueError('max_order must be positive')
        self.max_order = max_order
        self.min_support = min_support
        self.alpha = alpha
        self.evidence_temperature = evidence_temperature

    def build(
        self,
        train_sequences: Mapping[int, Sequence[int]],
        num_items: int,
    ) -> StateGraph:
        next_counts = self._count_successors(train_sequences)
        retained, supports, reliability = self._select_states(next_counts, num_items)
        states = sorted(retained, key=lambda state: (len(state), state[-1], state))
        state_to_id = {state: index for index, state in enumerate(states)}

        parent_ids = []
        for state in states:
            if len(state) == 1:
                parent_ids.append(-1)
                continue
            parent = state[1:]
            while parent not in state_to_id and len(parent) > 1:
                parent = parent[1:]
            parent_ids.append(state_to_id[parent])

        edge_counts: Counter[Tuple[int, int]] = Counter()
        for sequence in train_sequences.values():
            state_path = [
                self._longest_state(sequence[: index + 1], state_to_id)
                for index in range(len(sequence))
            ]
            edge_counts.update(zip(state_path[:-1], state_path[1:]))

        source_totals: Counter[int] = Counter()
        for (source, _), count in edge_counts.items():
            source_totals[source] += count

        sorted_edges = sorted(edge_counts)
        if sorted_edges:
            edge_index = torch.tensor(sorted_edges, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(
                [edge_counts[edge] / source_totals[edge[0]] for edge in sorted_edges],
                dtype=torch.float32,
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float32)

        out_degrees = torch.zeros(len(states), dtype=torch.float32)
        for (source, _), count in edge_counts.items():
            out_degrees[source] += count

        context_width = max(self.max_order - 1, 1)
        context_items = torch.zeros((len(states), context_width), dtype=torch.long)
        context_mask = torch.zeros((len(states), context_width), dtype=torch.bool)
        for state_id, state in enumerate(states):
            context = state[:-1]
            if context:
                context_items[state_id, : len(context)] = torch.tensor(context, dtype=torch.long)
                context_mask[state_id, : len(context)] = True

        target_lookup = torch.empty((0, 0), dtype=torch.long)
        base_state_ids = torch.tensor(
            [state_to_id[(item,)] for item in range(num_items)],
            dtype=torch.long,
        )
        target_keys_by_order: Dict[int, torch.Tensor] = {}
        target_state_ids_by_order: Dict[int, torch.Tensor] = {}
        for order in range(2, self.max_order + 1):
            encoded_states = []
            for state_id, state in enumerate(states):
                if len(state) != order:
                    continue
                key = 0
                for item in state:
                    key = key * num_items + int(item)
                encoded_states.append((key, state_id))
            encoded_states.sort()
            target_keys_by_order[order] = torch.tensor(
                [entry[0] for entry in encoded_states],
                dtype=torch.long,
            )
            target_state_ids_by_order[order] = torch.tensor(
                [entry[1] for entry in encoded_states],
                dtype=torch.long,
            )

        target_pairs = list(zip(
            target_keys_by_order.get(2, torch.empty(0, dtype=torch.long)).tolist(),
            target_state_ids_by_order.get(2, torch.empty(0, dtype=torch.long)).tolist(),
        ))
        if target_pairs:
            target_pair_keys = torch.tensor(
                [pair[0] for pair in target_pairs], dtype=torch.long
            )
            target_pair_state_ids = torch.tensor(
                [pair[1] for pair in target_pairs], dtype=torch.long
            )
        else:
            target_pair_keys = torch.empty((0,), dtype=torch.long)
            target_pair_state_ids = torch.empty((0,), dtype=torch.long)

        return StateGraph(
            states=states,
            state_to_id=state_to_id,
            base_items=torch.tensor([state[-1] for state in states], dtype=torch.long),
            orders=torch.tensor([len(state) for state in states], dtype=torch.long),
            supports=torch.tensor([supports.get(state, 0) for state in states], dtype=torch.float32),
            reliability=torch.tensor([reliability.get(state, 0.0) for state in states], dtype=torch.float32),
            out_degrees=out_degrees,
            parent_ids=torch.tensor(parent_ids, dtype=torch.long),
            context_items=context_items,
            context_mask=context_mask,
            edge_index=edge_index,
            edge_weight=edge_weight,
            target_lookup=target_lookup,
            target_pair_keys=target_pair_keys,
            target_pair_state_ids=target_pair_state_ids,
            target_keys_by_order=target_keys_by_order,
            target_state_ids_by_order=target_state_ids_by_order,
            base_state_ids=base_state_ids,
            max_order=self.max_order,
        )

    def _count_successors(
        self,
        train_sequences: Mapping[int, Sequence[int]],
    ) -> Dict[StateKey, Counter[int]]:
        counts: Dict[StateKey, Counter[int]] = defaultdict(Counter)
        for sequence in train_sequences.values():
            sequence = [int(item) for item in sequence]
            for target_index in range(1, len(sequence)):
                target = sequence[target_index]
                max_length = min(self.max_order, target_index)
                for length in range(1, max_length + 1):
                    context = tuple(sequence[target_index - length : target_index])
                    counts[context][target] += 1
        return counts

    def _select_states(
        self,
        next_counts: Mapping[StateKey, Counter[int]],
        num_items: int,
    ) -> Tuple[set, Dict[StateKey, int], Dict[StateKey, float]]:
        retained = {(item,) for item in range(num_items)}
        supports: Dict[StateKey, int] = {(item,): 0 for item in range(num_items)}
        reliability: Dict[StateKey, float] = {(item,): 0.0 for item in range(num_items)}

        for state, counts in next_counts.items():
            if len(state) == 1:
                supports[state] = sum(counts.values())

        candidates = sorted(
            (state for state in next_counts if len(state) > 1),
            key=lambda state: (len(state), state),
        )
        for state in candidates:
            support = sum(next_counts[state].values())
            parent = state[1:]
            parent_counts = next_counts.get(parent, Counter())
            divergence, degrees = self._kl_divergence(next_counts[state], parent_counts)
            evidence = support * divergence - 0.5 * degrees * log(1.0 + support)
            scaled = max(min(evidence / self.evidence_temperature, 60.0), -60.0)
            reliability_value = 1.0 / (1.0 + exp(-scaled))
            supports[state] = support
            reliability[state] = reliability_value
            if support >= self.min_support:
                retained.add(state)
        return retained, supports, reliability

    def _kl_divergence(
        self,
        child: Counter[int],
        parent: Counter[int],
    ) -> Tuple[float, int]:
        outcomes = sorted(set(child) | set(parent))
        if not outcomes:
            return 0.0, 1
        child_total = sum(child.values()) + self.alpha * len(outcomes)
        parent_total = sum(parent.values()) + self.alpha * len(outcomes)
        divergence = 0.0
        for item in outcomes:
            child_probability = (child[item] + self.alpha) / child_total
            parent_probability = (parent[item] + self.alpha) / parent_total
            divergence += child_probability * log(child_probability / parent_probability)
        return divergence, max(len(child) - 1, 1)

    def _longest_state(
        self,
        prefix: Sequence[int],
        state_to_id: Mapping[StateKey, int],
    ) -> int:
        max_length = min(len(prefix), self.max_order)
        for length in range(max_length, 0, -1):
            key = tuple(int(item) for item in prefix[-length:])
            if key in state_to_id:
                return state_to_id[key]
        raise KeyError(prefix[-1])
