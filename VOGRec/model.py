from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig


class DirectedStateLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_linear = nn.Linear(embedding_dim, embedding_dim)
        self.in_linear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_linear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.parent_linear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        states: torch.Tensor,
        edge_index: torch.Tensor,
        out_weight: torch.Tensor,
        in_weight: torch.Tensor,
        parent_ids: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        out_message = torch.zeros_like(states)
        in_message = torch.zeros_like(states)
        if edge_index.numel() > 0:
            source, target = edge_index
            out_message.index_add_(
                0, source, states[target] * out_weight.unsqueeze(-1)
            )
            in_message.index_add_(
                0, target, states[source] * in_weight.unsqueeze(-1)
            )

        update = self.self_linear(states)
        update = update + self.in_linear(in_message)
        update = update + self.out_linear(out_message)
        parent_message = torch.zeros_like(states)
        has_parent = parent_ids >= 0
        if has_parent.any():
            parent_message[has_parent] = states[parent_ids[has_parent]]
        update = update + (
            (1.0 - reliability).unsqueeze(-1)
            * self.parent_linear(parent_message)
        )
        return self.norm(states + self.dropout(F.gelu(update)))


class StateAdapter(nn.Module):
    def __init__(self, embedding_dim: int, adapter_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, embedding_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.layers(values)


class VOGRec(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        state_graph,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.config = config
        embedding_dim = config.embedding_dim

        self.embedding = nn.Embedding(num_users + num_items, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.context_attention = nn.Linear(embedding_dim, 1, bias=False)
        self.context_linear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.order_embedding = nn.Embedding(config.max_order + 1, embedding_dim)
        self.state_layers = nn.ModuleList(
            DirectedStateLayer(embedding_dim, config.dropout)
            for _ in range(config.state_gnn_layers)
        )
        self.state_adapter = StateAdapter(embedding_dim, config.adapter_dim)
        router_input_dim = embedding_dim * 3 + 3
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1),
        )
        self.flow_linear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.user_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.query_norm = nn.LayerNorm(embedding_dim)
        self.target_norm = nn.LayerNorm(embedding_dim)

        self.register_buffer("state_base_items", state_graph.base_items)
        self.register_buffer("state_orders", state_graph.orders)
        self.register_buffer("state_log_supports", torch.log1p(state_graph.supports))
        self.register_buffer("state_reliability", state_graph.reliability)
        self.register_buffer(
            "state_log_out_volume", torch.log1p(state_graph.out_degrees)
        )
        self.register_buffer("state_parent_ids", state_graph.parent_ids)
        self.register_buffer("state_context_items", state_graph.context_items)
        self.register_buffer("state_context_mask", state_graph.context_mask)
        self.register_buffer("state_edge_index", state_graph.edge_index)
        self.register_buffer("state_edge_weight", state_graph.edge_weight)

        incoming_weight = torch.zeros(
            len(state_graph.states), dtype=state_graph.edge_weight.dtype
        )
        if state_graph.edge_index.numel() > 0:
            incoming_weight.index_add_(
                0, state_graph.edge_index[1], state_graph.edge_weight
            )
            in_edge_weight = state_graph.edge_weight / incoming_weight[
                state_graph.edge_index[1]
            ].clamp_min(1e-8)
        else:
            in_edge_weight = state_graph.edge_weight
        self.register_buffer("state_in_edge_weight", in_edge_weight)

    def encode_graphs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_embeddings = self.embedding.weight[: self.num_users]
        item_embeddings = self.embedding.weight[self.num_users :]
        state_embeddings = self._initialize_states(item_embeddings)
        for layer in self.state_layers:
            state_embeddings = layer(
                state_embeddings,
                self.state_edge_index,
                self.state_edge_weight,
                self.state_in_edge_weight,
                self.state_parent_ids,
                self.state_reliability,
            )
        return user_embeddings, item_embeddings, state_embeddings

    def _initialize_states(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        base = item_embeddings[self.state_base_items]
        context = item_embeddings[self.state_context_items]
        attention_logits = self.context_attention(context).squeeze(-1)
        attention_logits = attention_logits.masked_fill(
            ~self.state_context_mask, -1e9
        )
        attention = torch.softmax(attention_logits, dim=-1)
        attention = attention * self.state_context_mask.float()
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        context_vector = (attention.unsqueeze(-1) * context).sum(dim=1)
        return (
            base
            + self.state_reliability.unsqueeze(-1)
            * self.context_linear(context_vector)
            + self.order_embedding(self.state_orders)
        )

    def session_representations(
        self,
        users: torch.Tensor,
        valid_state_ids: torch.Tensor,
        state_mask: torch.Tensor,
        user_embeddings: torch.Tensor,
        state_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_vector = user_embeddings[users]
        selected_states = state_embeddings[valid_state_ids]
        adapted = self.state_adapter(selected_states)
        order_vector = self.order_embedding(self.state_orders[valid_state_ids])
        expanded_user = user_vector.unsqueeze(1).expand_as(adapted)
        router_input = torch.cat(
            [
                expanded_user,
                adapted,
                order_vector,
                self.state_reliability[valid_state_ids].unsqueeze(-1),
                self.state_log_supports[valid_state_ids].unsqueeze(-1),
                self.state_log_out_volume[valid_state_ids].unsqueeze(-1),
            ],
            dim=-1,
        )
        logits = self.router(router_input).squeeze(-1)
        logits = logits.masked_fill(~state_mask, -1e9)
        routing = torch.softmax(logits, dim=-1)
        source_state = (routing.unsqueeze(-1) * adapted).sum(dim=1)
        return user_vector, source_state.unsqueeze(1), routing

    def score_items(
        self,
        user_vectors: torch.Tensor,
        source_states: torch.Tensor,
        routing: torch.Tensor,
        item_ids: torch.Tensor,
        target_state_ids: torch.Tensor,
        item_embeddings: torch.Tensor,
        state_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        del routing, item_ids, item_embeddings
        query = self.query_norm(
            self.user_query(user_vectors) + self.flow_linear(source_states[:, 0])
        )
        target = self.target_norm(
            self.state_adapter(state_embeddings[target_state_ids])
        )
        scale = math.sqrt(query.size(-1))
        if target_state_ids.dim() == 1:
            return (query * target).sum(dim=-1) / scale
        if target_state_ids.dim() == 2:
            return torch.einsum("bd,bcd->bc", query, target) / scale
        raise ValueError("target_state_ids must be one- or two-dimensional")

    def forward(
        self,
        users: torch.Tensor,
        valid_state_ids: torch.Tensor,
        state_mask: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
        positive_target_state_ids: torch.Tensor,
        negative_target_state_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        user_embeddings, item_embeddings, state_embeddings = self.encode_graphs()
        user_vectors, source_states, routing = self.session_representations(
            users,
            valid_state_ids,
            state_mask,
            user_embeddings,
            state_embeddings,
        )
        paired_scores = self.score_items(
            user_vectors,
            source_states,
            routing,
            torch.stack([positive_items, negative_items], dim=1),
            torch.stack(
                [positive_target_state_ids, negative_target_state_ids], dim=1
            ),
            item_embeddings,
            state_embeddings,
        )
        return {
            "positive_scores": paired_scores[:, 0],
            "negative_scores": paired_scores[:, 1],
            "state_embeddings": state_embeddings,
            "routing": routing,
        }

    def backoff_loss(self, state_embeddings: torch.Tensor) -> torch.Tensor:
        has_parent = self.state_parent_ids >= 0
        if not has_parent.any():
            return state_embeddings.new_zeros(())
        difference = (
            state_embeddings[has_parent]
            - state_embeddings[self.state_parent_ids[has_parent]]
        )
        weight = 1.0 - self.state_reliability[has_parent]
        return (weight * difference.pow(2).sum(dim=-1)).mean()
