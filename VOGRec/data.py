from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvalExample:
    user: int
    prefix: Tuple[int, ...]
    target: int


@dataclass
class PreparedData:
    interactions: pd.DataFrame
    train_sequences: Dict[int, List[int]]
    full_sequences: Dict[int, List[int]]
    validation: List[EvalExample]
    test: List[EvalExample]
    train_history: Dict[int, set]
    validation_history: Dict[int, set]
    test_history: Dict[int, set]
    user_history: Dict[int, set]
    user_map: Dict[object, int]
    item_map: Dict[object, int]
    reverse_user_map: Dict[int, object]
    reverse_item_map: Dict[int, object]
    num_users: int
    num_items: int


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix.lower() == ".gz" else path.open("rt", encoding="utf-8")


def _read_json_records(path: Path, max_interactions: int | None) -> pd.DataFrame:
    records = []
    with _open_text(path) as handle:
        for line in handle:
            if max_interactions is not None and len(records) >= max_interactions:
                break
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, list):
                records.extend(value)
            elif isinstance(value, dict):
                records.append(value)
            else:
                raise ValueError(f"Unsupported JSON record in {path}")
    if max_interactions is not None:
        records = records[:max_interactions]
    return pd.DataFrame.from_records(records)


def _resolve_column(
    columns: Iterable[object], requested: str, aliases: Sequence[str]
) -> object:
    available = list(columns)
    for candidate in (requested, *aliases):
        if candidate in available:
            return candidate
    raise ValueError(
        f"Could not resolve column {requested!r}; available columns are {available}"
    )


def load_interactions(
    path: str | Path,
    user_col: str = "user_id",
    item_col: str = "item_id",
    timestamp_col: str = "timestamp",
    delimiter: str | None = None,
    has_header: bool = True,
    column_names: Sequence[str] | None = None,
    rating_col: str | None = None,
    min_rating: float | None = None,
    deduplicate_user_items: bool = True,
    max_interactions: int | None = None,
) -> pd.DataFrame:
    """Load timestamped interactions through one format-agnostic interface.

    JSON/JSONL/JSON.GZ and delimited text files are supported. Column names can
    be supplied explicitly; common public recommendation field names are also
    recognized as aliases.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    lower_name = path.name.lower()
    is_json = lower_name.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz"))
    if is_json:
        frame = _read_json_records(path, max_interactions)
    else:
        if delimiter is None:
            if path.suffix.lower() == ".dat":
                delimiter = "::"
            elif path.suffix.lower() == ".tsv":
                delimiter = "\t"
            else:
                delimiter = ","
        header = 0 if has_header else None
        names = list(column_names) if column_names else None
        frame = pd.read_csv(
            path,
            sep=delimiter,
            engine="python" if len(delimiter) > 1 else "c",
            header=header,
            names=names,
            nrows=max_interactions,
        )

    if frame.empty:
        raise ValueError(f"No interactions were loaded from {path}")
    resolved_user = _resolve_column(
        frame.columns,
        user_col,
        ("user_id", "userId", "reviewerID", "user", "user_raw"),
    )
    resolved_item = _resolve_column(
        frame.columns,
        item_col,
        ("item_id", "itemId", "parent_asin", "asin", "item", "item_raw"),
    )
    resolved_timestamp = _resolve_column(
        frame.columns,
        timestamp_col,
        ("timestamp", "unixReviewTime", "time", "datetime"),
    )
    if min_rating is not None:
        if rating_col is None:
            rating_col = "rating"
        resolved_rating = _resolve_column(frame.columns, rating_col, ("rating",))
        frame = frame.loc[
            pd.to_numeric(frame[resolved_rating], errors="coerce") >= min_rating
        ].copy()

    frame = frame[[resolved_user, resolved_item, resolved_timestamp]].copy()
    frame.columns = ["user_raw", "item_raw", "timestamp"]
    frame = frame.dropna()
    numeric_time = pd.to_numeric(frame["timestamp"], errors="coerce")
    if numeric_time.notna().all():
        frame["timestamp"] = numeric_time
    else:
        parsed_time = pd.to_datetime(frame["timestamp"], errors="raise", utc=True)
        frame["timestamp"] = parsed_time
    frame["_input_order"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values(
        ["user_raw", "timestamp", "_input_order"], kind="stable"
    )
    loaded_count = len(frame)
    if deduplicate_user_items:
        frame = frame.drop_duplicates(["user_raw", "item_raw"], keep="last")
    frame = frame.drop(columns="_input_order").reset_index(drop=True)
    frame.attrs["loaded_interactions"] = loaded_count
    frame.attrs["duplicates_removed"] = loaded_count - len(frame)
    return frame


def iterative_k_core_filter(
    interactions: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> pd.DataFrame:
    current = interactions.copy()
    while True:
        before = len(current)
        user_counts = current.groupby("user_raw")["item_raw"].transform("size")
        current = current.loc[user_counts >= min_user_interactions]
        item_counts = current.groupby("item_raw")["user_raw"].transform("size")
        current = current.loc[item_counts >= min_item_interactions]
        if len(current) == before:
            break
    if current.empty:
        raise ValueError("No interactions remain after iterative k-core filtering")
    return current.copy()


def prepare_data(
    interactions: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    max_user_sequence_length: int | None = None,
) -> PreparedData:
    frame = interactions.copy()
    if max_user_sequence_length is not None:
        if max_user_sequence_length < 3:
            raise ValueError("max_user_sequence_length must be at least three")
        frame = frame.sort_values(["user_raw", "timestamp"], kind="stable")
        frame = frame.groupby("user_raw", sort=False, group_keys=False).tail(
            max_user_sequence_length
        )
    frame = iterative_k_core_filter(
        frame,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
    )
    frame = frame.sort_values(["user_raw", "timestamp"], kind="stable")
    frame = frame.reset_index(drop=True)

    user_values = frame["user_raw"].drop_duplicates().tolist()
    item_values = frame["item_raw"].drop_duplicates().tolist()
    user_map = {value: index for index, value in enumerate(user_values)}
    item_map = {value: index for index, value in enumerate(item_values)}
    reverse_user_map = {index: value for value, index in user_map.items()}
    reverse_item_map = {index: value for value, index in item_map.items()}
    frame["user_id"] = frame["user_raw"].map(user_map).astype(np.int64)
    frame["item_id"] = frame["item_raw"].map(item_map).astype(np.int64)

    train_sequences: Dict[int, List[int]] = {}
    full_sequences: Dict[int, List[int]] = {}
    validation: List[EvalExample] = []
    test: List[EvalExample] = []
    train_history: Dict[int, set] = {}
    validation_history: Dict[int, set] = {}
    test_history: Dict[int, set] = {}
    user_history: Dict[int, set] = {}
    for user, group in frame.groupby("user_id", sort=False):
        sequence = group["item_id"].astype(int).tolist()
        if len(sequence) < 3:
            raise ValueError("Every retained user needs at least three interactions")
        user = int(user)
        train_sequence = sequence[:-2]
        validation_item = sequence[-2]
        test_item = sequence[-1]
        train_sequences[user] = train_sequence
        full_sequences[user] = sequence
        validation.append(EvalExample(user, tuple(train_sequence), validation_item))
        test.append(EvalExample(user, tuple(sequence[:-1]), test_item))
        train_history[user] = set(train_sequence)
        validation_history[user] = set(train_sequence)
        test_history[user] = set(sequence[:-1])
        user_history[user] = set(sequence)

    return PreparedData(
        interactions=frame,
        train_sequences=train_sequences,
        full_sequences=full_sequences,
        validation=validation,
        test=test,
        train_history=train_history,
        validation_history=validation_history,
        test_history=test_history,
        user_history=user_history,
        user_map=user_map,
        item_map=item_map,
        reverse_user_map=reverse_user_map,
        reverse_item_map=reverse_item_map,
        num_users=len(user_map),
        num_items=len(item_map),
    )
