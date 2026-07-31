# VOGRec

This repository contains the anonymous reference implementation of **VOGRec:
Variable-Order Item-State Graph Learning for Recommendation**.

The release implements the model described in the paper: support-constrained
variable-order state discovery, exclusive directed state flows, reliability-
guided suffix backoff, directed StateGNN encoding, personalized source routing,
candidate-conditioned target resolution, and a single BPR-trained ranking score.

## Requirements

```bash
python -m pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

## Data interface

All datasets use the same loader. JSON, JSONL, compressed JSON/JSONL, CSV, TSV,
and other delimited text files are accepted. The loader needs a user ID, item
ID, and timestamp column. Common aliases such as `reviewerID`, `asin`,
`parent_asin`, `unixReviewTime`, `userId`, and `itemId` are recognized.

The preprocessing protocol is fixed to the paper:

1. sort each user's interactions chronologically;
2. remove repeated user-item interactions, keeping the latest event;
3. iteratively filter users and items until both have at least five events;
4. use the penultimate event for validation and the final event for testing;
5. construct all state statistics and graphs from training sequences only.

Deduplication is enabled by default and is harmless for an already deduplicated
dataset such as the Beauty data used in the paper. `--keep-duplicates` is
available only for controlled diagnostics.

## Training

MovieLens-style `::` data without a header:

```bash
python train.py \
  --data /path/to/ratings.dat \
  --dataset-name MovieLens-1M \
  --delimiter "::" \
  --no-header \
  --column-names user_id item_id rating timestamp \
  --max-interactions 1500000 \
  --output-dir runs/ml1m
```

Amazon Book JSONL:

```bash
python train.py \
  --data /path/to/Books.jsonl \
  --dataset-name Amazon-Book \
  --user-col user_id \
  --item-col parent_asin \
  --timestamp-col timestamp \
  --max-interactions 1500000 \
  --output-dir runs/book
```

Amazon Beauty review JSON:

```bash
python train.py \
  --data /path/to/reviews_Beauty_5.json.gz \
  --dataset-name Amazon-Beauty \
  --user-col reviewerID \
  --item-col asin \
  --timestamp-col unixReviewTime \
  --max-interactions 1000000 \
  --validation-switch-epoch 300 \
  --output-dir runs/beauty
```

The paper defaults are built in: embedding dimension 64, two StateGNN layers,
maximum order 2, minimum state support 2, backoff weight 0.01, batch size 4096,
learning rate 0.001, weight decay 0.00001, seed 42, and 800 epochs. Full
validation runs every 50 epochs through epoch 400 and every 25 epochs afterward.

Checkpoints are selected strictly by full-ranking validation HR@10. The test set
is evaluated once after training using the validation-selected checkpoint.

## Reproducing the main model

Maximum-order experiments use the same main entry point:

```bash
python train.py --data /path/to/data.csv --max-order 3 --output-dir runs/k3
```

Only the main paper model is included in this anonymous release. Ablation and
auxiliary analysis runners are intentionally omitted.

## Outputs

Each run writes:

- `run_config.json`: data, model, graph, and optimization metadata;
- `state_graph.json`: retained states, reliability, parents, and directed edges;
- `training_curve.json`: training loss and validation metrics;
- `best.pt`: checkpoint selected by validation HR@10;
- `final_results.json`: validation and final full-ranking test metrics.

## Code structure

```text
train.py                 paper-aligned training and model selection
trainBPR.py              compatibility entry point for the original layout
vogrec/data.py           generic loading, deduplication, iterative 5-core, split
vogrec/graph.py          variable-order state discovery and graph construction
vogrec/model.py          initialization, StateGNN, routing, resolution score
vogrec/train_utils.py    tensorization, negative sampling, target lookup, BPR
vogrec/evaluation.py     complete-catalog HR/NDCG/MRR evaluation
tests/test_pipeline.py   preprocessing, graph, model, and smoke tests
```

Raw datasets are not redistributed. Obtain them from their official sources and
pass their local paths through `--data`.
