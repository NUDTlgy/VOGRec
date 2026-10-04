# VOGRec: Variable-Order Item-State Graph Learning for Recommendation

This repository contains the self-contained no-item-bias reference implementation of VOGRec. It implements support-constrained variable-order state discovery, exclusive directed state flows, reliability-guided suffix backoff, directed StateGNN encoding, personalized source routing, candidate-conditioned target resolution, and BPR ranking.

The released code is the no-item-bias model used in our experiments. Its ranking score is the scaled inner product between the routed source representation and the candidate-conditioned target state. No trainable item-bias vector is used.

## Requirements

```bash
python -m pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

## Data interface

The loader accepts JSON, JSONL, compressed JSON/JSONL, CSV, TSV, and other delimited text files. Each input needs user ID, item ID, and timestamp columns. Common aliases such as `reviewerID`, `asin`, `parent_asin`, `unixReviewTime`, `userId`, and `itemId` are recognized.

The preprocessing protocol is:

1. sort each user's interactions chronologically;
2. remove repeated user-item interactions, keeping the latest event;
3. iteratively filter users and items until both have at least five events;
4. use the penultimate event for validation and the final event for testing;
5. construct all state statistics and graphs from training sequences only.

Deduplication is enabled by default. Use `--keep-duplicates` only for controlled diagnostics.

## Training

MovieLens-style `::` data without a header:

```bash
python -m VOGRec.trainBPR \
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
python -m VOGRec.trainBPR \
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
python -m VOGRec.trainBPR \
  --data /path/to/reviews_Beauty_5.json.gz \
  --dataset-name Amazon-Beauty \
  --user-col reviewerID \
  --item-col asin \
  --timestamp-col unixReviewTime \
  --max-interactions 1000000 \
  --validation-switch-epoch 300 \
  --output-dir runs/beauty
```

The paper defaults are built in: embedding dimension 64, two StateGNN layers, maximum order 2, minimum state support 2, backoff weight 0.01, batch size 4096, learning rate 0.001, weight decay 0.00001, seed 42, and 800 epochs. Checkpoints are selected by full-ranking validation HR@10; the test set is evaluated once after selection.

For maximum-order experiments, use the same entry point:

```bash
python -m VOGRec.trainBPR --data /path/to/data.csv --max-order 3 --output-dir runs/k3
```



## Outputs

Each run writes `run_config.json`, `state_graph.json`, `training_curve.json`, `best.pt`, and `final_results.json`.

## Code structure

```text
VOGRec/trainBPR.py       paper-aligned BPR training and model selection
VOGRec/__init__.py       public package exports
VOGRec/data.py           loading, deduplication, 5-core filtering, split
VOGRec/graph.py          variable-order state discovery and graph construction
VOGRec/model.py          no-item-bias StateGNN, routing, and target resolution
VOGRec/train_utils.py    tensorization, negative sampling, target lookup, BPR
VOGRec/evaluation.py     full-catalog HR/NDCG/MRR evaluation
VOGRec/tests/test_pipeline.py  preprocessing, graph, model, and smoke tests
```

Raw datasets are not redistributed. Obtain them from their official sources and pass their local paths through `--data`.
