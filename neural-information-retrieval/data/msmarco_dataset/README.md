# Installation

Run the following commands to download the necessary dataset:

```bash
# Install ir_datasets
pip install ir_datasets

# Download the queries
ir_datasets export msmarco-passage/train/split200-valid queries --format tsv > dev_queries.tsv

# Download the documents
ir_datasets export msmarco-passage/train/split200-valid docs --format tsv > collection.tsv

# Download the BM25 scored documents for reranking where needed.
ir_datasets export msmarco-passage/train/split200-valid scoreddocs --format trec > dev_bm25.trec

# Download the relevance judgments
ir_datasets export msmarco-passage/train/split200-valid qrels --format jsonl > dev_qrels.jsonl

# Process the relevance judgments to the used JSON format.
python process_qrels.py

# Download the triplets
ir_datasets export msmarco-passage/train/split200-valid docpairs --format tsv >  dev_triplets.tsv

# Remove the original relevance judgments
rm dev_qrels.jsonl
```
