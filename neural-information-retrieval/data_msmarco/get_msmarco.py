import ir_datasets
import csv
import json
from pathlib import Path

def save_to_tsv(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t").writerows(data)

def save_qrels_like_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f)

path = Path("data_msmarco")
path.mkdir(exist_ok=True)

query_path = path / "dev_queries.tsv"
collection_path = path / "collection.tsv"
qrel_path = path / "dev_qrels.json"

dataset = ir_datasets.load("msmarco-passage/dev/small")

queries = [(query.query_id, query.text) for query in dataset.queries_iter()[:500]]

save_to_tsv(queries, query_path)
print(f"Saved 500 queries from the MSMARCO dataset to {query_path}.")

# Save the collection and qrels in a format that can be used by the BM25 implementation
qrels_data = {}
filtered_docs = {}
for qrel in dataset.qrels_iter():
    if qrel.query_id not in qrels_data and str(qrel.query_id) in dict(queries):
        qrels_data[qrel.query_id] = {}
    if str(qrel.query_id) in dict(queries):
        qrels_data[qrel.query_id][qrel.doc_id] = qrel.relevance
        filtered_docs[qrel.doc_id] = True

doc_store = dataset.docs_store()
doc_func = lambda doc: (doc.doc_id, doc.text)
docs = [doc_func(doc_store.get(doc))
    for doc in filtered_docs
]

# Save the collection and qrels
save_to_tsv(docs, collection_path)
save_qrels_like_json(qrels_data, qrel_path)

print(
    "Processed and saved a small subset (500 queries + related documents) of the MSMARCO dataset."
)
