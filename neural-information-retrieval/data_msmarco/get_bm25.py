import csv
from rank_bm25 import BM25Okapi

def read_tsv(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        return list(reader)

def tokenize(text):
    return text.lower().split()

queries = read_tsv('data_msmarco/dev_queries.tsv')
queries = {query_id: tokenize(text) for query_id, text in queries}

docs = read_tsv('data_msmarco/collection.tsv')
doc_ids, doc_texts = zip(*docs)
doc_texts_tokenized = [tokenize(text) for text in doc_texts]

bm25 = BM25Okapi(doc_texts_tokenized)

with open('data_msmarco/bm25.trec', 'w') as f_out:
    for query_id, query in queries.items():
        scores = bm25.get_scores(query)
        ranked_doc_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        
        for rank, doc_idx in enumerate(ranked_doc_indices):
            score = scores[doc_idx]
            # if score > 0:
            doc_id = doc_ids[doc_idx]
            # TREC format: query_id, Q0, doc_id, rank, score, run_name
            f_out.write(f'{query_id} Q0 {doc_id} {rank + 1} {score} RUN\n')

print("Ranking complete and results written to bm25.trec.")
