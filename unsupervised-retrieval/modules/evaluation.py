import os
import numpy as np
from modules.utils import write_trec_run


def read_qrels(root_folder="./datasets/"):
    """
        Reads the qrels.text file.
        Output: A dictionary: query_id -> [list of relevant documents]
    """
    with open(os.path.join(root_folder, "qrels.text")) as reader:
        lines = reader.readlines()

    from collections import defaultdict
    relevant_docs = defaultdict(set)
    for line in lines:
        query_id, doc_id, _, _ = line.split()
        relevant_docs[str(int(query_id))].add(doc_id)
    return relevant_docs


# TODO: Implement this!
def precision_k(results, relevant_docs, k):
    """
        Compute Precision@K
        Input:
            results: A sorted list of 2-tuples (document_id, score),
                    with the most relevant document in the first position
            relevant_docs: A set of relevant documents.
            k: the cut-off
        Output: Precision@K
    """
    if k > len(results):
        k = len(results)
    # BEGIN SOLUTION
    return sum(1 for i in range(k) if results[i][0] in relevant_docs) / k
    # END SOLUTION


# TODO: Implement this!
def recall_k(results, relevant_docs, k):
    """
        Compute Recall@K
        Input:
            results: A sorted list of 2-tuples (document_id, score), with the most relevant document in the first position
            relevant_docs: A set of relevant documents.
            k: the cut-off
        Output: Recall@K
    """
    # BEGIN SOLUTION
    return sum(1 for i in range(k) if results[i][0] in relevant_docs) / len(relevant_docs)
    # END SOLUTION


# TODO: Implement this!
def average_precision(results, relevant_docs):
    """
        Compute Average Precision (for a single query - the results are
        averaged across queries to get MAP in the next few cells)
        Hint: You can use the recall_k and precision_k functions here!
        Input:
            results: A sorted list of 2-tuples (document_id, score), with the most
                    relevant document in the first position
            relevant_docs: A set of relevant documents.
        Output: Average Precision
    """
    # BEGIN SOLUTION
    return sum([precision_k(results, relevant_docs, k + 1) for k in range(len(results)) if results[k][0] in relevant_docs]) / len(relevant_docs)
    # END SOLUTION

def evaluate_search_fn(method_name, search_fn, metric_fns, dh, queries, qrels, index_set=None):
    # build a dict query_id -> query
    queries_by_id = dict((q[0], q[1]) for q in queries)

    metrics = {}
    for metric, metric_fn in metric_fns:
        metrics[metric] = np.zeros(len(qrels), dtype=np.float32)

    q_results = {}
    for i, (query_id, relevant_docs) in enumerate(qrels.items()):
        query = queries_by_id[query_id]
        results = search_fn(query, dh)
        
        q_results[query_id] = results
        for metric, metric_fn in metric_fns:
            metrics[metric][i] = metric_fn(results, relevant_docs)

    write_trec_run(q_results, f'{method_name}_{index_set}.trec')

    final_dict = {}
    for metric, metric_vals in metrics.items():
        final_dict[metric] = metric_vals.mean()

    return final_dict
