from collections import Counter
from math import log
import numpy as np
from .dataset import Dataset


# TODO: Implement this!
def compute_df(documents):
    """
    Compute the document frequency of all terms in the vocabulary.
    Input: A list of documents
    Output: A dictionary with {token: document frequency}
    """
    # BEGIN SOLUTION
    from collections import Counter

    return Counter([token for tokens in documents for token in set(tokens)])
    # END SOLUTION


# TODO: Implement this!
def tfidf_tf_score(tf):
    """
    Apply the correct formula (see instructions file) for the term frequency of the tfidf search ranking.
    Input:
        tf - the simple term frequency, representing how many times a term appears in a document
    Output: as single value for the term frequency term after applied a formula on it
    """
    # BEGIN SOLUTION
    return log(1 + tf)
    # END SOLUTION


# TODO: Implement this!
def tfidf_idf_score(tdf, N):
    """
    Apply the correct formula (see instructions file) for the idf score of the tfidf search ranking.
    Input:
        tdf - the document frequency for a single term
        N - the total number of documents
    Output: as single value for the inverse document frequency
    """
    # BEGIN SOLUTION
    return log(N / tdf)
    # END SOLUTION


# TODO: Implement this!
def tfidf_term_score(tf, tdf, N):
    """
    Combine the tf score and the idf score.
    Hint: Use the tfidf_idf_score and the tfidf_tf_score functions
    Input:
        tf - the simple term frequency, representing how many times a term appears in a document
        tdf - the document frequency for a single term
        N - the total number of documents
    Output: a single value for the tfidf score
    """
    # BEGIN SOLUTION
    return tfidf_tf_score(tf) * tfidf_idf_score(tdf, N)
    # END SOLUTION


# TODO: Implement this!
def tfidf_search(query: str, dh: Dataset) -> list:
    """
    Perform a search over all documents with the given query using tf-idf.
    Hint: Use the tfidf_term_score method
    Input:
        query - a (unprocessed) query
        dh - a Dataset object, containing index, df, etc.
    Output: a list of (document_id, score), sorted in descending relevance to the given query
    """
    index = dh.get_index()
    df = dh.get_df()
    processed_query = dh.preprocess_query(query)
    N = dh.n_docs
    # BEGIN SOLUTION
    scores = {}
    for term in processed_query:
        for doc_id, tf in index.get(term, []):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += tfidf_term_score(tf, df[term], N)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # END SOLUTION


# TODO: Implement this!

def term_counts(dh: Dataset):
    return Counter([token for id, tokens in dh.doc_rep for token in tokens])

def naive_ql_document_scoring(query: str, dh: Dataset) -> list:
    """
    Perform a search over all documents with the given query using a naive QL model.
    Input:
        query - a (unprocessed) query
        dh - a Dataset object, containing index, df, etc.
    Output: a list of (document_id, score), unsorted in relevance to the given query
    """
    index = dh.get_index()
    _doc_lengths = dh.get_doc_lengths()
    processed_query = dh.preprocess_query(query)
    term_counts = Counter([token for _, tokens in dh.doc_rep for token in tokens])
    # BEGIN SOLUTION
    scores = {}
    for term in processed_query:
        for doc_id, tf in index.get(term, []):
            if doc_id not in scores:
                scores[doc_id] = 0
            term_count = term_counts[term]
            scores[doc_id] += log(term_count) - log(_doc_lengths[doc_id])

    return list(scores.items())
    # END SOLUTION


# TODO: Implement this!
def naive_ql_document_ranking(results: list) -> list:
    """
    Sort the results.
    Input:
        result - a list of (document_id, score), unsorted in relevance to the given query
    Output: a list of (document_id, score), sorted in descending relevance to the given query
    """
    # BEGIN SOLUTION
    return sorted(results, key=lambda x: x[1], reverse=True)
    # END SOLUTION


# TODO: Implement this!
def naive_ql_search(query: str, dh: Dataset) -> list:
    """
    1. Perform a search over all documents with the given query using a naive QL model,
    using the method naive_ql_document_scoring
    2. Sort the results using the method naive_ql_document_ranking
    Input:
        query - a (unprocessed) query
        dh - a Dataset object, containing index, df, etc.
    Output: a list of (document_id, score), sorted in descending relevance to the given query
    """
    # BEGIN SOLUTION
    scores = naive_ql_document_scoring(query, dh)
    return naive_ql_document_ranking(scores)
    # END SOLUTION


# TODO: Implement this!
def ql_background_model(query: str, dh: Dataset) -> tuple:
    """
    Compute the background model of the smooth ql function.
    Input:
        query - a (unprocessed) query
        dh - a Dataset object, containing index, df, etc.
    Output: a tuple consisting of a (query, dh, collection_prob:dict with {term: collection frequency of term/collection_length})
    """
    # BEGIN SOLUTION
    collection_prob = {}
    collection_length = dh.n_docs
    term_counts = Counter([token for id, tokens in dh.doc_rep for token in tokens])
    for term, table in dh.get_index().items():
        collection_prob[term] = term_counts[term] / collection_length
    return query, dh, collection_prob
    # END SOLUTION


# TODO: Implement this!
def ql_document_scoring(
    query, dh: Dataset, collection_prob: dict, smoothing: float = 0.1
) -> list:
    """
    Perform a search over all documents with the given query using a QL model
    with Jelinek-Mercer Smoothing (with default smoothing=0.1).
    Input:
        query - a (unprocessed) query
        dh - a Dataset object, containing index, df, etc.
        collection_prob - a dictionary with {term: collection frequency of term/collection_length}
        smoothing - the smoothing parameter (lambda parameter in the smooth QL equation)
    Output: a list of (document_id, score), sorted in descending relevance to the given query
    """
    index = dh.get_index()
    doc_lengths = dh.get_doc_lengths()
    processed_query = dh.preprocess_query(query)
    # BEGIN SOLUTION
    for term in processed_query:
        if term not in collection_prob:
            collection_prob[term] = 0
    scores = {}
    for term in processed_query:
        for doc_id, tf in index.get(term, []):
            if doc_id not in scores:
                scores[doc_id] = 0
            term_count = sum(
                [count for token, count in index[term] if token == doc_id]
            )
            scores[doc_id] += log(
                (1 - smoothing) * term_count
                + smoothing * collection_prob[term]
            )

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # END SOLUTION


# TODO: Implement this!
def ql_search(query: str, dh: Dataset, smoothing: float = 0.1) -> list:
    """
    Perform a search over all documents with the given query using a QL model
    with Jelinek-Mercer Smoothing (set smoothing=0.1).

    1. Create the background model using the method ql_background_model
    2. Perform a search over all documents with the given query using a QL model,
    using the method ql_document_scoring
    3. Sort the results using the method ql_document_ranking

    Note #1: You might have to create some variables beforehand and use them in this function

    Input:
        query - a (unprocessed) query
        dh - a Dataset object, containing index, df, etc.
        smoothing - the smoothing parameter (lambda parameter in the smooth QL equation)
    Output: a list of (document_id, score), sorted in descending relevance to the given query
    """
    # BEGIN SOLUTION
    query, dh, collection_prob = ql_background_model(query, dh)
    documents = ql_document_scoring(query, dh, collection_prob, smoothing)
    return naive_ql_document_ranking(documents)
    # END SOLUTION


# TODO: Implement this!
def bm25_tf_score(tf, doclen, avg_doc_len, k_1, b):
    """
    Compute the bm25 tf score that uses two parts. The numerator,
    and the denominator.
    Input:
        tf - the term frequency used in bm25
        doclen - the document length of
        avg_doc_len - the average document length
        k_1 - contant of bm25
        b - constant of bm25
    Output: a single value for the tf part of bm25 score
    """
    # BEGIN SOLUTION
    return ((k_1 + 1) * tf) / (k_1 * (1 - b + b * (doclen / avg_doc_len)) + tf)
    # END SOLUTION


# TODO: Implement this!
def bm25_idf_score(df, N):
    """
    Compute the idf part of the bm25 and return its value.
    Input:
        df - document frequency
        N - total number of documents
    Output: a single value for the idf part of bm25 score
    """
    # BEGIN SOLUTION
    return log(N / df)
    # END SOLUTION


# TODO: Implement this!
def bm25_term_score(tf, df, doclen, avg_doc_len, k_1, b, N):
    """
    Compute the term score part of the bm25 and return its value.
    Hint 1: Use the bm25_tf_score method.
    Hint 2: Use the bm25_idf_score method.
    Input:
        tf - the term frequency used in bm25
        doclen - the document length of
        avg_doc_len - the average document length
        k_1 - contant of bm25
        b - constant of bm25
        df - document frequency
        N - total number of documents
        Output: a single value for the term score of bm25
    """
    # BEGIN SOLUTION
    return bm25_tf_score(tf, doclen, avg_doc_len, k_1, b) * bm25_idf_score(df, N)
    # END SOLUTION


# TODO: Implement this!
def bm25_search(query, dh: Dataset):
    """
    Perform a search over all documents with the given query using BM25. Use k_1 = 1.5 and b = 0.75
    Note #1: You have to use the `get_index` (and `get_doc_lengths`) function created in the previous cells
    Note #2: You might have to create some variables beforehand and use them in this function
    Hint: You have to use the bm25_term_score method
    Input:
        query - a (unprocessed) query
        dh - instance of a Dataset
    Output: a list of (document_id, score), sorted in descending relevance to the given query
    """

    index = dh.get_index()
    df = dh.get_df()
    doc_lengths = dh.get_doc_lengths()
    processed_query = dh.preprocess_query(query)
    # BEGIN SOLUTION
    k1 = 1.5
    b = 0.75
    N = dh.n_docs
    avg_doc_len = sum(doc_lengths.values()) / N

    for term in processed_query:
        if term not in df:
            df[term] = 0

    scores = {}
    for term in processed_query:
        for doc_id, tf in index.get(term, []):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += bm25_term_score(
                tf, df[term], doc_lengths[doc_id], avg_doc_len, k1, b, N
            )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # END SOLUTION
