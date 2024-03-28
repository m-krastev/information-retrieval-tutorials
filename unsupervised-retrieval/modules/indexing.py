
# TODO: Implement this!
def build_tf_index(documents):
    """
        Build an inverted index that maps tokens to inverted lists. The output is a dictionary which takes a token
        and returns a list of (document_id, count) tuples, where 'count' is the count of the 'token' in 'document_id'
        Input: a list of documents: (document_id, tokens)
        Output: An inverted index: [token] -> [(document_id, token_count)]
    """
    # BEGIN SOLUTION
    from collections import Counter
    out = {}
    for id, tokens in documents:
        counter = Counter(tokens)
        for token, count in counter.items():
            if token not in out:
                out[token] = []
            out[token].append((id, count))

    return out
    # END SOLUTION