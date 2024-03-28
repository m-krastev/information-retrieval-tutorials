#!/usr/bin/env python
# coding: utf-8

# # Homework 1 <a class="anchor" id="top"></a>
# 




# imports
# TODO: Ensure that no additional library is imported in the notebook.
# TODO: Only the standard library and the following libraries are allowed:
# TODO: You can also use unlisted classes from these libraries or standard libraries (such as defaultdict, Counter, ...).

from functools import partial

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from collections import namedtuple


# 
# # Part 1: Term-based Matching <a class="anchor" id="part1"></a>
# 
# [Back to top](#top)
# 
# In the first part, we will learn the basics of IR from loading and preprocessing the material, to implementing some well known search algorithms, to evaluating the ranking performance of the implemented algorithms. We will be using the CACM dataset throughout the assignment. The CACM dataset is a collection of titles and abstracts from the journal CACM (Communication of the ACM).
# 
# Table of contents:
# - [Section 1: Text Processing](#text_processing)
# - [Section 2: Indexing](#indexing)
# - [Section 3: Ranking](#ranking)
# - [Section 4: Evaluation](#evaluation)
# - [Section 5: Analysis](#analysis)
# 

# ---
# ## Section 1: Text Processing <a class="anchor" id="text_processing"></a>
# 
# [Back to Part 1](#part1)
# 
# In this section, we will load the dataset and learn how to clean up the data to make it usable for an IR system.
# First, go through the implementation of the following functions:
# - `read_cacm_docs`: Reads in the CACM documents.
# - `read_queries`: Reads in the CACM queries.
# - `load_stopwords`: Loads the stopwords.
# 
# The points of this section are earned for the following implementations:
# - `tokenize`: Tokenizes the input text.
# - `stem_token`: Stems the given token.
# 
# We are using the [CACM dataset](http://ir.dcs.gla.ac.uk/resources/test_collections/cacm/), which is a small, classic IR dataset, composed of a collection of titles and abstracts from the journal CACM. It comes with relevance judgements for queries, so we can evaluate our IR system.
# 

# ---
# ### 1.1 Read the CACM documents
# 
# The following cell downloads the dataset and unzips it to a local directory.




from modules.utils import download_dataset

download_dataset()


# ---
# 
# You can see a brief description of each file in the dataset by looking at the README file:




##### Read the README file
with open ("./datasets/README","r") as file:
    readme = file.read()
    print(readme)
#####


# ---
# We are interested in 4 files:
# - `cacm.all` : Contains the text for all documents. Note that some documents do not have abstracts available
# - `query.text` : The text of all queries
# - `qrels.text` : The relevance judgements
# - `common_words` : A list of common words. This may be used as a collection of stopwords




##### The first 45 lines of the CACM dataset forms the first record
# We are interested only in 3 fields.
# 1. the '.I' field, which is the document id
# 2. the '.T' field (the title) and
# 3. the '.W' field (the abstract, which may be absent)
with open ("./datasets/cacm.all","r") as file:
    cacm_all = "".join(file.readlines()[:45])
    print(cacm_all)
#####


# ---
# 
# The following function reads the `cacm.all` file. Note that each document has a variable number of lines. The `.I` field denotes a new document:




##### Function
from modules.dataset import read_cacm_docs

docs = read_cacm_docs()
n_docs = len(docs)

assert isinstance(docs, list)
assert n_docs == 3204, "There should be exactly 3204 documents"

unzipped_docs = list(zip(*docs))
assert np.sum(np.array(list(map(int,unzipped_docs[0])))) == 5134410


# ### 1.2 Read the CACM queries
# 
# Next, let us read the queries. They are formatted similarly:




##### The first 15 lines of 'query.text' has 2 queries
# We are interested only in 2 fields.
# 1. the '.I' - the query id
# 2. the '.W' - the query
with open ("./datasets/query.text","r") as file:
    query_file = "".join(file.readlines()[:16])
    print(query_file)
#####


# ---
# 
# The following function reads the `query.text` file:




from modules.dataset import read_queries

##### Function check
queries = read_queries()

assert isinstance(queries, list)
assert len(queries) == 64 and all([q[1] is not None for q in queries]), "There should be exactly 64 queries"

unzipped_queries = list(zip(*queries))
assert np.sum(np.array(list(map(int,unzipped_queries[0])))) == 2080


# ---
# ### 1.3 Read the stop words
# 
# We use the common words stored in `common_words`:




##### Read the stop words file
with open ("./datasets/common_words","r") as file:
    sw_file = "".join(file.readlines()[:10])
    print(sw_file)


# ---
# 
# The following function reads the `common_words` file:




from modules.preprocessing import load_stopwords
##### Function check
stopwords = load_stopwords()

assert isinstance(stopwords, set)
assert len(stopwords) == 428, "There should be exactly 428 stop words"

assert np.sum(np.array(list(map(len,stopwords)))) == 2234
#####


# ---
# ### 1.4 Tokenization 
# 
# We can now write some basic text processing functions.
# A first step is to tokenize the text.
# 
# **Note**: Use the  `WordPunctTokenizer` available in the `nltk` library:




from modules.preprocessing import tokenize

# ToDo:
# Implement the function 'tokenize'.

##### Function check
text = "the quick brown fox jumps over the lazy dog"
tokens = tokenize(text)

print(tokens)
# output: ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']


# ---
# ### 1.5 Stemming
# 
# Write a function to stem tokens.
# Again, you can use the nltk library for this:




from modules.preprocessing import stem_token
# ToDo:
# Implement the function 'stem_token'.

##### Function check

assert stem_token('test') is not None


# ---
# ### 1.6 Summary
# 
# The following function 'process_text' puts it all together. Given an input string, this functions tokenizes and processes it according to the flags that you set.
# This function is already implemented.




from modules.preprocessing import process_text


# ---
# Let's create two sets of preprocessed documents.
# We can process the documents and queries according to these two configurations:




# In this configuration:
# Don't preprocess the text, except to tokenize
config_1 = {
  "stem": False,
  "remove_stopwords" : False,
  "lowercase_text": True
}


# In this configuration:
# Preprocess the text, stem and remove stopwords
config_2 = {
  "stem": True,
  "remove_stopwords" : True,
  "lowercase_text": True,
}

####
doc_repr_1 = []
doc_repr_2 = []
for (doc_id, document) in docs:
    doc_repr_1.append((doc_id, process_text(document, stopwords, **config_1)))
    doc_repr_2.append((doc_id, process_text(document, stopwords, **config_2)))

####


# ---
# 
# ## Section 2: Indexing <a class="anchor" id="indexing"></a>
# 
# [Back to Part 1](#part1)
# 
# 
# 
# A retrieval function usually takes in a query document pair, and scores a query against a document.  Our document set is quite small - just a few thousand documents. However, consider a web-scale dataset with a few million documents. In such a scenario, it would become infeasible to score every query and document pair. In such a case, we can build an inverted index. From Wikipedia:
# 
# > ... , an inverted index (also referred to as a postings file or inverted file) is a database index storing a mapping from content, such as words or numbers, to its locations in a table, .... The purpose of an inverted index is to allow fast full-text searches, at a cost of increased processing when a document is added to the database. ...
# 
# 
# Consider a simple inverted index, which maps from word to document. This can improve the performance of a retrieval system significantly. In this assignment, we consider a *simple* inverted index, which maps a word to a set of documents. In practice, however, more complex indices might be used.
# 

# ### 2.1 Term Frequency-index 
# In this assignment, we will be using an index created in memory since our dataset is tiny. To get started, build a simple index that maps each `token` to a list of `(doc_id, count)` where `count` is the count of the `token` in `doc_id`.
# For consistency, build this index using a python dictionary.
# 
# Now, implement a function to build an index:




# ToDo:
# Implement the function 'build_tf_index'

from modules.indexing import build_tf_index


# ---
# Now we can build indexed documents and preprocess the queries based on the two configurations:




#### Indexed documents based on the two configs

# Create the 2 indices
tf_index_1 = build_tf_index(doc_repr_1)
tf_index_2 = build_tf_index(doc_repr_2)


# ---
# ## Section 3: Ranking <a class="anchor" id="ranking"></a>
# 
# [Back to Part 1](#part1)
# 
# Now that we have cleaned and processed our dataset, we can start building simple IR systems.
# 
# For now, we consider *simple* IR systems, which involve computing scores from the tokens present in the document/query. More advanced methods are covered in later assignments.
# 
# We will implement the following methods in this section:
# - [Section 3.1: TF-IDF](#tfidf) 
# - [Section 3.2: Query Likelihood Model](#qlm) 
# - [Section 3.3: BM25](#bm25) 
# 
# *All search functions should be able to handle multiple words queries.*
# 
# **Scoring policy:**
# Your implementations in this section are scored based on the expected performance of your ranking functions.
# You will get a full mark if your implementation meets the expected performance (measured by some evaluation metric).
# Otherwise, you may get partial credit.
# For example, if your *TF-IDF* ranking function has 60% of expected performance, you will get 6 out of 10.

# In order to check the output of your ranking functions, you can use the predefined function
# 'print_results'.




docs_by_id = dict(docs)
from modules.utils import print_results


# ### Section 3.1: TF-IDF <a class="anchor" id="tfidf"></a>
# 
# Before we implement the tf-idf scoring functions, let's first write a function to compute the document frequencies of all words.
# 
# #### 3.1.1 Document frequency
# Compute the document frequencies of all tokens in the collection.
# Your code should return a dictionary with tokens as its keys and the number of documents containing the token as values.
# For consistency, the values should have `int` type.
# 
# You can use the pre-defined class DocumentFrequencies and the function 'get_df' in the rest of the assignemnt to get
# document frequencies based on the index.




# ToDo:
#  Implement the following function 'compute_df'! 
from modules.ranking import compute_df

#### Compute df based on the two configs

# get the document frequencies of each document
df_1 = compute_df([d[1] for d in doc_repr_1])
df_2 = compute_df([d[1] for d in doc_repr_2])


# To make the implementation of ranking functions smoother, you can use the class Dataset in modules.dataset that
# includes helper functions to get information about documents and indexes. Through this class you can employ the
# following functions:
# 
#     - get_df
#     - get_doc_lengths
#     - get_index
#     - preprocess_query




from modules.dataset import Dataset

dh1 = Dataset(n_docs, docs_by_id, doc_repr_1, df_1, tf_index_1, stopwords, config_1)

dh2 = Dataset(n_docs, docs_by_id, doc_repr_2, df_2, tf_index_2, stopwords, config_2)





#### Function check

print(df_1['computer'])
print(df_2['computer'])


# #### 3.1.2 TF-IDF search 
# Next, implement a function that computes a tf-idf score, given a query.
# Use the following formulas for TF and IDF:
# 
# $$ TF=\log (1 + f_{d,t}) $$
# 
# $$ IDF=\log (\frac{N}{n_t})$$
# 
# where $f_{d,t}$ is the frequency of token $t$ in document $d$, $N$ is the number of total documents and $n_t$ is the number of documents containing token $t$.
# 
# **Note:** your implementation will be auto-graded assuming you have used the above formulas.
# 




# TODO:
# Implement the following functions
from modules.ranking import tfidf_tf_score, tfidf_idf_score, tfidf_term_score, tfidf_search





#### Function check
test_tfidf = tfidf_search("computer word search", dh2)[:5]
print(f"TFIDF Results:")
print_results(test_tfidf, docs_by_id)


# ---
# 
# ### Section 3.2: Query Likelihood Model  <a class="anchor" id="qlm"></a>
# 
# In this section, you will implement a simple query likelihood model.
# 
# 
# #### 3.2.1 Naive QL 
# 
# First, let us implement a naive version of a QL model, assuming a multinomial unigram language model (with a uniform prior over the documents).
# You should ignore any querm term that does not appear in the collection.




# ToDo:
#  Implement the following functions
from modules.ranking import naive_ql_document_scoring, naive_ql_document_ranking, naive_ql_search





#### Function check
test_naiveql = naive_ql_search("report", dh1)[:5]
print(f"Naive QL Results:")
print_results(test_naiveql, docs_by_id)


# ---
# #### 3.2.2 QL
# Now, let's implement a QL model that handles the issues with the naive version. In particular, you will implement a QL model with Jelinek-Mercer Smoothing. That means an interpolated score is computed per word - one term is the same as the previous naive version, and the second term comes from a unigram language model. In addition, you should accumulate the scores by summing the **log** (smoothed) probability which leads to better numerical stability.




# ToDo:
# Implement the following functions
from modules.ranking import ql_background_model, ql_document_scoring, ql_search





#### Function check
test_ql_results = ql_search("report", dh1)[:5]
print_results(test_ql_results, docs_by_id)


# ---
# 
# ### Section 3.3: BM25 <a class="anchor" id="bm25"></a>
# 
# In this section, we will implement the BM25 scoring function.
# 




# ToDo: 
# Implement the following functions
from modules.ranking import bm25_tf_score, bm25_idf_score, bm25_term_score, bm25_search





#### Function check
test_bm25_results = bm25_search("report", dh1)[:5]
print_results(test_bm25_results, docs_by_id)


# 
# ---
# 
# ### 3.4. Test Your Functions
# 
# The widget below allows you to play with the search functions you've written so far. Use this to test your search functions and ensure that they work as expected.




#### Highlighter function
# class for results
ResultRow = namedtuple("ResultRow", ["doc_id", "snippet", "score"])
# doc_id -> doc
docs_by_id = dict((d[0], d[1]) for d in docs)

def highlight_text(document, query, tol=17):
    import re
    tokens = tokenize(query)
    regex = "|".join(f"(\\b{t}\\b)" for t in tokens)
    regex = re.compile(regex, flags=re.IGNORECASE)
    output = ""
    i = 0
    for m in regex.finditer(document):
        start_idx = max(0, m.start() - tol)
        end_idx = min(len(document), m.end() + tol)
        output += "".join(["...",
                        document[start_idx:m.start()],
                        "<strong>",
                        document[m.start():m.end()],
                        "</strong>",
                        document[m.end():end_idx],
                        "..."])
    return output.replace("\n", " ")


def make_results(query, search_fn, index_set):
    results = []
    for doc_id, score in search_fn(query, index_set):
        highlight = highlight_text(docs_by_id[doc_id], query)
        if len(highlight.strip()) == 0:
            highlight = docs_by_id[doc_id]
        results.append(ResultRow(doc_id, highlight, score))
    return results
####





make_results('Matrix Arrays', bm25_search, dh1)[:10]


# ---
# 
# ## Section 4: Evaluation <a class="anchor" id="evaluation"></a>
# 
# [Back to Part 1](#part1)
# 
# In order to analyze the effectiveness of retrieval algorithms, we first have to learn how to evaluate such a system. In particular, we will work with offline evaluation metrics. These metrics are computed on a dataset with known relevance judgements.
# 
# Implement the following evaluation metrics.
# 
# 1. Precision 
# 2. Recall 
# 3. Mean Average Precision 
# 4. Expected Reciprocal Rank

# ---
# ### 4.1 Read relevance labels
# 
# Let's take a look at the `qrels.text` file, which contains the ground truth relevance scores. The relevance labels for CACM are binary - either 0 or 1.
# 




##### Read the stop words file
with open ("./datasets/qrels.text","r") as file:
    qr_file = "".join(file.readlines())
    print(qr_file)


# ---
# 
# The first column is the query_id and the second column is the document_id. We can safely ignore the 3rd and 4th columns.
# You can use the implemented function 'read_qrels'.




from modules.evaluation import read_qrels

#### Function check
qrels = read_qrels()

assert len(qrels) == 52, "There should be 52 queries with relevance judgements"
assert sum(len(j) for j in qrels.values()) == 796, "There should be a total of 796 Relevance Judgements"


# ---
# **Note:** For a given query `query_id`, you can assume that documents *not* in `qrels[query_id]` are not relevant to `query_id`.
# 

# ---
# ### 4.2 Precision
# Implement the `precision@k` metric:




# ToDo:
# Implement the following function 'precision_k'!
from modules.evaluation import precision_k

#### Function check
qid = queries[0][0]
qtext = queries[0][1]
print(f'query:{qtext}')
results = bm25_search(qtext, dh2)
precision = precision_k(results, qrels[qid], 10)
print(f'precision@10 = {precision}')
assert precision is not None


# ---
# ### 4.3 Recall 
# Implement the `recall@k` metric:




# ToDo:
# Implement the following function
from modules.evaluation import recall_k

#### Function check
qid = queries[10][0]
qtext = queries[10][1]
print(f'query:{qtext}')
results = bm25_search(qtext, dh2)
recall = recall_k(results, qrels[qid], 10)
print(f'recall@10 = {recall}')
assert recall is not None


# ---
# ### 4.4 Mean Average Precision
# Implement the `map` metric:




# ToDo:
# Implement the following function
from modules.evaluation import average_precision

#### Function check
qid = queries[20][0]
qtext = queries[20][1]
print(f'query:{qtext}')
results = bm25_search(qtext, dh2)
mean_ap = average_precision(results, qrels[qid])
print(f'MAP = {mean_ap}')
assert mean_ap is not None

