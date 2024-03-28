#!/usr/bin/env python

# %%
import argparse
import csv
import os
import pickle
import random
from argparse import Namespace

import ir_datasets
import numpy as np
import torch
from ltr.dataset import (
    DataSet,
    Documents,
    FeatureExtraction,
    GenerateFeatures,
    Preprocess,
    Queries,
)
from ltr.model import LTRModel
from ltr.train import train_listwise, train_pairwise_spedup, train_pointwise
from ltr.utils import seed
from tqdm import tqdm
from ltr.utils import create_results

seed(42)


# %%
def save_tsv(
    folder_name,
    dataset_path,
    concatenate_docs=False,
    doc_text=None,
    extract_partial=False,
    extract=None,
    provide_n_docs=False,
    n_docs=None,
    **kwargs,
):
    """
    Saves queries, documents and qrels in asisgnment-specific .tsv format.

    Input:
        - dataset_path: dataset to be downloaded using ir_dataset package
        - folder_name: name of the folder within the data_for_analysis folder
    """
    # Check if the folder exists, and if not, create it
    folder_path = "analysis_data/" + folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        dataset = ir_datasets.load(dataset_path)

    # Queries
    print(f"Extracting queries of {folder_name} ...")
    query_path = os.path.join(folder_path, "queries.tsv")
    if not os.path.exists(query_path):
        total_queries = sum(1 for _ in dataset.queries_iter())
        with open(query_path, "w", newline="", encoding="utf-8") as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter="\t")
            for query in tqdm(
                dataset.queries_iter(), total=total_queries, desc="Saving queries"
            ):
                tsv_writer.writerow([query[0], query[1]])

    # Docs
    print(f"Extracting documents of {folder_name} ...")
    doc_path = os.path.join(folder_path, "collection.tsv")
    if not os.path.exists(doc_path):
        total_docs = n_docs if provide_n_docs else sum(1 for _ in dataset.docs_iter())
        (
            random.sample(list(dataset.docs_iter()), extract)
            if extract_partial
            else dataset.docs_iter()
        )

        with open(doc_path, "w", newline="", encoding="utf-8") as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter="\t")
            for doc in tqdm(dataset.docs_iter(), total=total_docs, desc="Saving docs"):
                if concatenate_docs:
                    tsv_writer.writerow(
                        [doc[0], f"{doc[doc_text[0]]} {doc[doc_text[1]]}"]
                    )
                else:
                    tsv_writer.writerow([doc[0], doc[1]])

    # Qrels
    print(f"Extracting qrels of {folder_name} ...")
    ## Initialize file handles only if they don't exist
    train_file_path = os.path.join(folder_path, "train_pairs_graded.tsv")
    os.path.join(folder_path, "dev_pairs_graded.tsv")
    os.path.join(folder_path, "test_pairs_graded.tsv")

    if not os.path.exists(train_file_path):
        total_qrels = sum(1 for _ in dataset.qrels_iter())

        ## Calculate the indices to split the data
        index_80_percent = int(0.8 * total_qrels)
        index_90_percent = int(0.9 * total_qrels)

        with open(
            os.path.join(folder_path, "train_pairs_graded.tsv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as train_file, open(
            os.path.join(folder_path, "dev_pairs_graded.tsv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as dev_file, open(
            os.path.join(folder_path, "test_pairs_graded.tsv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as test_file:
            train_writer = csv.writer(train_file, delimiter="\t")
            dev_writer = csv.writer(dev_file, delimiter="\t")
            test_writer = csv.writer(test_file, delimiter="\t")

            for i, qrel in tqdm(
                enumerate(dataset.qrels_iter()), total=total_qrels, desc="Saving qrels"
            ):
                if i < index_80_percent:
                    train_writer.writerow([qrel[0], qrel[1], qrel[2]])
                elif i < index_90_percent:
                    dev_writer.writerow([qrel[0], qrel[1], qrel[2]])
                else:
                    test_writer.writerow([qrel[0], qrel[1], qrel[2]])

# %%
DATASETS = {
    "COVID": {
        "dataset_path": "beir/trec-covid",
        "concatenate_docs": True,
        "doc_text": [2, 1],
    },
    "GENOMICS": {
        "dataset_path": "medline/2004/trec-genomics-2005",
        "concatenate_docs": True,
        "doc_text": [1, 2],
    },
    "ARGS": {
        "dataset_path": "argsme/2020-04-01/touche-2021-task-1",
        "concatenate_docs": True,
        "doc_text": [3, 1],
    },
    "GAMING": {
        "dataset_path": "beir/cqadupstack/gaming",
        "concatenate_docs": True,
        "doc_text": [2, 1],
    },
    "NUTRITION": {
        "dataset_path": "nfcorpus/train/nontopic",
        "concatenate_docs": True,
        "doc_text": [2, 3],
    },
    "CLIMATE": {
        "dataset_path": "beir/climate-fever",
        "concatenate_docs": True,
        "doc_text": [2, 1],
    },
    "BUSINESS": {
        "dataset_path": "beir/fiqa/train",
    },
}

def load_data(args, SCENARIO, COLLECTION_PATH, QUERIES_PATH, TRAIN_PATH, DEV_PATH, TEST_PATH, STOP_WORDS_PATH, DOC_JSON, kwargs):
    print(f"Saving {SCENARIO} ...")
    save_tsv(SCENARIO, **DATASETS[SCENARIO])
    prp = Preprocess(STOP_WORDS_PATH)
    queries = Queries(prp)
    queries.preprocess_queries(QUERIES_PATH)
    if os.path.exists(DOC_JSON) and not args.reset:
        with open(DOC_JSON, "rb") as file:
            documents = pickle.load(file)
    else:
        documents = Documents(prp)
        documents.process_documents(COLLECTION_PATH)
        with open(DOC_JSON, "wb") as file:
            pickle.dump(documents, file)

    feature_ex = FeatureExtraction({}, documents, queries)
    feat_gen = GenerateFeatures(feature_ex)

    feat_gen.run(TRAIN_PATH, TRAIN_PATH + "g", **kwargs)
    feat_gen.run(DEV_PATH, DEV_PATH + "g", **kwargs)
    feat_gen.run(TEST_PATH, TEST_PATH + "g", **kwargs)

    fold_paths = [f"./analysis_data/{SCENARIO}/"]
    num_relevance_labels = 3
    num_nonzero_feat = args.n_features
    num_unique_feat = args.n_features
    data = DataSet(
        "ir1-2023", fold_paths, num_relevance_labels, num_unique_feat, num_nonzero_feat
    )

    data = data.get_data_folds()[0]
    data.read_data()
    return data


# %%
def analyze_for_15_features(args):
    args.n_features = 15
    kwargs = vars(args)

    data = load_data(
        args,
        SCENARIO,
        COLLECTION_PATH,
        QUERIES_PATH,
        TRAIN_PATH,
        DEV_PATH,
        TEST_PATH,
        STOP_WORDS_PATH,
        DOC_JSON,
        kwargs,
    )

    params_regr = Namespace(
        epochs=11,
        lr=1e-3,
        batch_size=1,
        metrics={"ndcg", "precision@05", "recall@05", "recall@100"},
    )

    pointwise_regression_model = LTRModel(data.num_features)
    regr = create_results(
        data,
        pointwise_regression_model,
        train_pointwise,
        pointwise_regression_model,
        args.output + "analysis_pointwise_res_15.json",
        params_regr,
    )

    model_savepath = args.output + "analysis_pointwise_model_15"

    torch.save(pointwise_regression_model.state_dict(), model_savepath)

    # %%
    # Pairwise
    params = Namespace(
        epochs=10,
        lr=1e-3,
        batch_size=1,
        metrics={"ndcg", "precision@05", "recall@05", "recall@100"},
    )
    sped_up_pairwise_model = LTRModel(data.num_features)

    pw_res = create_results(
        data,
        sped_up_pairwise_model,
        train_pairwise_spedup,
        sped_up_pairwise_model,
        args.output + "analysis_pairwise_spedup_res_15.json",
        params,
    )

    model_savepath = args.output + "analysis_pairwise_spedup_model_15"

    torch.save(sped_up_pairwise_model.state_dict(), model_savepath)

    # %%
    # Listwise
    params_regr = Namespace(
        epochs=11,
        lr=1e-4,
        batch_size=1,
        metrics={"ndcg", "precision@05", "recall@05", "recall@100"},
    )

    results_path = args.output + "analysis_listwise_res_15.json"
    listwise_model = LTRModel(data.num_features)
    listwise_res = create_results(
        data, listwise_model, train_listwise, listwise_model, results_path, params_regr
    )

    model_savepath = args.output + "analysis_listwise_model_15"

    torch.save(listwise_model.state_dict(), model_savepath)
    print(f"Model saved at {model_savepath}")

    return regr, pw_res, listwise_res


# %%
def analyze_17_vs_15(
    res_regr_15,
    res_pw_15,
    res_listwise_15,
    res_regr_17,
    res_pw_17,
    res_listwise_17,
    target_metric="ndcg",
):
    res_regr_15 = np.array(res_regr_15["test_query_level_metrics"][target_metric])
    res_pw_15 = np.array(res_pw_15["test_query_level_metrics"][target_metric])
    res_listwise_15 = np.array(
        res_listwise_15["test_query_level_metrics"][target_metric]
    )
    res_regr_17 = np.array(res_regr_17["test_query_level_metrics"][target_metric])
    res_pw_17 = np.array(res_pw_17["test_query_level_metrics"][target_metric])
    res_listwise_17 = np.array(
        res_listwise_17["test_query_level_metrics"][target_metric]
    )

    # Focus on listwise
    diff = res_listwise_17 - res_listwise_15
    print(f"Mean difference in {target_metric} for listwise: {diff.mean()}")
    sorted_indices = np.argsort(diff)[::-1]

    top_indices = sorted_indices[:5]
    bottom_indices = sorted_indices[-5:][::-1]

    with open(QUERIES_PATH, "r") as file:
        queries = [line.split("\t")[1] for line in file.readlines()]
        
    print(f"The additional features improved the scores of {sum(diff > 0)} out of {len(diff)} queries.")

    print(f"Top 5 queries with the highest positive difference in {target_metric}")
    for i in top_indices:
        print(f"{target_metric.upper()} {diff[i]}", queries[i], sep="\t")

    print(f"Top 5 queries with the highest negative difference in {target_metric}")
    for i in bottom_indices:
        print(f"{target_metric.upper()} {diff[i]}", queries[i], sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="NUTRITION")
    parser.add_argument("--reset", type=bool, default=False)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--idf_smoothing", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=11)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--metrics", type=set, default={"ndcg", "precision@05", "recall@05"}
    )
    parser.add_argument("--output", type=str, default="./outputs/")
    parser.add_argument("--n_features", type=int, default=17)
    parser.add_argument("--analyze_for_15_features", type=bool, default=False)
    args = parser.parse_args()

    SCENARIO = args.scenario.upper()
    # %%
    COLLECTION_PATH = f"./analysis_data/{SCENARIO}/collection.tsv"
    QUERIES_PATH = f"./analysis_data/{SCENARIO}/queries.tsv"
    TRAIN_PATH = f"./analysis_data/{SCENARIO}/train_pairs_graded.tsv"
    DEV_PATH = f"./analysis_data/{SCENARIO}/dev_pairs_graded.tsv"
    TEST_PATH = f"./analysis_data/{SCENARIO}/test_pairs_graded.tsv"
    STOP_WORDS_PATH = "./data/common_words"
    DOC_JSON = f"./analysis_data/{SCENARIO}/doc.pickle"

    # %%
    args.n_features = 17
    kwargs = vars(args)
    
    data = load_data(args, SCENARIO, COLLECTION_PATH, QUERIES_PATH, TRAIN_PATH, DEV_PATH, TEST_PATH, STOP_WORDS_PATH, DOC_JSON, kwargs)

    # %%
    # Pointwise
    params_regr = Namespace(epochs=11, 
                        lr=1e-3,
                        batch_size=1,
                        metrics={"ndcg", "precision@05", "recall@05", "recall@100"})

    pointwise_regression_model = LTRModel(data.num_features)
    point_res = create_results(data, pointwise_regression_model, 
                            train_pointwise, 
                            pointwise_regression_model,
                            args.output+"analysis_pointwise_res.json",
                            params_regr)

    model_savepath = args.output+"analysis_pointwise_model"
    torch.save(pointwise_regression_model.state_dict(), model_savepath)
    print(f"Model saved at {model_savepath}")

    # %%
    # Pairwise
    params = Namespace(epochs=10, 
                        lr=1e-3,
                        batch_size=1,
                        metrics={"ndcg", "precision@05", "recall@05", "recall@100"})

    sped_up_pairwise_model = LTRModel(data.num_features)

    pw_res = create_results(data, sped_up_pairwise_model, 
                train_pairwise_spedup, 
                sped_up_pairwise_model,
                args.output+"analysis_pairwise_spedup_res.json",
                params)

    model_savepath = args.output+"analysis_pairwise_spedup_model"
    torch.save(sped_up_pairwise_model.state_dict(), model_savepath)
    print(f"Model saved at {model_savepath}")

    # %%
    # Listwise
    params_regr = Namespace(
        epochs=11, lr=1e-4, batch_size=1, metrics={"ndcg", "precision@05", "recall@05", "recall@100"}
    )

    results_path = args.output+"analysis_listwise_res.json"
    listwise_model = LTRModel(data.num_features)
    lw_res = create_results(
        data, listwise_model, train_listwise, listwise_model, results_path, params_regr
    )

    model_savepath = args.output+"analysis_listwise_model"
    torch.save(listwise_model.state_dict(), model_savepath)
    print(f"Model saved at {model_savepath}")
    
    if args.analyze_for_15_features:
        point_res_15, pw_res_15, lw_res_15 = analyze_for_15_features(args)
        analyze_17_vs_15(point_res, pw_res, lw_res, point_res_15, pw_res_15, lw_res_15)
