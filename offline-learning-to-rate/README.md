[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/LrOAdpY3)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-7f7980b617ed060a017424585567c406b6ee15c891e84e1186181d67ecf80aa0.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=14203154)
# Assignment 2 - Part 1: Learning to Rank <a class="anchor" id="toptop"></a>

## Introduction
Welcome to the first part of the LTR assignment, which deals directly with learning to rank (LTR). In part 1, offline LTR, you will learn how to implement methods from the three approaches associated with learning to rank: pointwise, pairwise and listwise. 

**Learning Goals**:
  - Extract features from (query, document) pairs 
  - Implement pointwise, pairwise and listwise algorithms for learning-to-rank
  - Train and evaluate learning-to-rank methods

## Guidelines

### How to proceed?
#### Implementation Part
We have prepared a notebook: `hw2.ipynb` including the detailed guidelines of where to start and how to proceed with this part of the assignment. Alternatively, you can use `hw2.py` if you prefer a Python script that can easily be run from the command line.
The two files are equivalent.

You can find all the code of this assignment inside the `ltr` package. The structure of the `ltr` package is shown below, containing various modules that you need to implement. For the files that require changes, a :pencil2: icon is added after the file name. This icon is followed by the points you will receive if all unit tests related to this file pass. 

**NOTICE THAT YOU NEED TO PUT YOUR IMPLEMENTATION BETWEEN `BEGIN \ END SOLUTION` TAGS!** All of the functions that need to be implemented in modules directory files have a #TODO tag at the top of their definition.

**Structure of the ``ltr`` package:**

📦ltr\
 ┣ 📜dataset.py :pencil2: \
 ┣ 📜eval.py\
 ┣ 📜loss.py :pencil2: \
 ┣ 📜model.py :pencil2: \
 ┣ 📜train.py :pencil2: \
 ┗ 📜utils.py

In the `ltr.dataset.FeatureExtraction.extract()` method, you are required to compute various features for a (query, document) pair. The list of features and its definitions can be found in the following table:

| Feature                | Definition |
|------------------------|------------|
| bm25| [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) score. Parameters: k1 = 1.5, b = 0.75 |
|query_term_coverage| Number of query terms in the document |
|query_term_coverage_ratio| Ratio of # query terms in the document to #query terms in the query  |
|stream_length| Length of document |
|idf| Sum of document frequencies. idf_smoothing = 0.5 , formula: Log((N+1)/(df[term]+smoothing))| 
|sum_stream_length_normalized_tf| Sum over the ratios of each term to document length |
|min_stream_length_normalized_tf| Min over the ratios of each term to document length |
|max_stream_length_normalized_tf| Max over the ratios of each term to document length |
|mean_stream_length_normalized_tf| Mean over the ratios of each term to document length|
|var_stream_length_normalized_tf| Variance over the ratios of each term to document length |
|sum_tfidf| Sum of tf-idf|
|min_tfidf| Min of tf-idf|
|max_tfidf| Max of tf-idf |
|mean_tfidf| Mean of tf-idf |
|var_tfidf| Variance of tf-idf |

By completing all required implementation you can achieve full marks on the implementation part, which can be computed by summing the points from passing each test found in [autograding.json](.github/classroom/autograding.json). The tests also depend on the results from each of your models saved in the [outputs](./outputs/) folder.

- The expected files include: 
  - [`outputs/pointwise_res.json`](./outputs/pointwise_res.json)
  - [`outputs/pairwise_res.json`](./outputs/pairwise_res.json)
  - [`outputs/listwise_res.json`](./outputs/listwise_res.json)
  - [`outputs/new_features_res.json`](./outputs/new_features_res.json)

Make sure to __commit__ and __push__ the `.json` results files after training and running your models. For the analysis, you can base your observations on the reported metrics for the _test_ splits of the datasets that you train your models on.  

#### Analysis Part

This assignment also consists of an analysis part where you need to choose an application scenario to work on. More information can be found at [analysis.md](analysis.md). 


### Points
To score 100% on this assignment, you need to complete both the implementation component and the analysis component. More specifically,

* The implementation component is worth 67% of your assignment2-part1 grade. If all of the autograding tests pass and your implementation is correct, you receive all 67%.
* The analysis component is worth 33% of your assignment2-part1 grade.
* We suggest completing the implementation component first, and then moving on to the analysis component. If your implementation does not give reasonable results, it will be impossible for your analysis to be reasonable and be awarded points. You can look at the autograding output to confirm you've successfully completed the implementation component, which also tells you that your implementation should provide reasonable results for you to analyze.


**Important Remarks**:

Please note that you **SHOULD NOT** change params, seed(always=42), or any PATH variable given in the notebook for training LTR models with various loss functions, otherwise, you might lose points. We do NOT check the notebook; we ask this since we evaluate your results with the given parameters.

---
**Recommended Reading**:
  - Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. Learning to rank using gradient descent. InProceedings of the 22nd international conference on Machine learning, pages 89–96, 2005.
  - Christopher J Burges, Robert Ragno, and Quoc V Le. Learning to rank with nonsmooth cost functions. In Advances in neural information processing systems, pages 193–200, 2007
  - (Sections 1, 2 and 4) Christopher JC Burges. From ranknet to lambdarank to lambdamart: An overview. Learning, 11(23-581):81, 2010
  

Additional Resources: 
- This assignment requires knowledge of [PyTorch](https://pytorch.org/). If you are unfamiliar with PyTorch, you can go over [these series of tutorials](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

