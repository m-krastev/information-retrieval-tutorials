[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Sspt6QAG)
# Assignment 2 - Part 2: LTR from Interactions <a class="anchor" id="toptop"></a>

## Introduction
Welcome to the second part of the LTR assignment. In the previous part, you experimented with offline learning to rank. This assignment is on learning to rank from interactions. You will learn to train an unbiased model and to jointly estimate propensities using DLA (dual learning algorithm).

**Learning Goals**
- Simulate document clicks
- Implement biased and unbiased counterfactual learning to rank (LTR) and dual learning algorithms
- Evaluate and compare the methods

## Guidelines

### How to proceed?
We have prepared a notebook: `hw2-2.ipynb` including the detailed guidelines of where to start and how to proceed with this part of the assignment. Alternatively, you can use `hw2-2.py` if you prefer a Python script that can easily be run from the command line.
The two files are equivalent.

You can find all the code of this assignment inside the `ltr` package. The structure of the `ltr` package is shown below, containing various modules that you need to implement. For the files that require changes, a :pencil2: icon is added after the file name. This icon is followed by the points you will receive if all unit tests related to this file pass. 

📦ltr \
 ┣ 📜dataset.py :pencil2: \
 ┣ 📜eval.py\
 ┣ 📜logging_policy.py\
 ┣ 📜loss.py :pencil2: \
 ┣ 📜model.py :pencil2: \
 ┣ 📜train.py :pencil2: \
 ┗ 📜utils.py


Make sure to __commit__ and __push__ the results files after training and running your models.

**NOTICE THAT YOU NEED TO PUT YOUR IMPLEMENTATION BETWEEN `BEGIN \ END SOLUTION` TAGS!** All of the functions that need to be implemented in modules directory files have a #TODO tag at the top of their definition. As always, it's fine to create additional code/functions/classes/etc outside of these tags, but make sure not to change the behavior of the existing API.

Make sure to __commit__ and __push__ the `.json` results files after training and running your models. For the analysis, you can base your observations on the reported metrics for the _test_ splits of the datasets that you train your models on.

By completing all required implementation and passing all the unit tests you are eligible for full marks on the implementation part, which can be computed by summing the points from passing each test found in [autograding.json](.github/classroom/autograding.json). The tests also depend on the results from each of your model saved in the [outputs](./outputs/) folder.

#### Analysis Part

This assignment also consists of an analysis part that requires you to conduct experiments investigating the impact of specific components. The details are in [analysis.md](analysis.md). 

### Points
To score 100% on this assignment, you need to complete both the implementation component and the analysis component. More specifically,

* The implementation component is worth 67% of your assignment2-part2 grade.
* The analysis component is worth 33% of your assignment2-part2 grade.
* We suggest completing the implementation component first, and then moving on to the analysis component. If your implementation does not give reasonable results, it will be impossible for your analysis to be reasonable and be awarded points. You can look at the autograding output to confirm you've successfully completed the implementation component, which also suggests that your implementation should provide reasonable results for you to analyze.
* Parts 1 and 2 are each 50% of your assignment 2 grade.


**Important Remarks**:

Please note that you **SHOULD NOT** change params, seed(always=42), or any PATH variable given in the notebook for training LTR models with various loss functions, otherwise, you might lose points. We do NOT check the notebook; we ask this since we evaluate your results with the given parameters.

## Table of Contents

Table of contents:

 - _Chapter 1: Offline LTR (previous assignment)_
 - Chapter 2: Counterfactual LTR
    - Section 1: Dataset and utils
    - Section 2: Biased ListNet
    - Section 3: Unbiased ListNet
    - Section 4: Propensity estimation
    - Section 5: Evaluation
