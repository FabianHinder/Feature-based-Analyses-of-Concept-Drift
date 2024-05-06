# Feature-wise Analysis of Concept Drift

Experimental code of journal paper. [Paper](TODO) 

## Abstract

Feature selection is one of the most relevant preprocessing and analysis techniques in machine learning. It can dramatically increase the performance of learning algorithms and also provide relevant information on the data. In the scenario of online and stream learning, concept drift, i.e., changes of the underlying distribution over time, can cause significant problems for learning models and data analysis. While there do exist feature selection methods for online learning, none of the methods targets feature selection for drift detection, i.e., to the challenge to increase the performance of drift detectors and to analyze the drift rather than increase model accuracy. This challenge is particularly relevant for common unsupervised scenarios. In this work, we study feature selection for concept drift detection and drift monitoring. We develop a formal definition for a feature-wise notion of drift that allows semantic interpretation. We derive an efficient algorithm by reducing the problem to classical feature selection and analyze the applicability of our approach to feature selection for drift detection on a theoretical level. Finally, we empirically show the relevance of our considerations on several benchmarks.

**Keywords:** Feature Selection, Concept Drift, Explaining Streaming Data, Explainable AI 

## Requirements

* Python 
* Numpy, SciPy, Pandas, Matplotlib
* scikit-learn
* BorutaPy

## Usage

To run the experiments, there are three stages 1. create the datasets (`--make`) which creates the datasets and stores them in a local directory, 2. splits the experimental setups in several chunks (`--setup #n`) for parallel processing on different devices, and 3. running the experiments (`--run_exp #n`) which runs the chunk as indicated by the command line attribute.

## Cite

Cite our Paper
```
TODO
```

## License

This code has a MIT license.
