# Online Machine Teaching under Learner Uncertainty

This repository contains the MATLAB code for the paper titled "Online Machine Teaching under Learner's Uncertainty: Gradient Descent Learners of a Quadratic Loss," authored by Belen Martin-Urcelay, Christopher J. Rozell, and Matthieu R. Bloch, 2024.

## Abstract

Machine teaching is a subfield of active learning that aims to develop example selection policies that teach a known target concept to a machine learning algorithm with the least number of examples possible. Online machine teaching algorithms usually assume that the entity selecting the examples, the teacher, has absolute knowledge about the machine learning algorithm's, the learner's, status, making its implementation often unrealistic. We relax the omniscience assumption by proving that efficient machine teaching is possible even when the teacher is uncertain about the learner initialization. Our analysis holds for learners that perform gradient descent of a quadratic loss to learn a linear classifier. We propose an online algorithm in which the teacher simultaneously learns about the learner state while teaching the learner. We theoretically and empirically show that the learner’s mean square error decreases exponentially with the number of examples, thus achieving a performance similar to the omniscient case. We successfully apply our findings to the cross-lingual sentiment analysis problem.

## Requirements

- Matlab R2022b with Optimization Toolbox version 9.4
- Monolingual word embeddings curated by Artetxe et al. (2016) that can be downloaded here: http://ixa2.si.ehu.es/martetxe/vecmap/es.emb.txt.gz

## Synthetic

The *synthetic* folder contains the code to implement the online machine teaching algorithms to teach a synthetic 2D classifier.

1. "synth2.m": Runs the proposed machine teaching algorithm without and with noisy feedback (i.e., Algorithm 1 and 2), as well as the baselines: the omniscient teacher's algorithm and random sample selection (SGD). 
   1. It calls "maximizer_MT.m" to select the example label pair than maximize T (i.e. greedily minimize the MSE).
   2. The results are stored in "syn2_results.mat".
   3. Update the parameter 'sigma' to change the noise level of the feedback.
   4. This code produces Figures 2a, 2b, 2c and 2d; as well as red and blue curves in Figures 2e and 2f.

2. "syn2_LearningForOmniscience.m": Runs the state of the art learning for omniscience (LfO) algorithm.
   1. It calls "probing_LfO.m" during the probing phase, to select the example-label pair that greedily minimize the uncertainty (norm of the covariance) about the learner.
   2. Update the parameter 'delta' to change uncertainty threshold before moving on to the second phase, the teaching phase, in which the teacher proceeds as if it were omniscient.
   3. This code produces doted lines in Figures 2e and 2f.
3. "synth2_imperfectOrthonormality": Analyzes the performance when the mapping from the teacher to the learner's space suffers from deviations from perfect orthonormality.


## Sentiment Analysis

The *sentiment analysis* folder contain the cross-lingual online machine teaching algorithms to teach a positive vs. negative word classifier. The teacher knows this classifier in Spanish, but the learner exists on a Italian embedding. The goal is to teach the corresponding classifier for Italian words.

1. "initialization.mat": Dataset with normalized word embeddings and classifiers.
   1. We use pre-existing monolingual word embeddings curated by Artetxe et al. (2016) that can be downloaded here: http://ixa2.si.ehu.es/martetxe/vecmap/es.emb.txt.gz
   2. We use the positive and negative word lists found here: https://www.kaggle.com/datasets/rtatman/sentiment-lexicons-for-81-languages
   3. "unilanguage_embedding.m" normalizes the word embeddings and finds the corresponding positive vs. negative classifiers in Spanish and Italian.
2. "predefinedDictionaryMT_estoit.m": Teaching algorithm for cross lingual machine teaching from Spanish to Italian of positive vs. negative word classifier. We recreate the setting without feedback from learner (SMTL) as well with noisy feedback (SMTL-F).
   1. It uses "maximizer_predefinedPool.m" to select the example-label pair that maximize T, i.e., minimize the MSE from one iteration to the next.
   2. The resulting solutions are saved as "predefinedPoolMT_results.mat"
   3. It replicates blue and red curves in Figures 3 and 4.
3. "random_it.m": Random baseline
   1. In each iteration, it randomly samples among 10,000 most common words. This word together with its true label are passed to the learner.
   2. It replicates green curves in Figure 2.
4. "rescalablePoolMT_estoit.m": Extension to rescalable pool based setting of teaching algorithm. 
   1. The resulting solutions are saved as "rescalablePoolMT_results.mat"
   2. This algorithm is described further in Section 2 of the Supplemental Material 

## Contact Information 

Email: burcelay3@gatech.edu
