# srl-context-emb

A small research of interaction between span-based Model from [Ouchi et al. (2018)](https://github.com/hiroki13/span-based-srl) and types of contextualized embeddings (focused on ELMo, BERT and stacked Embeddings between Flair and BERT) with SENNA Embeddings as an example of baseline. All datas used come from [CoNLL 2012](http://conll.cemantix.org/2012/task-description.html).

## Folder *scripts*
There are 3 folders in the folder ***scripts***:

### *create_embedding*
This file provides some codes to create the contextualized embeddings I used for my experiment.

Read USAGE.md to run these

### *plot*
This file provides scripts to create 3 types of plots for results in experiment (scatter plot, imshow and line plot).

### *prepare*
This file provides codes to prepare the raw datas for embeddings training and gold standards for evaluation.

## Folder *results*
Results for each types of embeddings including visualizations, evaluations for results at different epochs for test set, training report (output from this model).
**Note**: all train reports for *BERT* and *Stacked* have the same name types of *ELMo* as Ouchi only offers 2 types of trained embeddings: non-contextualised and ELMo (on behalf of contextualised) embeddings.
