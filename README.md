# A show of ELMo, BERT, Flair on **Semantic Role Labeling** task

A small research of interaction between span-based Model from [Ouchi et al. (2018)](https://github.com/hiroki13/span-based-srl) and types of contextualized embeddings (focused on ***ELMo***, ***BERT*** and ***stacked Embeddings between Flair and BERT***) with ***SENNA*** Embeddings as an example of baseline. All datas used come from [CoNLL 2012](http://conll.cemantix.org/2012/task-description.html).

## Structure

```bash
├── scripts
│   ├── create_embedding - create the contextualized embeddings used in experiments
│   ├── plot - create 3 types of plots for results in experiment (scatter plot, imshow and line plot)
│   ├── prepare - prepare the raw datas for embeddings training and gold standards for evaluation
└── results - Results for each types of embeddings including visualizations, evaluations for results at different epochs for test set, training report (output from this model)
│   ├── bert
│   ├── elmo
│   ├── senna
│   └── stacked_bert-flair
```


**Note**: all train reports for *BERT* and *Stacked* have the same name types of *ELMo* as Ouchi only offers 2 types of trained embeddings: non-contextualised and ELMo (on behalf of contextualised) embeddings.

## Report
- Report in German written for the results created in October 2019.
