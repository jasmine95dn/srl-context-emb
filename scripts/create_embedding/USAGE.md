# Create embeddings for data set from pretrained embeddings

This script is based on the same format of the output of ```allennlp elmo <input> <output.hdf5> --all``` to train BERT Embedding and Stacked Embedding for experiments with span-based model of Ouchi et al. 2018

### Requirements
* [python3](https://www.python.org/downloads/)
* [allennlp](https://github.com/allenai/allennlp/tree/v0.6.1)
* [flair](https://github.com/zalandoresearch/flair)
* [h5py](https://www.h5py.org/)

### Usage
* Train ELMo Embedding: 
```allennlp elmo <sentences.txt> <output.hdf5> --all```

* Train BERT Embedding:
```python bert_flair_emb.py -f <sentences.txt> ```

* Train Stacked Flair-Embedding:
```python bert_flair_emb.py -f <sentences.txt> -s```

*sentences.txt*: Data set in form of each line a sentence (all tokens including punctuations are separated through space/tab)

More details: ```python bert_flair_emb.py -h```
