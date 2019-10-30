# A small python wrapper to plot some categories from the results of training and testing the span-based model of Ouchi et al. (2018)

All these files must be in folder span-based-srl/scripts (span-based-srl is the cloned git repository of Ouchi's model under https://github.com/hiroki13/span-based-srl)

### Requirements
run this file to set requirements ```install.sh```

## Run scatter plot for label embeddings
```
python plot.py path/to/param.epoch-0.pkl.gz -v -e embedding_name -l path/to/label_ids.txt (-cl labels/with/space or $(cat default_chosen_labels.txt)) (--save) (--output_dir output) (-rs 7 7) (--icr/--kernel)
```

## Run imshow for confusion matrix
```
python plot.py path/to/[gold_test_label] path/to/[predicted_label].prop -m -e embedding_name -l path/to/label_ids.txt (-cl labels/with/space or $(cat default_chosen_labels.txt)) (-rs 5 5) (--save) (--output_dir output) --error
```

## Run line plot for training progress on development set
```
python plot.py path/to/hist.txt -p -e embedding_name --valid(/--train)
```
