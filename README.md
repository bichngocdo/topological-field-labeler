# Topological Field Labeler

This repository contains code for the topological field labeler from the paper
[Parsers Know Best: German PP Attachment Revisited](https://aclanthology.org/2020.coling-main.185/)
published at COLING 2020.

## Usage

### Requirements
* Python 3.6
* Install dependencies in [requirements.txt](requirements.txt)

### Training
Run:
```shell
PYTHONPATH=`pwd` python seqlbl/model/experiment.py train --help
```
to see possible arguments.

For example, to train a model on the sample dataset, run:

```shell
PYTHONPATH=`pwd` python seqlbl/model/experiment.py train \
  --train_file data/sample/train.txt \
  --dev_file data/sample/dev.txt \
  --test_file data/sample/test.txt \
  --model_dir runs/sample \
  --word_dim 5 \
  --tag_dim 4 \
  --hidden_dim 7 \
  --num_lstms 2 \
  --num_mlps 1 \
  --mlp_dim 8 \
  --interval 1 \
  --num_train_buckets 2 \
  --num_dev_buckets 2 \
  --max_epoch 5
```

### Evaluation
Run:
```shell
PYTHONPATH=`pwd` python seqlbl/model/experiment.py eval --help
```
to see possible arguments.

For example, to evaluate a trained model on the sample dataset, run:

```shell
PYTHONPATH=`pwd` python seqlbl/model/experiment.py eval \
  --test_file data/sample/test.txt \
  --output_file runs/sample/result.txt \
  --model_dir runs/sample \
  --input_cols 0 1 2
```

### Tensorboard

```shell
tensorboard --logdir runs/sample/log
```


## Reproduction

All trained models contain:
* File `config.cfg` that records all parameters used to produce the model.
* Folder `log` records training and evaluation metrics, which can be viewed by `tensorboard`.
* See more information at [data](data) and [models](models).


## Citation

```bib
@inproceedings{do-rehbein-2020-parsers,
    title = "Parsers Know Best: {G}erman {PP} Attachment Revisited",
    author = "Do, Bich-Ngoc and Rehbein, Ines",
    editor = "Scott, Donia and Bel, Nuria and Zong, Chengqing",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2020.coling-main.185",
    doi = "10.18653/v1/2020.coling-main.185",
    pages = "2049--2061",
}
```