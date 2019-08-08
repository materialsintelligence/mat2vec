This document describes how to train a word2vec model on PubMed abstracts.

***Most of this code was copied from
[mat2vec repository](https://github.com/materialsintelligence/mat2vec)***


## Installation

Run the following to install the module:
```
cd foundation/src/python/third_party/mat2vec
python setup.py install
```

## Data Processing

Download PubMed abstracts:
```
wget -P data -m ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
```

Convert the XMLs to a file containing one document per line:
```
nohup python mat2vec/processing/pubmed_to_lines.py \
    --input_folder data/ftp.ncbi.nlm.nih.gov/pubmed/baseline \
    --output_folder data/pubmed/word2vec \
    --n_processes 6 \
    > processing.log 2>&1 &
```

Merge the text files into one:
```
cat data/pubmed/word2vec/* > data/pubmed/word2vec/corpus.txt
```

After this preprocessing, you should end up with 32M documents, 4.5B tokens,
and 19M unique words.

## Training

We can now train a word2vec on the PubMed corpus:
```
nohup python mat2vec/training/phrase2vec.py \
    --corpus=data/pubmed/word2vec/corpus.txt \
    --output_folder=data/output \
    > training.log 2>&1 &
```

It should take 12 hours to train on a 8-core machine for 4 epochs (i.e.,
iterating over the training data for 4 times).
The output of this training will result in 2.2M unique words, each paired with
a 200-dimension vector.

## Web Server
For a web server of the trained word2vec model, please refer to 
`foundatation/src/python/eai/word2vec/server`

