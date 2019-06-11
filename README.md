# mat2vec
Supplementary code and data for Tshitoyan, V., Dagdelen, J., Weston, L., Dunn, A., Rong, Z., Kononova, O., Persson, K. A., 
Ceder, G. and Jain, A. "Unsupervised word embeddings capture latent knowledge from materials science literature", 
*Nature* (2019)

### Set up

1. Make sure you have `python3.6` and the `pip` module installed. 
We recommend using [conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
1. Navigate to the root folder (the same folder that contains this README file)
and run `pip install -r requirements.txt`
1. Wait for all the requirements to be downloaded and installed.
1. run `pip install .` to install this module.
1. You are ready to go!

#### Processing

Example usage in a python console:

```
>>> from mat2vec.processing import process
>>> text_processing = process.MaterailsTextProcessing()
>>> text_processing.process("LiCoO2 is a bettery cathode material.")
(['CoLiO2', 'is', 'a', 'bettery', 'cathode', 'material', '.'], [('LiCoO2', 'CoLiO2')])
```

For the various methods and options see the docstrings in the code.

#### Training
To run an example training, navigate to *mat2vec/training/* and run

```
python phrase2vec.py --corpus=data/corpus_example --model_name=model_exampe
```

from the terminal. It should run an example training and save the files in *models*
and *tmp* folders. It should take a few seconds since the example corpus has only 5 abstracts.

For more options, run

```
python phrase2vec.py --help
```

#### Pretrained Embeddings

Load and query for similar words and phrases

```
>>> from gensim.models import Word2Vec
>>> w2v_model = Word2Vec.load("mat2vec/training/models/pretrained_embeddings")
>>> w2v_model.wv.most_similar("thermoelectric")
[('thermoelectrics', 0.8435688018798828), ('thermoelectric_properties', 0.8339033126831055), ('thermoelectric_power_generation', 0.7931368350982666), ('thermoelectric_figure_of_merit', 0.7916493415832
52), ('seebeck_coefficient', 0.7753845453262329), ('thermoelectric_generators', 0.7641351819038391), ('figure_of_merit_ZT', 0.7587921023368835), ('thermoelectricity', 0.7515754699707031), ('Bi2Te3', 0
.7480161190032959), ('thermoelectric_modules', 0.7434879541397095)]
```
