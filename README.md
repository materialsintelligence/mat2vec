### Supplementary Materials for "Unsupervised word embeddings capture latent knowledge from materials science literature", *Nature*  571, 95–98 (2019).
#### Tshitoyan, V., Dagdelen, J., Weston, L., Dunn, A., Rong, Z., Kononova, O., Persson, K. A., Ceder, G. and Jain, A. 
doi: [10.1038/s41586-019-1335-8](https://www.nature.com/articles/s41586-019-1335-8)

A view-only (no download) link to the paper: https://rdcu.be/bItqk

For those interested in the *ab initio* thermoelectric data, [see below](#thermoelectric-data)

### Set up

1. Make sure you have `python3.6` and the `pip` module installed. 
We recommend using [conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
1. Navigate to the root folder of this repository (the same folder that contains this README file)
and run `pip install --ignore-installed -r requirements.txt`. Note: If you are using a conda env and any packages fail to compile during this step, you may need to first install those packages separately with `conda install package_name`. 
1. Wait for all the requirements to be downloaded and installed.
1. Run `python setup.py install` to install this module. This will also download the Word2vec model files.
If the download fails, manually download the [model](https://storage.googleapis.com/mat2vec/pretrained_embeddings), 
[word embeddings](https://storage.googleapis.com/mat2vec/pretrained_embeddings.wv.vectors.npy) and 
[output embeddings](https://storage.googleapis.com/mat2vec/pretrained_embeddings.trainables.syn1neg.npy) and put them in mat2vec/training/models.
1. Finalize your chemdataextractor installation by executing ``cde data download`` (You may need to restart your virtual environment for the cde command line interface to be found).
1. You are ready to go!

#### Processing

Example python usage:

```python
from mat2vec.processing import MaterialsTextProcessor
text_processor = MaterialsTextProcessor()
text_processor.process("LiCoO2 is a battery cathode material.")
```
> (['CoLiO2', 'is', 'a', 'battery', 'cathode', 'material', '.'], [('LiCoO2', 'CoLiO2')])

For the various methods and options see the docstrings in the code.

#### Pretrained Embeddings

Load and query for similar words and phrases:
```python
from gensim.models import Word2Vec
w2v_model = Word2Vec.load("mat2vec/training/models/pretrained_embeddings")
w2v_model.wv.most_similar("thermoelectric")
```
> [('thermoelectrics', 0.8435688018798828), ('thermoelectric_properties', 0.8339033126831055), ('thermoelectric_power_generation', 0.7931368350982666), ('thermoelectric_figure_of_merit', 0.7916493415832
52), ('seebeck_coefficient', 0.7753845453262329), ('thermoelectric_generators', 0.7641351819038391), ('figure_of_merit_ZT', 0.7587921023368835), ('thermoelectricity', 0.7515754699707031), ('Bi2Te3', 0
.7480161190032959), ('thermoelectric_modules', 0.7434879541397095)]

Phrases can be queried with underscores:
```python
w2v_model.wv.most_similar("band_gap", topn=5)
```
> [('bandgap', 0.934801459312439), ('band_-_gap', 0.933477520942688), ('band_gaps', 0.8606899380683899), ('direct_band_gap', 0.8511275053024292), ('bandgaps', 0.818678617477417)]

Analogies:
```python
# helium is to He as ___ is to Fe? 
w2v_model.wv.most_similar(
    positive=["helium", "Fe"], 
    negative=["He"], topn=1)
```
> [('iron', 0.7700884938240051)]

Material formulae need to be normalized before analogies:
```python
# "GaAs" is not normalized
w2v_model.wv.most_similar(
    positive=["cubic", "CdSe"], 
    negative=["GaAs"], topn=1)
```
> KeyError: "word 'GaAs' not in vocabulary"
```python
from mat2vec.processing import MaterialsTextProcessor
text_processor = MaterialsTextProcessor()
w2v_model.wv.most_similar(
    positive=["cubic", text_processor.normalized_formula("CdSe")], 
    negative=[text_processor.normalized_formula("GaAs")], topn=1)
```
> [('hexagonal', 0.6162797212600708)]

Keep in mind that words should also be processed before queries.
Most of the time this is as simple as lowercasing, however, it is the safest
to use the `process()` method of `mat2vec.processing.MaterialsTextProcessor`.

#### Training

To run an example training, navigate to *mat2vec/training/* and run

```shell
python phrase2vec.py --corpus=data/corpus_example --model_name=model_example
```

from the terminal. It should run an example training and save the files in *models*
and *tmp* folders. It should take a few seconds since the example corpus has only 5 abstracts.

For more options, run

```shell
python phrase2vec.py --help
```

#### Thermoelectric Datasets

Dataset 1: Computed entries (**we provide this dataset in this repository**)

You can find the condensed thermoelectric CRTA data in the `thermoelectric_data` directory.

This is a dataset of power factors (~48k) originally taken from the materials project in which we condense seebeck/conductivity tensors into one number representing a “best case power factor” in a polycrystal -  we call this PF (T-cut, c-cut) avg. This dataset really only deals in these DFT + BoltzTrap computed power factors under quite a few assumptions (const. relaxation time approximation, polycrystal averaging, Temperature and carrier concentration cutoffs, “optimistic” behaviors, etc.). It does not involve zTs, as zTs require a lattice thermal conductivity (not easily calculable in high throughput as opposed to DFT+BoltzTrap). So this dataset is essentially lower fidelity power factor data only.
In general, we used the following procedure (which you can also find in the nature article’s methods section):

Each materials structure optimization and band structure calculation was performed with density functional theory (DFT)
using the projector augmented wave (PAW)29 pseudopotentials and the Perdew– Burke–Ernzerhof (PBE)30 generalized-gradient approximation (GGA), imple- mented in the Vienna Ab initio Simulation Package (VASP)31,32. A +U correction was applied to transition metal oxides16. Seebeck coefficient (S) and electrical conductivity (σ) were calculated using the BoltzTraP package33 using a constant relaxation time of 10−14 s at simulated temperatures between 300 K and 1,300 K and for carrier concentrations (doping) between 1016 cm−3 and 1022 cm−3. A 48,770-material subset of the calculations was taken from a pre- vious work16; the remaining calculations were performed in this work using the software atomate34. All calculations used the pymatgen28 Python library within the FireWorks35 workflow management framework. To more realistically eval- uate the thermoelectric potential of a candidate material, we devised a simple strategy to condense the complex behaviour of the S and σ tensors into a single power factor metric. For each semiconductor type η ∈ {n, p}, temperature T, and doping level c, the S and σ tensors were averaged over the three crystallo- graphic directions, and the average power factor, PFavg, was computed. PFavg is a crude estimation of the polycystalline power factor from the power factor of a perfect single crystal. To account for the complex behaviour of S and σ with T, c, and η, we then took the maximum average power factor over T, c, and η constrained to a maximum cutoff temperature Tcut and maximum cutoff doping ccut.Formally,thisisPFTcut,ccut ≡maxPF(η,T,c) such that T≤T ,c≤c .We
chose Tcut = 600 K and ccut = 10^20 cm^-3  because these values resulted in better  orrespondence with the experimental dataset than more optimistic values, owing to the limitations of the constant relaxation time approximation. The  resulting power factor, PF600 K, 10 20, is equated with ‘computed power factor’ in this study. To rank materials according to experimental power factors (or zT), we used the maximum value for a given stoichiometry across all experimental conditions present in the dataset from Gaultois et al.19

Some of the formatting here might be messed up so you would be better off just looking at the original publication.

Dataset 2: Experimental entries (**we do not provide this dataset in this repository**). 

The other is a dataset aggregated from experimental zT entries taken from this publication: https://pubs.acs.org/doi/10.1021/cm400893e, known as the UCSB thermoelectrics dataset. You can download a reduced version of it here: https://figshare.com/articles/dataset/ucsb_thermoelectrics/14746212. This is the dataset which contains the 83 entries mentioned in the paper.



### Related Work

- Weston, L., Tshitoyan, V., Dagdelen, J., Kononova, O., Persson, K. A., Ceder, G. and Jain, A. Named Entity Recognition and Normalization Applied to Large-Scale Information Extraction from the Materials Science Literature, [ChemRxiv. Preprint.](https://chemrxiv.org/articles/Named_Entity_Recognition_and_Normalization_Applied_to_Large-Scale_Information_Extraction_from_the_Materials_Science_Literature/8226068) (2019).

### Issues?

You can either report an issue on github or contact one of us directly. 
Try [vahe.tshitoyan@gmail.com](mailto:vahe.tshitoyan@gmail.com), 
[jdagdelen@berkeley.edu](mailto:jdagdelen@berkeley.edu), 
[lweston@lbl.gov](mailto:lweston@lbl.gov) or 
[ajain@lbl.gov](mailto:ajain@lbl.gov).
