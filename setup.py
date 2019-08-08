from setuptools import setup
from setuptools import find_packages
from setuptools.command.install import install


class Word2vecModelDownload(install):
    """Downloads Word2vec models after installation if not already present."""

    _MODELS_URL = 'https://storage.googleapis.com/mat2vec/'
    _MODEL_FILES = [
        'pretrained_embeddings',
        'pretrained_embeddings.wv.vectors.npy',
        'pretrained_embeddings.trainables.syn1neg.npy',
    ]
    _DOWNLOAD_LOCATION = 'mat2vec/training/models'

    def run(self):
        install.run(self)


with open('README.md') as f:
    readme = f.read()

setup(
    name='mat2vec',
    version='0.2',
    description='Word2vec training and text processing code for Tshitoyan '
                'et al. Nature (2019).',
    long_description=readme,
    author='Authors of Tshitoyan et al. Nature (2019)',
    author_email='vahe.tshitoyan@gmail.com, jdagdelen@lbl.gov, '
                 'lweston@lbl.gov, ajain@lbl.gov',
    packages=find_packages(),
    cmdclass={
        'install': Word2vecModelDownload,
    },
    include_package_data=True,
)
