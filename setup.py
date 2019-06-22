import math
import logging
import os
import requests

from setuptools import setup
from setuptools import find_packages
from setuptools.command.install import install

from tqdm import tqdm
import urllib.parse

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",
                    level=logging.INFO)


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
        for model_file in self._MODEL_FILES:
            file_url = urllib.parse.urljoin(self._MODELS_URL, model_file)
            final_location = os.path.join(self._DOWNLOAD_LOCATION, model_file)
            r = requests.get(file_url, stream=True)

            total_size = int(r.headers.get('content-length', 0))
            if self._file_exists_correct_size(model_file, total_size):
                logging.info("{} already present, skipping download.".format(model_file))
                continue  # If the file is already there, skip downloading it.

            logging.info('Starting download for {}'.format(model_file))
            block_size, wrote = 1024, 0
            with open(final_location, 'wb') as downloaded_file:
                for data in tqdm(r.iter_content(block_size),
                                 total=math.ceil(total_size // block_size),
                                 unit='KB',
                                 unit_scale=True):
                    wrote = wrote + len(data)
                    downloaded_file.write(data)
            if total_size != 0 and wrote != total_size:
                logging.ERROR(
                    "Something went wrong during the download "
                    "of {}, the size of the file is not correct. "
                    "Please retry.".format(model_file))
            else:
                logging.info("{} successfully downloaded.".format(model_file))
        install.run(self)

    def _file_exists_correct_size(self, filename, expected_size):
        """Checks if the file exists in the download location and has the correct size.

        Args:
            filename: The name of the file in the download location.
            expected_size: The expected size in bytes.

        Returns:
            True if the file exists and has the expected size, False otherwise.
        """
        full_file_path = os.path.join(self._DOWNLOAD_LOCATION, filename)
        if (not os.path.exists(full_file_path) or
                os.path.getsize(full_file_path) != expected_size):
            return False
        return True


with open('README.md') as f:
    readme = f.read()

setup(
    name='mat2vec',
    version='0.2',
    description='Word2vec training and text processing code for Tshitoyan et al. Nature (2019).',
    long_description=readme,
    author='Authors of Tshitoyan et al. Nature (2019)',
    author_email='vahe.tshitoyan@gmail.com, jdagdelen@lbl.gov, lweston@lbl.gov, ajain@lbl.gov',
    packages=find_packages(),
    cmdclass={
        'install': Word2vecModelDownload,
    },
    include_package_data=True,
)
