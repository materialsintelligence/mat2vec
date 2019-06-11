from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='mat2vec',
    version='0.2',
    description='Word2vec training and text processing code for Tshitoyan et al. Nature (2019).',
    long_description=readme,
    author='Authors of Tshitoyan et al. Nature (2019)',
    author_email='vahe.tshitoyan@gmail.com, jdagdelen@lbl.gov, lweston@lbl.gov',
    packages=find_packages(),
    include_package_data=True
)
