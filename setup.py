# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="delab-trees",
    version="0.4.2",
    description="a library to analyse reply trees in forums and social media",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/juliandehne/delab-trees",
    author="Julian Dehne",
    author_email="julian.dehne@gmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(),
    package_data={'delab_trees.data': ['dataset_reddit_no_text.pkl', 'dataset_twitter_no_text.pkl']},
    include_package_data=True,
    install_requires=[
        "numpy==1.22.3",
        "pandas~=1.5.3",
        "networkx~=2.8",
        "scikit-learn",
        "keras",
        "matplotlib~=3.4.3",
        "tensorflow",
        "tqdm",
        "pytest==7.1.2",
        "jupyter"
    ],
    python_requires='>=3.9, <3.10',  # Allows all Python 3.9.x versions
    setup_requires=[
        "setuptools>=64"
    ],
)
