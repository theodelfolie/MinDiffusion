# -*- coding: utf-8 -*-

# Python System imports
from os import path

# Third-party imports
from setuptools import setup, find_packages

NAME = "minDiffusion"
DESCRIPTION = "A minimal, educational implementation of a *Denoising Diffusion Probabilistic Model (DDPM)* in PyTorch."

URL = 'https://github.com/theodelfolie/MinDiffusion'
EMAIL = 'tdelfolie5@gmail.com'
AUTHOR = 'Théo Delfolie'
REQUIRES_PYTHON = '>=3.10.0'

CLASSIFIERS = [
    # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.10',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
]

with open(path.join(path.dirname(__file__), "requirements.txt")) as req:
    REQUIREMENTS = list(line.strip() for line in req)

setup(
    name=NAME,
    version="0.1.0",
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    classifiers=CLASSIFIERS,
    license='MIT License',
)
