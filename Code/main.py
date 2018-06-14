"""
BSCS14002           Mohammad Asim Iqbal
BSCS14008           Mahad Saleem
BSCS14027           Muhammad Murtaza Azam Khan

ARTIFICIAL INTELLIGENCE
FINAL PROJECT
SPEAKER RECOGNITION USING MEL FREQUENCY CEPSTRAL COEFFICIENTS
Main
"""


from __future__ import division
from train import training
from test import testing

SpeakerN = 10
filtbankN = 12

codebooks = training(filtbankN)

testing(SpeakerN, codebooks, filtbankN)