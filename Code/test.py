"""
BSCS14002           Mohammad Asim Iqbal
BSCS14008           Mahad Saleem
BSCS14027           Muhammad Murtaza Azam Khan

ARTIFICIAL INTELLIGENCE
FINAL PROJECT
SPEAKER RECOGNITION USING MEL FREQUENCY CEPSTRAL COEFFICIENTS
Test
"""


from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from LBG import EUCDIST as EUDistance
#from MEL_COEFFICIENTS import MFCC
from python_speech_features import mfcc as MFCC

dir_test = 'D:/_python/SpeakerRecognition/3/Data/test'

def distance_min(feature, codebook):
    distmini = np.inf
    speakernum = 0
    for l in range(np.shape(codebook)[0]):
        Dis = EUDistance(feature, codebook[l,:,:])
        dista = np.sum(np.min(Dis, axis = 1))/(np.shape(Dis)[0])
        if dista < distmini:
            speakernum = l
            distmini = dista
            
    return speakernum

def testing(SpeakerN, codebooks, filterbankN):
    file = str()
    identified = 0
    for i in range(SpeakerN):
        file = '/Speaker' + str(i+1) + '.wav'
        (fs,sig) = read(dir_test + file)
        print ('\n\nTesting the features of speaker ', str(i+1))
#        melcoef = MFCC(fs,sig,filterbankN)#--------TESTING
        melcoef = MFCC ( sig , fs )#--------TESTING
        melcoef = np.transpose ( melcoef )#--------TESTING
       
        sp_iden = distance_min(melcoef, codebooks)
            
        if i == sp_iden:
            identified += 1
        
        print ('Given Speaker: ', (i+1), '  (Test data)\nMatched with Speaker: ', (sp_iden+1), '  (Training data)')
    
    Accuracy = (identified/SpeakerN)*100
    print ('\n=> Accuracy: ', Accuracy, '%')