"""
BSCS14002           Mohammad Asim Iqbal
BSCS14008           Mahad Saleem
BSCS14027           Muhammad Murtaza Azam Khan

ARTIFICIAL INTELLIGENCE
FINAL PROJECT
SPEAKER RECOGNITION USING MEL FREQUENCY CEPSTRAL COEFFICIENTS
Train
"""


from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from LBG import LBG as lbg
#from MEL_COEFFICIENTS import MFCC
from python_speech_features import mfcc as MFCC
import matplotlib.pyplot as plt

def training(filtbankN):
    Centroids = 16
    SpeakerN = 10
    dir_train = 'D:/_python/SpeakerRecognition/3/Data/train'
    file = str()
    codebooks_org = np.empty((SpeakerN,filtbankN,Centroids))
    
    for i in range(SpeakerN):
        file = '/Speaker' + str(i+1) + '.wav'
        (fs,sig) = read(dir_train + file)
        print ('Training the features of speaker ',str(i+1))
#        melcoeffs = MFCC(fs, sig, filtbankN)#--------TESTING
        melcoeffs = MFCC ( sig , fs )#--------TESTING
        melcoeffs = np.transpose ( melcoeffs )#--------TESTING
        codebooks_org[i,:,:] = lbg(melcoeffs, Centroids)
        
        plt.figure(i)
        plt.title('Speaker ' + str(i+1) + ' codeword with ' + str(Centroids) + ' centroids')
        for j in range(Centroids):
            plt.plot(211)
            plt.stem(codebooks_org[i,:,j])
            
    plt.show()
    print ('Training of model is complete!')

    #ploting actual cook books for first two speakers
    codebooks = np.empty((2, filtbankN, Centroids))
    melcoeffs = np.empty((2, filtbankN, 68))

    for i in range(2):
        file = '/Speaker' + str(i+1) + '.wav'
        (fs,sig) = read(dir_train + file)
#        melcoeffs[i,:,:] = MFCC(fs, sig, filtbankN)[:,0:68]#--------TESTING
#        codebooks[i,:,:] = lbg(melcoeffs[i,:,:], Centroids)#--------TESTING
        
        temp1 = MFCC(sig, fs)#--------TESTING
        temp1 = np.transpose(temp1)#--------TESTING
        melcoeffs[i,:,:] = temp1 [:,0:68]#--------TESTING
        codebooks[i,:,:] = lbg(melcoeffs[i,:,:], Centroids)#--------TESTING
        

    plt.figure(SpeakerN + 1)
    c1 = plt.scatter(codebooks[0,4,:], codebooks[1,5,:], s = 100, color = 'g', marker = '+')
    s1 = plt.scatter(melcoeffs[0,4,:], melcoeffs[0,5,:], s = 100, color = 'g', marker = 'o')
    c2 = plt.scatter(codebooks[1,4,:], codebooks[1,5,:], s = 100, color = 'b', marker = '+')
    s2 = plt.scatter(melcoeffs[1,4,:], melcoeffs[1,5,:], s = 100, color = 'b', marker = 'o')
    plt.grid()
    plt.legend((s1, s2, c1, c2), ('Speak1','Speak2','Speak1 centroids', 'Speak2 centroids'),scatterpoints=1,
    loc = 'lower right')
    plt.show()
    
    return (codebooks_org)