"""
BSCS14002           Mohammad Asim Iqbal
BSCS14008           Mahad Saleem
BSCS14027           Muhammad Murtaza Azam Khan

ARTIFICIAL INTELLIGENCE
FINAL PROJECT
SPEAKER RECOGNITION USING MEL FREQUENCY CEPSTRAL COEFFICIENTS
Mel Frequency Cepstral Coefficient Calculation
"""

from __future__ import division
from scipy.fftpack import dct
from scipy.fftpack import fft
from scipy.fftpack import fftshift
from scipy.signal import hamming
import numpy as nu

def ToHertz ( mel ):
    return ( 700 * ( nu.exp ( mel / 1125 ) - 1 ) )
    
def ToMel ( hertz ):
    return ( 1125 * nu.log ( 1 + hertz / 700 ) )
    
def FilterBank ( fs , fftSize , filterBankSize ):
    lowerLimit = ToMel ( 300 )
    upperLimit = ToMel ( 8000 )
    mels = nu.linspace ( lowerLimit , upperLimit , filterBankSize + 2 )
    hertz = [ ToHertz ( mel ) for mel in mels ]
    bins = [ int ( freq * ( ( fftSize / 2 ) + 1 ) / fs ) for freq in hertz ]
    filterBank = nu.empty ( ( ( ( fftSize  / 2 ) + 1 ) , filterBankSize ) )
    for i in range ( 1 , filterBankSize + 1 ) :
        for k in range ( int ( ( fftSize / 2 ) + 1 ) ) :
            if k >= bins [ i ] and k <= bins [ i + 1 ] :
                filterBank [ k , i - 1 ] = ( bins [ i + 1 ] - k ) / ( bins [ i + 1 ] - bins [ i ] )
            elif k >= bins [ i - 1 ] and k < bins [ i ] :
                filterBank [ k , i - 1 ] = ( k - bins [ i - 1 ] ) / ( bins [ i ] - bins [ i - 1 ] )
            elif k < bins [ i - 1 ] :
                filterBank [ k , i - 1 ] = 0
            else :
                filterBank [ k , i - 1 ] = 0 
    return filterBank
    
def MFCC ( fs , sig , filterBankSize ) :
    sampleCount = nu.int32 ( fs * ( 25 / 1000 ) )
    overlap = nu.int32 ( fs * ( 10 / 1000 ) )
    frameCount = nu.int32 ( nu.ceil ( len ( sig ) / ( sampleCount - overlap ) ) )
    padCount = ( ( sampleCount - overlap ) * frameCount ) - len ( sig )
    if padCount > 0 :
        signal = nu.append ( sig , nu.zeros ( padCount ) )
    else :
        signal = sig
    window = nu.empty ( ( sampleCount , frameCount ) )
    mark = 0 
    for i in range ( frameCount ) :
        window [ : , i ] = signal [ mark : mark + sampleCount ]
        mark = i * ( sampleCount - overlap )
        
    fftSize = 512
    absSpect = nu.empty ( ( frameCount , ( int ) ( ( fftSize / 2 ) + 1 ) ) )
    for i in range ( frameCount ) :
        temp = hamming ( sampleCount ) * window [ : , i ]
        spect = fftshift ( fft ( temp , fftSize ) ) 
        absSpect [ i , : ] = abs ( spect [ ( ( fftSize / 2 ) - 1 ) : ] ) / sampleCount
        
    filterBank = FilterBank ( fs , fftSize , filterBankSize )
    mfcc = nu.empty ( ( filterBankSize , frameCount ) )
    for i in range ( filterBankSize ) :
        for k in range ( frameCount ) :
            mfcc [ i , k ] = nu.sum ( absSpect [ k , : ] * filterBank [ : , i ] )
            
    mfcc = nu.log10 ( mfcc )
    mfcc = dct ( mfcc )
    mfcc [ 0 , : ] = nu.zeros ( frameCount )
    return mfcc
    