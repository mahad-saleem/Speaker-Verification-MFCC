"""
BSCS14002           Mohammad Asim Iqbal
BSCS14008           Mahad Saleem
BSCS14027           Muhammad Murtaza Azam Khan

ARTIFICIAL INTELLIGENCE
FINAL PROJECT
SPEAKER RECOGNITION USING MEL FREQUENCY CEPSTRAL COEFFICIENTS
Linde–Buzo–Gray algorithm for Vector Quantization
"""

from __future__ import division
import numpy as nu

def EUCDIST ( FEAT , CB ) :
    x = nu.shape ( FEAT ) [ 1 ]                     # reshape the FEATURES matrix by flattening it
    y = nu.shape ( CB ) [ 1 ]                       # reshape the CODEBOOK matrix by flattening it
    dist = nu.empty ( ( x , y ) )
    
    if x < y :
        for i in range ( x ) :
            temp = nu.transpose ( nu.tile ( FEAT [ : , i ] , ( y , 1 ) ) )
            dist [ i , : ] = nu.sum ( ( temp - CB ) ** 2 , 0 )
    else :
        for i in range ( y ) :
            temp = nu.transpose ( nu.tile ( CB [ : , i ] , ( x , 1 ) ) )
            dist [ : , i ] = nu.transpose ( nu.sum ( ( FEAT - temp ) ** 2 , 0 ) )
            
    dist = nu.sqrt ( dist )
    return dist
    
def LBG ( FEAT , NC ) :
    centroidNum = 1
    distortion = 1
    corr = 0.01
    cb = nu.mean ( FEAT , 1 )
    
    while centroidNum < NC :
        tempCB = nu.empty ( ( len ( cb ) , centroidNum * 2 ) )
        if centroidNum != 1 :
            for i in range ( centroidNum ) :
                tempCB [ : , i * 2 ] = cb [ : , i ] * ( 1 + corr )
                tempCB [ : , ( i * 2 ) + 1 ] = cb [ : , i ] * ( 1 - corr )
        else :
            tempCB [ : , 0 ] = cb * ( 1 + corr )
            tempCB [ : , 1 ] = cb * ( 1 - corr )
            
        cb = tempCB
        centroidNum = nu.shape ( cb ) [ 1 ]
        distArr = EUCDIST ( FEAT , cb )
        while nu.abs ( distortion ) > corr :
            prev = nu.mean ( distArr )
            nCB = nu.argmin ( distArr , axis = 1 )
            for i in range ( centroidNum ) :
                cb [ : , i ] = nu.mean ( FEAT [ : , nu.where ( nCB == i ) ] , 2 ).T
            cb = nu.nan_to_num ( cb )
            distArr = EUCDIST ( FEAT , cb )
            distortion = ( prev - nu.mean ( distArr ) ) / prev
    return cb