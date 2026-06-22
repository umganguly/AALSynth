import numpy as np
from astropy import constants as const

###############################################################################################
# Returns an (nwave,nref) array with velocities
def calcvel(waveall,waveref):
    nref  = waveref.size
    nwave = waveall.size
    wsq   = np.square(np.transpose(np.broadcast_to(waveall, (nref,nwave))) /
                      np.broadcast_to(waveref, (nwave,nref)))
    return const.c * (wsq - 1.0) / (wsq + 1.0)

###############################################################################################
def calcvelzz(z1,z2):
    z1p1 = z1+1
    z2p1 = z2+1
    return const.c * (z1p1*z1p1 - z2p1*z2p1)/(z1p1*z1p1 + z2p1*z2p1)

###############################################################################################
# Returns an (nvel,nwave) array with observed wavelengths
def calcwave(velocity,waveref):
    nvel = velocity.size
    nwave = waveref.size
    beta = (np.copy(velocity)/const.c).decompose()
    wsq = (1 + beta)/(1 - beta)

    bwaveref = np.broadcast_to(waveref,      (nvel,nwave))
    bwsq     = np.broadcast_to(np.sqrt(wsq), (nwave,nvel))
    bwsqt    = np.transpose(bwsq)
    return np.multiply(bwaveref,bwsqt)
      
