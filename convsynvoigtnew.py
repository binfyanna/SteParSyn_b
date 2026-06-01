#!/usr/bin/python
# 2-CLAUSE BSD LICENCE
#Copyright 2015-2026 Hugo Tabernero, Jonay Gonzalez Hernandez, and Emilio Marfil
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

from astropy.convolution import convolve,convolve_fft
from astropy.modeling.functional_models import Voigt1D
from scipy import interpolate, special
import scipy.signal as ss
import numpy as np
from numpy import mean
import matplotlib.pyplot as plt

def vlambda(inlamb,vstep):

    vlight=2.99792458e5
    xw2=max(inlamb)-0.1
    xw1=min(inlamb)+0.1
    iw1=np.where(inlamb > xw1)
    iw2=np.where(inlamb > xw2)
    iw1=iw1[0]
    iw2=iw2[0]
    w1=inlamb[iw1[0]+1]
    w2=inlamb[iw2[0]-1]

    npix=np.int64((iw2[0]-1)-(iw1[0]+1)*vstep)
    wmid=np.sqrt(w1*w2)
    vave=vlight*(10.**(np.log10(w2/w1)/(npix-1.))-1.)
    dwave=(np.log10(w2)-np.log10(w1))/(npix-1.)

    vwavel=np.log10(wmid)-(dwave*np.arange(np.int64(npix/2.)))
    vwaveu=np.log10(wmid)+(dwave*np.arange(np.int64(npix/2.)))
    vwavel=10.**vwavel[1:len(vwavel)-1]
    vwavel=np.sort(vwavel)
    vwaveu=10.**vwaveu[1:len(vwaveu)-1]
    vwave=np.concatenate((vwavel,[wmid],vwaveu))

    return vwave


def lsf_carmenes(channel, wave, x):

    """
    Input:
    - channel: string 'vis' or 'nir'.
    - wave: wavelength in microns.
    - x: wavelength in microns around wave.
    Output:
    - ils_vgt: Instrument Line Profile (Voigt).
    """

    # FWHM (delta_wave/wave) supplied by Evangelos.
    if type(channel) is str:
        if channel == 'vis':
            res_gauss = 1.01e-5
            res_lorentz = 2.1e-6
        elif channel == 'nir':
            res_gauss = 1.18e-5
            res_lorentz = 1.7e-6
        else:
            raise Exception('"channel" must be either "vis" or "nir" when calling lsf_carmenes()!')
    else:
        raise Exception('Check "channel" string when calling lsf_carmenes()!')

    # ad has been checked with Evangelos.
    ad = res_gauss * wave/(2. * np.sqrt(np.log(2.))) # v3 (missing 2?).

    # WARNING: check the parameters when calling the function down below.
    vgt = Voigt1D(x_0=0, amplitude_L=1, fwhm_L=res_lorentz * wave, fwhm_G=res_gauss * wave)
    ils_vgt = (1./(ad * np.sqrt(np.pi))) * vgt(x)

    return ils_vgt/sum(ils_vgt)


def convol_lsf(wn1, tr1, channel, plot=False):

    """Performs convolution of the synthetic spectrum."""

    # Interpolate at equidistant grid in wavelength.
    grid = 0.001  # Step of interpolation in angstroms.
    x0 = min(wn1)
    x1 = max(wn1)
    l_min = x0 - 1.
    l_max = x1 + 1.
    n = np.int64(((l_max - l_min)/grid) + 1)
    print('Points for fine grid in wavelength ' + str(n))
    wn2 = np.linspace(l_min, l_max, n)
    tr2inter = interpolate.interp1d(wn1, tr1, bounds_error=False, fill_value='extrapolate')
    tr2 = tr2inter(wn2)
    
    # Convolving, the original is preserved.
    wni = np.copy(wn2)
    tri = np.copy(tr2)
    nwi = len(wni)  # Equal to n as well.

    # Grid of the ILS kernel.    
    xmaxx = 0.8
    xminn = -xmaxx
    nk = np.int64((2 * xmaxx/grid) + 1)
    print('Number of points of the kernel ' + str(nk))
    x = np.linspace(xminn, xmaxx, nk)

    # Output grid. Reduced range to avoid problems in the convolution at the edges.
    wmin = wni[0] + 1.
    wmax = wni[-1] - 1.
    
    # Output grid (angstrom).
    grido = 0.01

    nwo = np.int64(((wmax - wmin)/grido) + 1)
    print('Number of points of the output spectrum ' + str(nwo))
    wno = np.linspace(wmin, wmax, nwo)
    tro = np.empty(nwo)
    
    # Loop over all output wns.
    for i in range(nwo):
        i0 = np.argmin(np.absolute(wni - wno[i]))
        i1 = max([0, i0 - nk/2])
        i2 = min([i0 + nk/2, nwi - 1])
        trr = tri[i1:i2]
        # Calculate the ILS at the given wavelength. Instrument Line Shape = ILS!
        kernel = lsf_carmenes(channel, wno[i], x)
        trx = ss.convolve(trr, kernel, mode='same')
        tro[i] = trx[nk/2]
    
    return tro


def conkern(inlamb, influx, vbroad, ldc, kop, channel=None):
    # You must provide vbroad in km/s.
    # Alternatively, you can also give the resolution.
    if vbroad >= 0.: 
        vstep=1
        vlight=2.99792458e5
        vwave=vlambda(inlamb,vstep)
        xw2=max(inlamb)-0.1
        xw1=min(inlamb)+0.1
        iw1=np.where(inlamb > xw1)
        iw2=np.where(inlamb > xw2)
        iw1=iw1[0]
        iw2=iw2[0]
        w1=inlamb[iw1[0]+1]
        w2=inlamb[iw2[0]-1]
        tck=interpolate.splrep(inlamb,influx,k=3, s=0)
        vflux=interpolate.splev(vwave,tck,der=0)
        npix=np.int64((iw2[0]-1)-(iw1[0]+1)*vstep)
        wmid=np.sqrt(w1*w2)
        vave=vlight*(10.**(np.log10(w2/w1)/(npix-1.))-1.)
	
        x1=vwave
        y1=vflux
        if kop == 'g': 
            if vbroad > 1000. and kop == 'g':
                vibr=(vlight/vbroad)/(2.*np.sqrt(2.*np.log(2.)))
            else:
                vibr=vbroad#(2.*np.sqrt(2.*np.log(2.)))
            sigma=vibr*wmid/vlight
            nx1=len(x1)
            dx1=(x1[nx1-1]-x1[0])/float(nx1-1)
            xk = (np.arange(nx1)-nx1/2)*dx1
            a1=0.
            a2=sigma
            zk=(xk-a1)/a2
            a0=1./np.sqrt(2.*np.pi)/sigma
            yk=a0*np.exp(-(zk**2.)/2.)
            #ii = np.where(yk > 0.0)

        elif kop == 'v':
            nx1=len(x1)
            dx1=(x1[nx1-1]-x1[0])/float(nx1-1)
            xk = (np.arange(nx1)-nx1/2)*dx1
            if type(channel) is str:
                 if channel == 'vis':
                     res_gauss = 1.01e-5
                     res_lorentz = 2.1e-6
                 elif channel == 'nir':
                     res_gauss = 1.18e-5
                     res_lorentz = 1.7e-6
                 else:
                     raise Exception('"channel" must be either "vis" or "nir" when calling lsf_carmenes()!')
            else:
                raise Exception('Check "channel" string when calling lsf_carmenes()!')
                # ad has been checked with Evangelos.
            ad = res_gauss * wmid/(2. *np.sqrt(np.log(2.))) # v3 (missing 2?).
            vgt = Voigt1D(x_0=0, amplitude_L=1, fwhm_L=res_lorentz * wmid, fwhm_G=res_gauss * wmid)
            yk = (1./(ad * np.sqrt(np.pi))) * vgt(xk)

        elif kop == 'r':
            vrot=vbroad
            xi=vrot/vlight*wmid
            d  = 1.0/(np.pi*(1.0-ldc/3.0))
            c1 = 2.0*(1.0-ldc)*d
            c2 = 0.5*np.pi*ldc*d
            nx1 = len(x1)
            dx1 = (x1[nx1-1]-x1[0])/float(nx1-1)
            xk = (np.arange(nx1)-nx1/2)*dx1
            d2 = 1.0-(xk/xi)**2
            ii = np.where(d2 <= 0.0)
            #xk = xk[ii]
            d2[ii] = 0
            yk = c1*np.sqrt(d2)+c2*d2
 
        elif kop == 'm':
            vmac = vbroad
            xi = vmac/vlight*wmid
            nx1 = len(x1)
            dx1 = (x1[nx1 - 1] - x1[0])/float(nx1 - 1)
            xk = (np.arange(nx1) - nx1/2) * dx1
            xk = (xk/xi)
            spi = np.sqrt(np.pi)
            A = 2./(spi*vmac)
            yk = A*(np.exp(-xk**2)-(spi*abs(xk)*special.erfc(abs(xk))))
        nfact = np.sum(yk)
        if nfact > 0.:
            outflux = convolve_fft(y1, yk/nfact, boundary='fill',fill_value=1.)  
        else:
            outflux = y1
        tck2 = interpolate.splrep(vwave,outflux,k=3, s=0)
        bflux = interpolate.splev(inlamb+0.01, tck2, der=0)

    else:
        bflux = influx

    return bflux

