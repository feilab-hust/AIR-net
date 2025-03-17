# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 09:42:16 2023

@author: sleepingcat
email
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def propagation_tf(u1, L,W, wavelen, z, kernel = 'AS'):
 
    Nx, Ny = u1.shape[0], u1.shape[1]
    k = 2*np.pi/wavelen
    dx = L/Nx
    fy = np.linspace(-1/(2*dx), 1/(2*dx)-1/L, Nx)
    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / W, Ny)
    FX, FY = np.meshgrid(fx, fy)
    
    if kernel == 'Fresnel':
        H = np.exp(-1j * np.pi * wavelen * z *(FX**2+FY**2) )
    else:
        temp = 1 - ((wavelen *FX)**2 + (wavelen *FY)**2)
        temp[temp<0] = 0
        H = np.exp(1j * k * z * np.sqrt(temp))
        H[temp<0] = 0

    H=fftshift(H)
    U1 = fft2(u1)
    U2 = U1*H
    return ifft2(U2)

def propagation_ir(u1, L,W, wavelen, z, kernel = 'R-S'):
  
    Nx, Ny = u1.shape[0], u1.shape[1]
    k = 2*np.pi/wavelen
    dx = L/Nx
    y = np.linspace(-L/2, L/2-L/Nx, Nx)
    x = np.linspace(-W / 2, W / 2 - W / Ny, Ny)
    x, y = np.meshgrid(x, y)
    
    if kernel == 'Fresnel':
        h= 1/ (1j*wavelen*z) * np.exp(1j*k/ (2*z) * (x**2+ y**2))
    else:
        r = np.sqrt(x**2 + y**2 + z**2)
        h = z/1j/wavelen/r**2* np.exp(1j*k*r)
     
    H=fft2(fftshift(h))*dx**2
    U1 = fft2(u1)
    # tif.imwrite('H.tif', abs(H) ** 2)
    U2 = U1*H
    return ifftshift(ifft2(U2))
    

