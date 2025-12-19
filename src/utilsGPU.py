import numpy as np
from numba import jit,prange, njit, cuda
import math

@cuda.jit()
def updateWaveEquationGPU(Uf,Uc,vp,nz,nx,dz,dx,dt):
    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.

    x, y = cuda.grid(2)

    if (x >= 4 and x < nx - 4 and y >= 4 and y < nz - 4):
        pxx = (c0 * Uc[y, x] + c1 * (Uc[y, x+1] + Uc[y, x-1]) + c2 * (Uc[y, x+2] + Uc[y, x-2]) + c3 * (Uc[y, x+3] + Uc[y, x-3]) + c4 * (Uc[y, x+4] + Uc[y, x-4])) / (dx*dx)
        pzz = (c0 * Uc[y, x] + c1 * (Uc[y+1, x] + Uc[y-1, x]) + c2 * (Uc[y+2, x] + Uc[y-2, x]) + c3 * (Uc[y+3, x] + Uc[y-3, x]) + c4 * (Uc[y+4, x] + Uc[y-4, x])) / (dz*dz)

        Uf[y, x] = (vp[y, x] * vp[y, x]) * (dt * dt) * (pxx + pzz) + 2. * Uc[y, x] - Uf[y, x]

@cuda.jit()
def updateWaveEquationCPMLGPU(Uf, Uc, vp, nx_abc, nz_abc, dz, dx, dt, PsixFR, PsixFL, PsizFU, PsizFD, ZetaxFR, ZetaxFL, ZetazFU, ZetazFD, N_abc):
    
    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    x,y = cuda.grid(2)

    # Região Interior 
    if (x >= N_abc and x < nx_abc - N_abc and y >= N_abc and y < nz_abc - N_abc):
        pxx = (c0 * Uc[y, x] + c1 * (Uc[y, x+1] + Uc[y, x-1]) +
                c2 * (Uc[y, x+2] + Uc[y, x-2]) + c3 * (Uc[y, x+3] + Uc[y, x-3]) +
                c4 * (Uc[y, x+4] + Uc[y, x-4])) / (dx * dx)
        pzz = (c0 * Uc[y, x] + c1 * (Uc[y+1, x] + Uc[y-1, x]) +
                c2 * (Uc[y+2, x] + Uc[y-2, x]) + c3 * (Uc[y+3, x] + Uc[y-3, x]) +
                c4 * (Uc[y+4, x] + Uc[y-4, x])) / (dz * dz)
        Uf[y, x] = (vp[y, x] * vp[y, x]) * (dt * dt) * (pxx + pzz) + 2. * Uc[y, x] - Uf[y, x]

    # Região Esquerda 
    elif (x >= 4 and x < N_abc and y >= N_abc and y < nz_abc - N_abc):
        pxx = (c0 * Uc[y, x] + c1 * (Uc[y, x+1] + Uc[y, x-1]) +
                c2 * (Uc[y, x+2] + Uc[y, x-2]) + c3 * (Uc[y, x+3] + Uc[y, x-3]) +
                c4 * (Uc[y, x+4] + Uc[y, x-4])) / (dx * dx)
        pzz = (c0 * Uc[y, x] + c1 * (Uc[y+1, x] + Uc[y-1, x]) +
                c2 * (Uc[y+2, x] + Uc[y-2, x]) + c3 * (Uc[y+3, x] + Uc[y-3, x]) +
                c4 * (Uc[y+4, x] + Uc[y-4, x])) / (dz * dz)
        psix = (a1 * (PsixFL[y, x+1] - PsixFL[y, x-1]) +
                a2 * (PsixFL[y, x+2] - PsixFL[y, x-2]) +
                a3 * (PsixFL[y, x+3] - PsixFL[y, x-3]) +
                a4 * (PsixFL[y, x+4] - PsixFL[y, x-4])) / dx

        Uf[y, x] = (vp[y, x] * vp[y, x]) * (dt*dt) * (pxx + pzz + psix + ZetaxFL[y, x]) + 2. * Uc[y, x] - Uf[y, x]
            
    # Região Direita
    elif (x >= nx_abc - N_abc and x < nx_abc - 4 and y >= N_abc and y < nz_abc - N_abc):
        idx = x - (nx_abc - N_abc)

        pxx = (c0 * Uc[y, x] + c1 * (Uc[y, x+1] + Uc[y, x-1]) +
            c2 * (Uc[y, x+2] + Uc[y, x-2]) + c3 * (Uc[y, x+3] + Uc[y, x-3]) +
            c4 * (Uc[y, x+4] + Uc[y, x-4])) / (dx * dx)
        pzz = (c0 * Uc[y, x] + c1 * (Uc[y+1, x] + Uc[y-1, x]) +
            c2 * (Uc[y+2, x] + Uc[y-2, x]) + c3 * (Uc[y+3, x] + Uc[y-3, x]) +
            c4 * (Uc[y+4, x] + Uc[y-4, x])) / (dz * dz)
        psix = (a1 * (PsixFR[y, idx+1] - PsixFR[y, idx-1]) +
                a2 * (PsixFR[y, idx+2] - PsixFR[y, idx-2]) +
                a3 * (PsixFR[y, idx+3] - PsixFR[y, idx-3]) +
                a4 * (PsixFR[y, idx+4] - PsixFR[y, idx-4])) / dx

        Uf[y, x] = (vp[y, x] * vp[y, x]) * (dt * dt) * (pxx + pzz + psix + ZetaxFR[y, idx]) + 2. * Uc[y, x] - Uf[y, x]

    # Região Superior 
    elif (x >= N_abc and x < nx_abc - N_abc and y >= 4 and y < N_abc):
        pxx = (c0 * Uc[y, x] + c1 * (Uc[y, x+1] + Uc[y, x-1]) +
                c2 * (Uc[y, x+2] + Uc[y, x-2]) + c3 * (Uc[y, x+3] + Uc[y, x-3]) +
                c4 * (Uc[y, x+4] + Uc[y, x-4])) / (dx * dx)
        pzz = (c0 * Uc[y, x] + c1 * (Uc[y+1, x] + Uc[y-1, x]) +
                c2 * (Uc[y+2, x] + Uc[y-2, x]) + c3 * (Uc[y+3, x] + Uc[y-3, x]) +
                c4 * (Uc[y+4, x] + Uc[y-4, x])) / (dz * dz)
        psiz = (a1 * (PsizFU[y+1, x] - PsizFU[y-1, x]) +
                a2 * (PsizFU[y+2, x] - PsizFU[y-2, x]) +
                a3 * (PsizFU[y+3, x] - PsizFU[y-3, x]) +
                a4 * (PsizFU[y+4, x] - PsizFU[y-4, x])) / dz          

        Uf[y, x] = (vp[y, x] * vp[y, x]) * (dt * dt) * (pxx + pzz + psiz + ZetazFU[y, x]) + 2. * Uc[y, x] - Uf[y, x]

    # Região Inferior
    elif (x >= N_abc and x < nx_abc - N_abc and y >= nz_abc - N_abc and y < nz_abc - 4):
        jdx = y - (nz_abc - N_abc)
        pxx = (c0 * Uc[y, x] + c1 * (Uc[y, x+1] + Uc[y, x-1]) +
                c2 * (Uc[y, x+2] + Uc[y, x-2]) + c3 * (Uc[y, x+3] + Uc[y, x-3]) +
                c4 * (Uc[y, x+4] + Uc[y, x-4])) / (dx * dx)
        pzz = (c0 * Uc[y, x] + c1 * (Uc[y+1, x] + Uc[y-1, x]) +
                c2 * (Uc[y+2, x] + Uc[y-2, x]) + c3 * (Uc[y+3, x] + Uc[y-3, x]) +
                c4 * (Uc[y+4, x] + Uc[y-4, x])) / (dz * dz)
        psiz = (a1 * (PsizFD[jdx+1, x] - PsizFD[jdx-1, x]) +
                a2 * (PsizFD[jdx+2, x] - PsizFD[jdx-2, x]) +
                a3 * (PsizFD[jdx+3, x] - PsizFD[jdx-3, x]) +
                a4 * (PsizFD[jdx+4, x] - PsizFD[jdx-4, x])) / dz
        
        Uf[y, x] = (vp[y, x] * vp[y, x]) * (dt * dt) * (pxx + pzz + psiz + ZetazFD[jdx, x]) + 2. * Uc[y, x] - Uf[y, x]

    # Quina Superior Esquerda
    elif (x >= 4 and x < N_abc and y >= 4 and y < N_abc):
        pxx = (c0 * Uc[y, x] + c1 * (Uc[y, x+1] + Uc[y, x-1]) +
                c2 * (Uc[y, x+2] + Uc[y, x-2]) + c3 * (Uc[y, x+3] + Uc[y, x-3]) +
                c4 * (Uc[y, x+4] + Uc[y, x-4])) / (dx * dx)
        pzz = (c0 * Uc[y, x] + c1 * (Uc[y+1, x] + Uc[y-1, x]) +
                c2 * (Uc[y+2, x] + Uc[y-2, x]) + c3 * (Uc[y+3, x] + Uc[y-3, x]) +
                c4 * (Uc[y+4, x] + Uc[y-4, x])) / (dz * dz)
        psiz = (a1 * (PsizFU[y+1, x] - PsizFU[y-1, x]) +
                a2 * (PsizFU[y+2, x] - PsizFU[y-2, x]) +
                a3 * (PsizFU[y+3, x] - PsizFU[y-3, x]) +
                a4 * (PsizFU[y+4, x] - PsizFU[y-4, x])) / dz   
        psix = (a1 * (PsixFL[y, x+1] - PsixFL[y, x-1]) +
                a2 * (PsixFL[y, x+2] - PsixFL[y, x-2]) +
                a3 * (PsixFL[y, x+3] - PsixFL[y, x-3]) +
                a4 * (PsixFL[y, x+4] - PsixFL[y, x-4])) / dx
        
        Uf[y, x] = (vp[y, x] * vp[y, x]) * (dt * dt) * (pxx + pzz + psix + psiz + ZetaxFL[y, x] + ZetazFU[y, x]) + 2. * Uc[y, x] - Uf[y, x]

    # Quina Superior Direita 
    elif (x >= nx_abc - N_abc and x < nx_abc - 4 and y >= 4 and y < N_abc):
        idx = x - (nx_abc - N_abc)
        pxx = (c0 * Uc[y, x] + c1 * (Uc[y, x+1] + Uc[y, x-1]) +
                c2 * (Uc[y, x+2] + Uc[y, x-2]) + c3 * (Uc[y, x+3] + Uc[y, x-3]) +
                c4 * (Uc[y, x+4] + Uc[y, x-4])) / (dx * dx)
        pzz = (c0 * Uc[y, x] + c1 * (Uc[y+1, x] + Uc[y-1, x]) +
                c2 * (Uc[y+2, x] + Uc[y-2, x]) + c3 * (Uc[y+3, x] + Uc[y-3, x]) +
                c4 * (Uc[y+4, x] + Uc[y-4, x])) / (dz * dz)
        psix = (a1 * (PsixFR[y, idx+1] - PsixFR[y, idx-1]) +
                    a2 * (PsixFR[y, idx+2] - PsixFR[y, idx-2]) +
                    a3 * (PsixFR[y, idx+3] - PsixFR[y, idx-3]) +
                    a4 * (PsixFR[y, idx+4] - PsixFR[y, idx-4])) / dx
        psiz = (a1 * (PsizFU[y+1, x] - PsizFU[y-1, x]) +
                a2 * (PsizFU[y+2, x] - PsizFU[y-2, x]) +
                a3 * (PsizFU[y+3, x] - PsizFU[y-3, x]) +
                a4 * (PsizFU[y+4, x] - PsizFU[y-4, x])) / dz          
        
        Uf[y, x] = (vp[y, x] * vp[y, x]) * (dt * dt) * (pxx + pzz + psix + psiz + ZetaxFR[y, idx] + ZetazFU[y, x]) + 2. * Uc[y, x] - Uf[y, x]

    # Quina Inferior Esquerda 
    elif (x >= 4 and x < N_abc and y >= nz_abc - N_abc and y < nz_abc - 4):
        jdx = y - (nz_abc - N_abc)

        pxx = (c0 * Uc[y, x] + c1 * (Uc[y, x+1] + Uc[y, x-1]) +
                c2 * (Uc[y, x+2] + Uc[y, x-2]) + c3 * (Uc[y, x+3] + Uc[y, x-3]) +
                c4 * (Uc[y, x+4] + Uc[y, x-4])) / (dx * dx)
        pzz = (c0 * Uc[y, x] + c1 * (Uc[y+1, x] + Uc[y-1, x]) +
                c2 * (Uc[y+2, x] + Uc[y-2, x]) + c3 * (Uc[y+3, x] + Uc[y-3, x]) +
                c4 * (Uc[y+4, x] + Uc[y-4, x])) / (dz * dz)
        psix = (a1 * (PsixFL[y, x+1] - PsixFL[y, x-1]) +
                a2 * (PsixFL[y, x+2] - PsixFL[y, x-2]) +
                a3 * (PsixFL[y, x+3] - PsixFL[y, x-3]) +
                a4 * (PsixFL[y, x+4] - PsixFL[y, x-4])) / dx
        psiz = (a1 * (PsizFD[jdx+1, x] - PsizFD[jdx-1, x]) +
                a2 * (PsizFD[jdx+2, x] - PsizFD[jdx-2, x]) +
                a3 * (PsizFD[jdx+3, x] - PsizFD[jdx-3, x]) +
                a4 * (PsizFD[jdx+4, x] - PsizFD[jdx-4, x])) / dz
        
        Uf[y, x] = (vp[y, x] * vp[y, x]) * (dt * dt) * (pxx + pzz + psix + psiz + ZetaxFL[y, x] + ZetazFD[jdx, x]) + 2. * Uc[y, x] - Uf[y, x]

    # Quina Inferior Direita
    elif (x >= nx_abc - N_abc and x < nx_abc - 4 and y >= nz_abc - N_abc and y < nz_abc - 4): 
        idx = x - (nx_abc - N_abc)
        jdx = y - (nz_abc - N_abc)

        pxx = (c0 * Uc[y, x] + c1 * (Uc[y, x+1] + Uc[y, x-1]) +
                c2 * (Uc[y, x+2] + Uc[y, x-2]) + c3 * (Uc[y, x+3] + Uc[y, x-3]) +
                c4 * (Uc[y, x+4] + Uc[y, x-4])) / (dx * dx)
        pzz = (c0 * Uc[y, x] + c1 * (Uc[y+1, x] + Uc[y-1, x]) +
            c2 * (Uc[y+2, x] + Uc[y-2, x]) + c3 * (Uc[y+3, x] + Uc[y-3, x]) +
            c4 * (Uc[y+4, x] + Uc[y-4, x])) / (dz * dz)
        psix = (a1 * (PsixFR[y, idx+1] - PsixFR[y, idx-1]) +
                a2 * (PsixFR[y, idx+2] - PsixFR[y, idx-2]) +
                a3 * (PsixFR[y, idx+3] - PsixFR[y, idx-3]) +
                a4 * (PsixFR[y, idx+4] - PsixFR[y, idx-4])) / dx   
        psiz = (a1 * (PsizFD[jdx+1, x] - PsizFD[jdx-1, x]) +
                a2 * (PsizFD[jdx+2, x] - PsizFD[jdx-2, x]) +
                a3 * (PsizFD[jdx+3, x] - PsizFD[jdx-3, x]) +
                a4 * (PsizFD[jdx+4, x] - PsizFD[jdx-4, x])) / dz
        
        Uf[y, x] = (vp[y, x] * vp[y, x]) * (dt * dt) * (pxx + pzz + psix + psiz + ZetaxFR[y, idx] + ZetazFD[jdx, x]) + 2. * Uc[y, x] - Uf[y, x]

@cuda.jit()
def updatePsiGPU(PsixFR, PsixFL, PsizFU, PsizFD, nx_abc, nz_abc, Uc, dx,dz, N_abc,ax,bx,az,bz):

    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    x,y = cuda.grid(2)

    if (x >= 4 and x < N_abc and y >= 4 and y < nz_abc - 4):      
        px = (a1 * (Uc[y, x+1] - Uc[y, x-1]) +
            a2 * (Uc[y, x+2] - Uc[y, x-2]) +
            a3 * (Uc[y, x+3] - Uc[y, x-3]) +
            a4 * (Uc[y, x+4] - Uc[y, x-4])) / dx
        
        PsixFL[y, x] = ax[y,x] * PsixFL[y, x] + bx[y,x] * px

    elif (x >= nx_abc - N_abc and x < nx_abc - 4 and y >= 4 and y < nz_abc - 4):
        idx = x - (nx_abc - N_abc)

        px = (a1 * (Uc[y, x+1] - Uc[y, x-1]) +
            a2 * (Uc[y, x+2] - Uc[y, x-2]) +
            a3 * (Uc[y, x+3] - Uc[y, x-3]) +
            a4 * (Uc[y, x+4] - Uc[y, x-4])) / dx
        
        PsixFR[y, idx] = ax[y,x] * PsixFR[y, idx] + bx[y,x] * px

    elif (x >= 4 and x < nx_abc - 4 and y >= 4 and y < N_abc):
        pz = (a1 * (Uc[y+1, x] - Uc[y-1, x]) +
            a2 * (Uc[y+2, x] - Uc[y-2, x]) +
            a3 * (Uc[y+3, x] - Uc[y-3, x]) +
            a4 * (Uc[y+4, x] - Uc[y-4, x])) / dz 
        
        PsizFU[y, x] = az[y,x] * PsizFU[y, x] + bz[y,x] * pz

    elif (x >= 4 and x < nx_abc - 4 and y >= nz_abc - N_abc and y < nz_abc - 4):
        jdx = y - (nz_abc - N_abc)

        pz = (a1 * (Uc[y+1, x] - Uc[y-1, x]) +
            a2 * (Uc[y+2, x] - Uc[y-2, x]) +
            a3 * (Uc[y+3, x] - Uc[y-3, x]) +
            a4 * (Uc[y+4, x] - Uc[y-4, x])) / dz 
        
        PsizFD[jdx, x] = az[y,x] * PsizFD[jdx, x] + bz[y,x] * pz

@cuda.jit()
def updateZetaGPU(PsixFR, PsixFL, ZetaxFR, ZetaxFL,PsizFU, PsizFD, ZetazFU, ZetazFD, nx_abc, nz_abc, Uc, dx, dz, N_abc,ax,bx,az,bz):

    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    x,y = cuda.grid(2)

    if (x >= 4 and x < N_abc and y >= 4 and y < nz_abc - 4):    
        pxx = (c0 * Uc[y, x] + c1 * (Uc[y, x+1] + Uc[y, x-1]) +
            c2 * (Uc[y, x+2] + Uc[y, x-2]) + 
            c3 * (Uc[y, x+3] + Uc[y, x-3]) +
            c4 * (Uc[y, x+4] + Uc[y, x-4])) / (dx * dx)
    
        psix = (a1 * (PsixFL[y, x+1] - PsixFL[y, x-1]) +
                a2 * (PsixFL[y, x+2] - PsixFL[y, x-2]) +
                a3 * (PsixFL[y, x+3] - PsixFL[y, x-3]) +
                a4 * (PsixFL[y, x+4] - PsixFL[y, x-4])) / dx

        ZetaxFL[y, x] = ax[y,x] * ZetaxFL[y, x] + bx[y,x] * (pxx + psix)

    elif (x >= nx_abc - N_abc and x < nx_abc - 4 and y >= 4 and y < nz_abc - 4):
        idx = x - (nx_abc - N_abc) 
            
        pxx = (c0 * Uc[y, x] + c1 * (Uc[y, x+1] + Uc[y, x-1]) +
            c2 * (Uc[y, x+2] + Uc[y, x-2]) + 
            c3 * (Uc[y, x+3] + Uc[y, x-3]) +
            c4 * (Uc[y, x+4] + Uc[y, x-4])) / (dx * dx)
    
        psix = (a1 * (PsixFR[y, idx+1] - PsixFR[y, idx-1]) +
                a2 * (PsixFR[y, idx+2] - PsixFR[y, idx-2]) +
                a3 * (PsixFR[y, idx+3] - PsixFR[y, idx-3]) +
                a4 * (PsixFR[y, idx+4] - PsixFR[y, idx-4])) / dx

        ZetaxFR[y, idx] = ax[y,x] * ZetaxFR[y, idx] + bx[y,x] * (pxx + psix)

    elif (x >= 4 and x < nx_abc - 4 and y >= 4 and y < N_abc):        
        pzz = (c0 * Uc[y, x] + c1 * (Uc[y+1, x] + Uc[y-1, x]) +
            c2 * (Uc[y+2, x] + Uc[y-2, x]) + 
            c3 * (Uc[y+3, x] + Uc[y-3, x]) +
            c4 * (Uc[y+4, x] + Uc[y-4, x])) / (dz * dz)               
        psiz = (a1 * (PsizFU[y+1, x] - PsizFU[y-1, x]) +
                a2 * (PsizFU[y+2, x] - PsizFU[y-2, x]) +
                a3 * (PsizFU[y+3, x] - PsizFU[y-3, x]) +
                a4 * (PsizFU[y+4, x] - PsizFU[y-4, x])) / dz
        
        ZetazFU[y, x] = az[y,x] * ZetazFU[y, x] + bz[y,x] * (pzz + psiz)

    elif (x >= 4 and x < nx_abc - 4 and y >= nz_abc - N_abc and y < nz_abc - 4):
        jdx = y - (nz_abc - N_abc)
        
        pzz = (c0 * Uc[y, x] + c1 * (Uc[y+1, x] + Uc[y-1, x]) +
            c2 * (Uc[y+2, x] + Uc[y-2, x]) + 
            c3 * (Uc[y+3, x] + Uc[y-3, x]) +
            c4 * (Uc[y+4, x] + Uc[y-4, x])) / (dz * dz)               
        psiz = (a1 * (PsizFD[jdx+1, x] - PsizFD[jdx-1, x]) +
                a2 * (PsizFD[jdx+2, x] - PsizFD[jdx-2, x]) +
                a3 * (PsizFD[jdx+3, x] - PsizFD[jdx-3, x]) +
                a4 * (PsizFD[jdx+4, x] - PsizFD[jdx-4, x])) / dz
        
        ZetazFD[jdx, x] = az[y,x] * ZetazFD[jdx, x] + bz[y,x] * (pzz + psiz) 

@cuda.jit()
def updateWaveEquationVTIGPU(Uf, Uc, Qc, Qf, nx, nz, dt, dx, dz, vpz, epsilon, delta):  
    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.

    x, y = cuda.grid(2)

    if (x >= 4 and x < nx - 4 and y >= 4 and y < nz - 4):
            
            cx = vpz[y,x]**2 * (1 + 2 * epsilon[y,x])
            bx = vpz[y,x]**2 * (1 + 2 * delta[y,x])
            cz = bz = vpz[y,x]**2      
            pxx = (c0 * Uc[y, x] + c1 * (Uc[y, x+1] + Uc[y, x-1]) + c2 * (Uc[y, x+2] + Uc[y, x-2]) + 
                   c3 * (Uc[y, x+3] + Uc[y, x-3]) + c4 * (Uc[y, x+4] + Uc[y, x-4])) / (dx * dx)
            qzz = (c0 * Qc[y, x] + c1 * (Qc[y+1, x] + Qc[y-1, x]) + c2 * (Qc[y+2, x] + Qc[y-2, x]) + 
                   c3 * (Qc[y+3, x] + Qc[y-3, x]) + c4 * (Qc[y+4, x] + Qc[y-4, x])) / (dz * dz)
            Uf[y, x] = 2 * Uc[y, x] - Uf[y, x] + (dt**2) * (cx * pxx  + cz * qzz)
            Qf[y, x] = 2 * Qc[y, x] - Qf[y, x] + (dt**2) * (bx * pxx  + bz * qzz)

@cuda.jit()
def updateWaveEquationVTICPMLGPU(Uf, Uc, Qc, Qf, dt, dx, dz, vpz, epsilon, delta,
                               nx_abc, nz_abc, PsixFR, PsixFL, PsizqFU, PsizqFD, ZetaxFR, ZetaxFL, ZetazqFU, ZetazqFD, N_abc):
    
    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    i,j = cuda.grid(2)

    # Região Interior
    if (i >= N_abc and i < nx_abc - N_abc and j >= N_abc and j < nz_abc - N_abc):
        cx = vpz[j,i]**2 * (1 + 2 * epsilon[j,i])
        bx = vpz[j,i]**2 * (1 + 2 * delta[j,i])
        cz = bz = vpz[j,i]**2      
        pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) + c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                c3 * (Uc[j, i+3] + Uc[j, i-3]) + c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
        qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                c3 * (Qc[j+3, i] + Qc[j-3, i]) + c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
        Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * pxx  + cz * qzz)
        Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * pxx  + bz * qzz)

    # Região Esquerda
    elif (i >= 4 and i < N_abc and j >= N_abc and j < nz_abc - N_abc):
            cx = vpz[j,i]**2 * (1 + 2 * epsilon[j,i])
            bx = vpz[j,i]**2 * (1 + 2 * delta[j,i])
            cz = bz = vpz[j,i]**2
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) + c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) + c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                   c3 * (Qc[j+3, i] + Qc[j-3, i]) + c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx           
                  
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * (pxx + psix + ZetaxFL[j,i] ) + cz * qzz)
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * (pxx + psix + ZetaxFL[j,i]) + bz * qzz)

    # Região Direita
    elif (i >= nx_abc - N_abc and i < nx_abc - 4 and j >= N_abc and j < nz_abc - N_abc):
        idx = i - (nx_abc - N_abc)

        cx = vpz[j,i]**2 * (1 + 2 * epsilon[j,i])
        bx = vpz[j,i]**2 * (1 + 2 * delta[j,i])
        cz = bz = vpz[j,i]**2
        pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) + c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                c3 * (Uc[j, i+3] + Uc[j, i-3]) + c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
        qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                c3 * (Qc[j+3, i] + Qc[j-3, i]) + c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
        psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx           
                
        Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * (pxx + psix + ZetaxFR[j,idx] ) + cz * qzz)
        Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * (pxx + psix + ZetaxFR[j,idx]) + bz * qzz)

    # Região Superior
    elif (i >= N_abc and i < nx_abc - N_abc and j >= 4 and j < N_abc):
            cx = vpz[j,i]**2 * (1 + 2 * epsilon[j,i])
            bx = vpz[j,i]**2 * (1 + 2 * delta[j,i])
            cz = bz = vpz[j,i]**2
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) + c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) + c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                   c3 * (Qc[j+3, i] + Qc[j-3, i]) + c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
            psiqz = (a1 * (PsizqFU[j+1, i] - PsizqFU[j-1, i]) +
                    a2 * (PsizqFU[j+2, i] - PsizqFU[j-2, i]) +
                    a3 * (PsizqFU[j+3, i] - PsizqFU[j-3, i]) +
                    a4 * (PsizqFU[j+4, i] - PsizqFU[j-4, i])) / dz           
                  
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * pxx + cz *(qzz + psiqz + ZetazqFU[j,i]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * pxx + bz *(qzz + psiqz + ZetazqFU[j,i]))

    # Região Inferior
    elif (i >= N_abc and i < nx_abc - N_abc and j >= nz_abc - N_abc and j < nz_abc - 4):
        jdx = j - (nz_abc - N_abc)

        cx = vpz[j,i]**2 * (1 + 2 * epsilon[j,i])
        bx = vpz[j,i]**2 * (1 + 2 * delta[j,i])
        cz = bz = vpz[j,i]**2
        pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) + c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                c3 * (Uc[j, i+3] + Uc[j, i-3]) + c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
        qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                c3 * (Qc[j+3, i] + Qc[j-3, i]) + c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
        psiqz = (a1 * (PsizqFD[jdx+1, i] - PsizqFD[jdx-1, i]) +
                a2 * (PsizqFD[jdx+2, i] - PsizqFD[jdx-2, i]) +
                a3 * (PsizqFD[jdx+3, i] - PsizqFD[jdx-3, i]) +
                a4 * (PsizqFD[jdx+4, i] - PsizqFD[jdx-4, i])) / dz           
                
        Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * pxx  + cz *(qzz + psiqz + ZetazqFD[jdx,i]))
        Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * pxx  + bz *(qzz + psiqz + ZetazqFD[jdx,i]))

    # Quina Superior Esquerda
    elif (i >= 4 and i < N_abc and j >= 4 and j < N_abc):
            cx = vpz[j,i]**2 * (1 + 2 * epsilon[j,i])
            bx = vpz[j,i]**2 * (1 + 2 * delta[j,i])
            cz = bz = vpz[j,i]**2
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) + c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) + c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                   c3 * (Qc[j+3, i] + Qc[j-3, i]) + c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
            psiqz = (a1 * (PsizqFU[j+1, i] - PsizqFU[j-1, i]) +
                    a2 * (PsizqFU[j+2, i] - PsizqFU[j-2, i]) +
                    a3 * (PsizqFU[j+3, i] - PsizqFU[j-3, i]) +
                    a4 * (PsizqFU[j+4, i] - PsizqFU[j-4, i])) / dz
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx           
                  
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * (pxx + psix + ZetaxFL[j,i]) + cz *(qzz + psiqz + ZetazqFU[j,i]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * (pxx + psix + ZetaxFL[j,i]) + bz *(qzz + psiqz + ZetazqFU[j,i]))

    # Quina Superior Direita
    elif (i >= nx_abc - N_abc and i < nx_abc - 4 and j >= 4 and j < N_abc):
        idx = i - (nx_abc - N_abc)

        cx = vpz[j,i]**2 * (1 + 2 * epsilon[j,i])
        bx = vpz[j,i]**2 * (1 + 2 * delta[j,i])
        cz = bz = vpz[j,i]**2
        pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) + c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                c3 * (Uc[j, i+3] + Uc[j, i-3]) + c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
        qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                c3 * (Qc[j+3, i] + Qc[j-3, i]) + c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
        psiqz = (a1 * (PsizqFU[j+1, i] - PsizqFU[j-1, i]) +
                a2 * (PsizqFU[j+2, i] - PsizqFU[j-2, i]) +
                a3 * (PsizqFU[j+3, i] - PsizqFU[j-3, i]) +
                a4 * (PsizqFU[j+4, i] - PsizqFU[j-4, i])) / dz
        psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx           
                
        Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * (pxx + psix + ZetaxFR[j,idx]) + cz *(qzz + psiqz + ZetazqFU[j,i]))
        Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * (pxx + psix + ZetaxFR[j,idx]) + bz *(qzz + psiqz + ZetazqFU[j,i]))

    # Quina Inferior Esquerda
    elif (i >= 4 and i < N_abc and j >= nz_abc - N_abc and j < nz_abc - 4):
        jdx = j - (nz_abc - N_abc)

        cx = vpz[j,i]**2 * (1 + 2 * epsilon[j,i])
        bx = vpz[j,i]**2 * (1 + 2 * delta[j,i])
        cz = bz = vpz[j,i]**2
        pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) + c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                c3 * (Uc[j, i+3] + Uc[j, i-3]) + c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
        qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                c3 * (Qc[j+3, i] + Qc[j-3, i]) + c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
        psiqz = (a1 * (PsizqFD[jdx+1, i] - PsizqFD[jdx-1, i]) +
                a2 * (PsizqFD[jdx+2, i] - PsizqFD[jdx-2, i]) +
                a3 * (PsizqFD[jdx+3, i] - PsizqFD[jdx-3, i]) +
                a4 * (PsizqFD[jdx+4, i] - PsizqFD[jdx-4, i])) / dz
        psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx           
                
        Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * (pxx + psix + ZetaxFL[j,i]) + cz *(qzz + psiqz + ZetazqFD[jdx,i]))
        Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * (pxx + psix + ZetaxFL[j,i]) + bz *(qzz + psiqz + ZetazqFD[jdx,i]))

    # Quina Inferior Direita
    elif (i >= nx_abc - N_abc and i < nx_abc - 4 and j >= nz_abc - N_abc and j < nz_abc - 4):  
        idx = i - (nx_abc - N_abc)
        jdx = j - (nz_abc - N_abc)

        cx = vpz[j,i]**2 * (1 + 2 * epsilon[j,i])
        bx = vpz[j,i]**2 * (1 + 2 * delta[j,i])
        cz = bz = vpz[j,i]**2
        pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) + c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                c3 * (Uc[j, i+3] + Uc[j, i-3]) + c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
        qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                c3 * (Qc[j+3, i] + Qc[j-3, i]) + c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
        psiqz = (a1 * (PsizqFD[jdx+1, i] - PsizqFD[jdx-1, i]) +
                a2 * (PsizqFD[jdx+2, i] - PsizqFD[jdx-2, i]) +
                a3 * (PsizqFD[jdx+3, i] - PsizqFD[jdx-3, i]) +
                a4 * (PsizqFD[jdx+4, i] - PsizqFD[jdx-4, i])) / dz
        psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx           
                
        Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * (pxx + psix + ZetaxFR[j,idx]) + cz *(qzz + psiqz + ZetazqFD[jdx,i]))
        Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * (pxx + psix + ZetaxFR[j,idx]) + bz *(qzz + psiqz + ZetazqFD[jdx,i]))

@cuda.jit()
def updatePsiVTIGPU (PsizqFU, PsizqFD, nx_abc, nz_abc, a_z, b_z, Qc, dz, N_abc):

    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    i,j = cuda.grid(2)

    if (i >= 4 and i < nx_abc - 4 and j >= 4 and j < N_abc):
        qz = (a1 * (Qc[j+1, i] - Qc[j-1, i]) +
            a2 * (Qc[j+2, i] - Qc[j-2, i]) +
            a3 * (Qc[j+3, i] - Qc[j-3, i]) +
            a4 * (Qc[j+4, i] - Qc[j-4, i])) / dz 
        
        PsizqFU[j, i] = a_z[j, i] * PsizqFU[j, i] + b_z[j, i] * qz

    if (i >= 4 and i < nx_abc - 4 and j >= nz_abc - N_abc and j < nz_abc - 4):
        jdx = j - (nz_abc - N_abc)

        qz = (a1 * (Qc[j+1, i] - Qc[j-1, i]) +
            a2 * (Qc[j+2, i] - Qc[j-2, i]) +
            a3 * (Qc[j+3, i] - Qc[j-3, i]) +
            a4 * (Qc[j+4, i] - Qc[j-4, i])) / dz 
        
        PsizqFD[jdx, i] = a_z[j, i] * PsizqFD[jdx, i] + b_z[j, i] * qz


@cuda.jit()
def updateZetaVTIGPU (PsizqFU, PsizqFD, ZetazqFU, ZetazqFD, nx_abc, nz_abc, a_z, b_z, Qc, dz, N_abc):

    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    i,j = cuda.grid(2)

    if (i >= 4 and i < nx_abc - 4 and j >= 4 and j < N_abc):
        qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + 
                c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                c3 * (Qc[j+3, i] + Qc[j-3, i]) + 
                c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
        psiqz = (a1 * (PsizqFU[j+1, i] - PsizqFU[j-1, i]) +
                a2 * (PsizqFU[j+2, i] - PsizqFU[j-2, i]) +
                a3 * (PsizqFU[j+3, i] - PsizqFU[j-3, i]) +
                a4 * (PsizqFU[j+4, i] - PsizqFU[j-4, i])) / dz

        ZetazqFU[j, i] = a_z[j, i] * ZetazqFU[j, i] + b_z[j, i] * (qzz + psiqz)

    if (i >= 4 and i < nx_abc - 4 and j >= nz_abc - N_abc and j < nz_abc - 4):
        jdx = j - (nz_abc - N_abc) 
        qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + 
                c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                c3 * (Qc[j+3, i] + Qc[j-3, i]) + 
                c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
        psiqz = (a1 * (PsizqFD[jdx+1, i] - PsizqFD[jdx-1, i]) +
                a2 * (PsizqFD[jdx+2, i] - PsizqFD[jdx-2, i]) +
                a3 * (PsizqFD[jdx+3, i] - PsizqFD[jdx-3, i]) +
                a4 * (PsizqFD[jdx+4, i] - PsizqFD[jdx-4, i])) / dz

        ZetazqFD[jdx, i] = a_z[j, i] * ZetazqFD[jdx, i] + b_z[j, i] * (qzz + psiqz)

@cuda.jit()
def updateWaveEquationTTIGPU(Uf, Uc, Qc, Qf, nx, nz, dt, dx, dz, vpz, vsz, epsilon, delta, theta):
    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 4. / 5.
    a2 = -1. / 5.
    a3 = 4./105.
    a4 = -1./280.

    i, j = cuda.grid(2)

    if (i >= 4 and i < nx - 4 and j >= 4 and j < nz - 4):
        vpx = vpz[j, i] * math.sqrt(1 + 2*epsilon[j, i])
        vpn = vpz[j, i] * math.sqrt(1 + 2*delta[j, i])
        cpx = vpx**2 * math.cos(theta[j, i])**2 + vsz[j, i]**2 * math.sin(theta[j, i])**2
        cpz = vpx**2 * math.sin(theta[j, i])**2 + vsz[j, i]**2 * math.cos(theta[j, i])**2
        cpxz = vsz[j, i]**2 * math.sin(2 * theta[j, i]) - vpx**2 * math.sin(2 * theta[j, i])
        dpx = vpz[j, i]**2 * math.sin(theta[j, i])**2 - vsz[j, i]**2 * math.sin(theta[j, i])**2
        dpz = vpz[j, i]**2 * math.cos(theta[j, i])**2 - vsz[j, i]**2 * math.cos(theta[j, i])**2
        dpxz = vpz[j, i]**2 * math.sin(2 * theta[j, i]) - vsz[j, i]**2 * math.sin(2 * theta[j, i])
        cqx = vpn**2 * math.cos(theta[j, i])**2 - vsz[j, i]**2 * math.cos(theta[j, i])**2
        cqz = vpn**2 * math.sin(theta[j, i])**2 - vsz[j, i]**2 * math.sin(theta[j, i])**2  
        cqxz = vsz[j, i]**2 * math.sin(2 * theta[j, i]) - vpn**2 * math.sin(2 * theta[j, i])
        dqx = vsz[j, i]**2 * math.cos(theta[j, i])**2 + vpz[j, i]**2 * math.sin(theta[j, i])**2  
        dqz = vpz[j, i]**2 * math.cos(theta[j, i])**2 + vsz[j, i]**2 * math.sin(theta[j, i])**2
        dqxz = vpz[j, i]**2 * math.sin(2 * theta[j, i]) - vsz[j, i]**2 * math.sin(2 * theta[j, i])

        pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) + c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                c3 * (Uc[j, i+3] + Uc[j, i-3]) + c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
        
        pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) + c2 * (Uc[j+2, i] + Uc[j-2, i]) +
                c3 * (Uc[j+3, i] + Uc[j-3, i]) + c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
        
        pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
                a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
                a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
                a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

                a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
                a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
                a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
                a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

                a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
                a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
                a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
                a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

                a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
                a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
                a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
                a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx)
        
        qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + c2 * (Qc[j+2, i] + Qc[j-2, i]) +
                c3 * (Qc[j+3, i] + Qc[j-3, i]) + c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
        
        qxx = (c0 * Qc[j, i] + c1 * (Qc[j, i+1] + Qc[j, i-1]) + c2 * (Qc[j, i+2] + Qc[j, i-2]) +
                c3 * (Qc[j, i+3] + Qc[j, i-3]) + c4 * (Qc[j, i+4] + Qc[j, i-4])) / (dx * dx)
        
        qxz =  (a1*a1*(Qc[j+1,i+1] - Qc[j-1,i+1] + Qc[j-1,i-1] - Qc[j+1,i-1]) +
                a1*a2*(Qc[j+2,i+1] - Qc[j-2,i+1] + Qc[j-2,i-1] - Qc[j+2,i-1]) +
                a1*a3*(Qc[j+3,i+1] - Qc[j-3,i+1] + Qc[j-3,i-1] - Qc[j+3,i-1]) +
                a1*a4*(Qc[j+4,i+1] - Qc[j-4,i+1] + Qc[j-4,i-1] - Qc[j+4,i-1]) +

                a2*a1*(Qc[j+1,i+2] - Qc[j-1,i+2] + Qc[j-1,i-2] - Qc[j+1,i-2]) +
                a2*a2*(Qc[j+2,i+2] - Qc[j-2,i+2] + Qc[j-2,i-2] - Qc[j+2,i-2]) +
                a2*a3*(Qc[j+3,i+2] - Qc[j-3,i+2] + Qc[j-3,i-2] - Qc[j+3,i-2]) +
                a2*a4*(Qc[j+4,i+2] - Qc[j-4,i+2] + Qc[j-4,i-2] - Qc[j+4,i-2]) +

                a3*a1*(Qc[j+1,i+3] - Qc[j-1,i+3] + Qc[j-1,i-3] - Qc[j+1,i-3]) +
                a3*a2*(Qc[j+2,i+3] - Qc[j-2,i+3] + Qc[j-2,i-3] - Qc[j+2,i-3]) +
                a3*a3*(Qc[j+3,i+3] - Qc[j-3,i+3] + Qc[j-3,i-3] - Qc[j+3,i-3]) +
                a3*a4*(Qc[j+4,i+3] - Qc[j-4,i+3] + Qc[j-4,i-3] - Qc[j+4,i-3]) +

                a4*a1*(Qc[j+1,i+4] - Qc[j-1,i+4] + Qc[j-1,i-4] - Qc[j+1,i-4]) +
                a4*a2*(Qc[j+2,i+4] - Qc[j-2,i+4] + Qc[j-2,i-4] - Qc[j+2,i-4]) +
                a4*a3*(Qc[j+3,i+4] - Qc[j-3,i+4] + Qc[j-3,i-4] - Qc[j+3,i-4]) +
                a4*a4*(Qc[j+4,i+4] - Qc[j-4,i+4] + Qc[j-4,i-4] - Qc[j+4,i-4])) / (dz * dx) 

        Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cpx * pxx + cpz * pzz + cpxz * pxz + dpx * qxx + dpz * qzz + dpxz * qxz)
        Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (cqx * pxx + cqz * pzz + cqxz * pxz + dqx * qxx + dqz * qzz + dqxz * qxz)

@cuda.jit
def AbsorbingBoundaryGPU(N_abc, nz_abc, nx_abc, f, A):
    x, y = cuda.grid(2)

    if x < nx_abc and y < nz_abc:
        if x < N_abc:
            f[y, x] *= A[x]
        elif x >= nx_abc - N_abc:
            f[y, x] *= A[nx_abc - 1 - x]
        if y < N_abc:
            f[y, x] *= A[y]
        elif y >= nz_abc - N_abc:
            f[y, x] *= A[nz_abc - 1 - y]
