import numpy as np
from numba import jit,prange, njit, cuda
import math

def ricker(f0, t,dt,dx,dz):
    pi = np.pi
    td  = t - 2 * np.sqrt(pi) / f0
    fcd = f0 / (np.sqrt(pi) * 3) 
    source = (1 - 2 * pi * (pi * fcd * td) * (pi * fcd * td)) * np.exp(-pi * (pi * fcd * td) * (pi * fcd * td)) * dt**2/(dx*dz)
    return source

@njit(inline = "always")
def horizontal_dampening_profiles(N_abc,nx_abc, dx, vp, f_pico, d0, dt, i, j):
    # retirar o ax e bx inicial
    ax = 1
    bx = 0
    d = 0.
    alpha = 0.
    if i < N_abc:
        points_CPML = (N_abc - i - 1.)*dx
        posicao_relativa = points_CPML / (N_abc * dx)
        d = d0/(2 * N_abc * dx) * (posicao_relativa**2) * vp[j,i]
        alpha = np.pi* f_pico * (1. - posicao_relativa**2)

    elif i >= nx_abc - N_abc:
        points_CPML = (i - nx_abc + N_abc)*dx
        posicao_relativa = points_CPML / (N_abc * dx)
        d = d0/(2 * N_abc * dx) * (posicao_relativa**2) * vp[j,i]
        alpha = np.pi* f_pico * (1. - posicao_relativa**2)

    ax = np.exp(-(d + alpha) * dt)
    if (np.abs((d + alpha)) > 1e-6):
        bx = (d / (d + alpha)) * (ax - 1.)
    
    return ax, bx

@njit(inline = "always")
def vertical_dampening_profiles(N_abc,nz_abc, dz, vp, f_pico, d0, dt, i, j):
    az = 1
    bz = 0
    d = 0.
    alpha = 0. 
    if j < N_abc:
        points_CPML = (N_abc - j - 1.)*dz
        posicao_relativa = points_CPML / (N_abc * dz)
        d = d0/(2 * N_abc * dz) * (posicao_relativa**2) * vp[j,i]
        alpha = np.pi* f_pico * (1. - posicao_relativa**2)

    elif j >= nz_abc - N_abc:
        points_CPML = (j - nz_abc + N_abc)*dz
        posicao_relativa = points_CPML / (N_abc * dz)
        d = d0/(2 * N_abc * dz) * (posicao_relativa**2) * vp[j,i]
        alpha = np.pi* f_pico * (1. - posicao_relativa**2)

    az = np.exp(-(d + alpha) * dt)
    if (np.abs((d + alpha)) > 1e-6):
        bz = (d / (d + alpha)) * (az - 1.)
       
    return az, bz


@jit(nopython=True,parallel=True)
def updateWaveEquation(Uf,Uc,vp,nz,nx,dz,dx,dt):
    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    for i in prange(4,nx-4):
        for j in prange(4,nz-4):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) + c2 * (Uc[j, i+2] + Uc[j, i-2]) +c3 * (Uc[j, i+3] + Uc[j, i-3]) +c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) + c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) + c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz) + 2 * Uc[j, i] - Uf[j, i]

    return Uf
    
@jit(nopython=True, parallel=True)
def updateWaveEquationCPML(Uf, Uc, vp, nx_abc, nz_abc, dz, dx, dt, PsixFR, PsixFL, PsizFU, PsizFD, ZetaxFR, ZetaxFL, ZetazFU, ZetazFD, N_abc):
    
    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    # Região Interior 
    for j in prange(N_abc, nz_abc - N_abc):
        for i in prange(N_abc, nx_abc - N_abc):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz) + 2 * Uc[j, i] - Uf[j, i]

    # Região Esquerda 
    for j in prange(N_abc, nz_abc - N_abc):
        for i in prange(4, N_abc):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx

            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + ZetaxFL[j, i]) + 2 * Uc[j, i] - Uf[j, i]
            
    # Região Direita
    for i in prange(nx_abc - N_abc, nx_abc - 4):
            idx = i - (nx_abc - N_abc)
            for j in range(N_abc, nz_abc - N_abc):
                pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                    c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
                pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
                psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                        a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                        a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                        a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx
    
                Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + ZetaxFR[j, idx]) + 2 * Uc[j, i] - Uf[j, i]

    # Região Superior 
    for j in prange(4, N_abc):
        for i in range(N_abc, nx_abc - N_abc):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz          

            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psiz + ZetazFU[j, i]) + 2 * Uc[j, i] - Uf[j, i]

    # Região Inferior
    for j in prange(nz_abc - N_abc, nz_abc - 4):
        jdx = j - (nz_abc - N_abc)
        for i in range(N_abc, nx_abc - N_abc):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz
            
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psiz + ZetazFD[jdx, i]) + 2 * Uc[j, i] - Uf[j, i]

    # Quina Superior Esquerda
    for i in prange(4, N_abc):
        for j in range(4, N_abc):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz   
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx
            
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + psiz + ZetaxFL[j, i] + ZetazFU[j, i]) + 2 * Uc[j, i] - Uf[j, i]

    # Quina Superior Direita 
    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in range(4, N_abc):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                        a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                        a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                        a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz          
            
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + psiz + ZetaxFR[j, idx] + ZetazFU[j, i]) + 2 * Uc[j, i] - Uf[j, i]

    # Quina Inferior Esquerda 
    for i in prange(4, N_abc):
        for j in range(nz_abc - N_abc, nz_abc - 4):
            jdx = j - (nz_abc - N_abc)

            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz
            
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + psiz + ZetaxFL[j, i] + ZetazFD[jdx, i]) + 2 * Uc[j, i] - Uf[j, i]

    # Quina Inferior Direita 
    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in range(nz_abc - N_abc, nz_abc - 4):
            jdx = j - (nz_abc - N_abc)

            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                    c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx   
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz
            
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + psiz + ZetaxFR[j, idx] + ZetazFD[jdx, i]) + 2 * Uc[j, i] - Uf[j, i]

    return Uf

@jit(nopython=True, parallel=True)
def updatePsi(PsixFR, PsixFL, PsizFU, PsizFD, nx_abc, nz_abc, Uc, dx,dz, N_abc, f_pico, d0, dt, vp):

    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    for j in prange(4, nz_abc - 4):
        for i in prange(4, N_abc):

            ax, bx = horizontal_dampening_profiles(N_abc,nx_abc, dx, vp, f_pico, d0, dt, i, j)

            px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
                a2 * (Uc[j, i+2] - Uc[j, i-2]) +
                a3 * (Uc[j, i+3] - Uc[j, i-3]) +
                a4 * (Uc[j, i+4] - Uc[j, i-4])) / dx
            
            PsixFL[j, i] = ax * PsixFL[j, i] + bx * px

    for j in prange(4, nz_abc - 4):
        for i in prange(nx_abc - N_abc, nx_abc - 4):
            idx = i - (nx_abc - N_abc)

            ax, bx = horizontal_dampening_profiles(N_abc,nx_abc, dx, vp, f_pico, d0, dt, i, j)

            px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
                a2 * (Uc[j, i+2] - Uc[j, i-2]) +
                a3 * (Uc[j, i+3] - Uc[j, i-3]) +
                a4 * (Uc[j, i+4] - Uc[j, i-4])) / dx
            
            PsixFR[j, idx] = ax * PsixFR[j, idx] + bx * px

    for j in prange(4, N_abc):
        for i in prange(4, nx_abc - 4):

            az,bz = vertical_dampening_profiles(N_abc,nz_abc, dz, vp, f_pico, d0, dt, i, j)
            
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz 
            
            PsizFU[j, i] = az * PsizFU[j, i] + bz * pz

    for j in prange(nz_abc - N_abc, nz_abc - 4):  
        jdx = j - (nz_abc - N_abc)
        for i in prange(4, nx_abc - 4):

            az,bz = vertical_dampening_profiles(N_abc,nz_abc, dz, vp, f_pico, d0, dt, i, j)

            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz 
            
            PsizFD[jdx, i] = az * PsizFD[jdx, i] + bz * pz

    return PsixFR, PsixFL, PsizFU, PsizFD

@jit(nopython=True, parallel=True)
def updateZeta(PsixFR, PsixFL, ZetaxFR, ZetaxFL,PsizFU, PsizFD, ZetazFU, ZetazFD, nx_abc, nz_abc, Uc, dx, dz, N_abc, f_pico, d0, dt, vp):

    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    for j in prange(4, nz_abc - 4):
        for i in prange(4, N_abc):
            ax, bx = horizontal_dampening_profiles(N_abc,nx_abc, dx, vp, f_pico, d0, dt, i, j)

            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
        
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx

            ZetaxFL[j, i] = ax * ZetaxFL[j, i] + bx * (pxx + psix)

    for j in prange(4, nz_abc - 4):
        for i in prange(nx_abc - N_abc, nx_abc - 4):
            idx = i - (nx_abc - N_abc) 

            ax, bx = horizontal_dampening_profiles(N_abc,nx_abc, dx, vp, f_pico, d0, dt, i, j)

            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
        
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx

            ZetaxFR[j, idx] = ax * ZetaxFR[j, idx] + bx * (pxx + psix)

    for j in prange(4, N_abc):
        for i in prange(4, nx_abc - 4):
            az,bz = vertical_dampening_profiles(N_abc,nz_abc, dz, vp, f_pico, d0, dt, i, j)
           
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)               
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz
            
            ZetazFU[j, i] = az * ZetazFU[j, i] + bz * (pzz + psiz)

    for j in prange(nz_abc - N_abc, nz_abc - 4):
        jdx = j - (nz_abc - N_abc) 
        for i in prange(4, nx_abc - 4):
            az,bz = vertical_dampening_profiles(N_abc,nz_abc, dz, vp, f_pico, d0, dt, i, j)
            
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)               
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz
            
            ZetazFD[jdx, i] = az * ZetazFD[jdx, i] + bz * (pzz + psiz) 

    return ZetaxFR, ZetaxFL, ZetazFU, ZetazFD

@jit(nopython=True,parallel=True)
def updateWaveEquationVTI(Uf, Uc, nx, nz, dt, dx, dz, vp, epsilon, delta):
    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.
    for i in prange(4,nx-4):
        for j in prange(4,nz-4):
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*(px*px)*(pz*pz)
            den = (1.0 + 2.0*epsilon[j,i])*(px*px*px*px) + (pz*pz*pz*pz) + 2.0*(1.0 + delta[j,i])*(px*px)*(pz*pz)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * pxx + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * pzz

    return Uf

@jit(nopython=True, parallel=True)
def updateWaveEquationVTICPML(Uf, Uc, dt, dx, dz, vp, epsilon, delta,
                               nx_abc, nz_abc, PsixFR, PsixFL,PsizFU,PsizFD, ZetaxFR, ZetaxFL,ZetazFU, ZetazFD, N_abc):
    
    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    # Região Interior
    for i in prange(N_abc, nx_abc - N_abc):  
        for j in prange(N_abc, nz_abc - N_abc):
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*(px*px)*(pz*pz)
            den = (1.0 + 2.0*epsilon[j,i])*(px*px*px*px) + (pz*pz*pz*pz) + 2.0*(1.0 + delta[j,i])*(px*px)*(pz*pz)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * pxx + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * pzz

    # Região Esquerda
    for i in prange(4, N_abc):
        for j in range(N_abc, nz_abc - N_abc):
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx  
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*((px + psix)**2)*((pz)**2)
            den = (1.0 + 2.0*epsilon[j,i])*((px + psix)**4) + ((pz)**4) + 2.0*(1.0 + delta[j,i])*((px + psix)**2)*((pz)**2)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * (pxx + psix + ZetaxFL[j,i]) + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * pzz          
                  
    # Região Direita
    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in range(N_abc, nz_abc - N_abc):
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx  
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*((px + psix)**2)*((pz)**2)
            den = (1.0 + 2.0*epsilon[j,i])*((px + psix)**4) + ((pz)**4) + 2.0*(1.0 + delta[j,i])*((px + psix)**2)*((pz)**2)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * (pxx + psix + ZetaxFR[j,idx]) + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * pzz          
                     
    # Região Superior
    for i in prange(N_abc, nx_abc - N_abc):
        for j in range(4, N_abc):
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz  
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*((px)**2)*((pz + psiz)**2)
            den = (1.0 + 2.0*epsilon[j,i])*((px)**4) + ((pz + psiz)**4) + 2.0*(1.0 + delta[j,i])*((px)**2)*((pz + psiz)**2)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * (pxx) + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * (pzz + psiz + ZetazFU[j,i])                   

    # Região Inferior
    for i in prange(N_abc, nx_abc - N_abc):
        for j in range(nz_abc - N_abc, nz_abc - 4):
            jdx = j - (nz_abc - N_abc)

            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz   
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*((px)**2)*((pz + psiz)**2)
            den = (1.0 + 2.0*epsilon[j,i])*((px)**4) + ((pz + psiz)**4) + 2.0*(1.0 + delta[j,i])*((px)**2)*((pz + psiz)**2)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * (pxx) + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * (pzz + psiz + ZetazFD[jdx,i])                   

    # Quina Superior Esquerda
    for i in prange(4, N_abc):
        for j in range(4, N_abc):
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx            
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz  
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*((px + psix)**2)*((pz + psiz)**2)
            den = (1.0 + 2.0*epsilon[j,i])*((px + psix)**4) + ((pz + psiz)**4) + 2.0*(1.0 + delta[j,i])*((px + psix)**2)*((pz + psiz)**2)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * (pxx + psix + ZetaxFL[j,i]) + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * (pzz + psiz + ZetazFU[j,i])                   

    # Quina Superior Direita
    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in range(4, N_abc):
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx          
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz  
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*((px + psix)**2)*((pz + psiz)**2)
            den = (1.0 + 2.0*epsilon[j,i])*((px + psix)**4) + ((pz + psiz)**4) + 2.0*(1.0 + delta[j,i])*((px + psix)**2)*((pz + psiz)**2)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * (pxx + psix + ZetaxFR[j,idx]) + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * (pzz + psiz + ZetazFU[j,i])                   
    
    # Quina Inferior Esquerda
    for i in prange(4, N_abc):
        for j in range(nz_abc - N_abc, nz_abc - 4):
            jdx = j - (nz_abc - N_abc)

            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx           
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz  
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*((px + psix)**2)*((pz + psiz)**2)
            den = (1.0 + 2.0*epsilon[j,i])*((px + psix)**4) + ((pz + psiz)**4) + 2.0*(1.0 + delta[j,i])*((px + psix)**2)*((pz + psiz)**2)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * (pxx + psix + ZetaxFL[j,i]) + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * (pzz + psiz + ZetazFD[jdx,i])                   
    
    # Quina Inferior Direita
    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in range(nz_abc - N_abc, nz_abc - 4):
            jdx = j - (nz_abc - N_abc)

            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx           
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz  
            
            num = -2.0*(epsilon[j,i]-delta[j,i])*((px + psix)**2)*((pz + psiz)**2)
            den = (1.0 + 2.0*epsilon[j,i])*((px + psix)**4) + ((pz + psiz)**4) + 2.0*(1.0 + delta[j,i])*((px + psix)**2)*((pz + psiz)**2)
                
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i]) + Sd) * (pxx + psix + ZetaxFR[j,idx]) + (vp[j, i] * vp[j, i]) * (dt * dt) *(1. + Sd) * (pzz + psiz + ZetazFD[jdx,i])                   


    return Uf

# @jit(nopython=True, parallel=True)
# def updateWaveEquationTTICPML(Uf, Uc, Qc, Qf, nx, nz, dt, dx, dz, vpz, vsz, epsilon, delta, theta,PsixF,PsizF,PsixqF,PsizqF,ZetaxF,ZetazF,ZetaxzF,ZetaxqF,ZetazqF,ZetaxzqF):
#     c0 = -205. / 72.
#     c1 = 8. / 5.
#     c2 = -1. / 5.
#     c3 = 8. / 315.
#     c4 = -1. / 560.
#     a1 = 4. / 5.
#     a2 = -1. / 5.
#     a3 = 4./105.
#     a4 = -1./280.
#     for i in prange(4, nx - 4):
#         for j in prange(4, nz - 4):
#             vpx = vpz[j, i] * np.sqrt(1 + 2*epsilon[j, i])
#             vpn = vpz[j, i] * np.sqrt(1 + 2*delta[j, i])
#             cpx = vpx**2 * np.cos(theta[j, i])**2 + vsz[j, i]**2 * np.sin(theta[j, i])**2
#             cpz = vpx**2 * np.sin(theta[j, i])**2 + vsz[j, i]**2 * np.cos(theta[j, i])**2
#             cpxz = vsz[j, i]**2 * np.sin(2 * theta[j, i]) - vpx**2 * np.sin(2 * theta[j, i])
#             dpx = vpz[j, i]**2 * np.sin(theta[j, i])**2 - vsz[j, i]**2 * np.sin(theta[j, i])**2
#             dpz = vpz[j, i]**2 * np.cos(theta[j, i])**2 - vsz[j, i]**2 * np.cos(theta[j, i])**2
#             dpxz = vpz[j, i]**2 * np.sin(2 * theta[j, i]) - vsz[j, i]**2 * np.sin(2 * theta[j, i])
#             cqx = vpn**2 * np.cos(theta[j, i])**2 - vsz[j, i]**2 * np.cos(theta[j, i])**2
#             cqz = vpn**2 * np.sin(theta[j, i])**2 - vsz[j, i]**2 * np.sin(theta[j, i])**2  
#             cqxz = vsz[j, i]**2 * np.sin(2 * theta[j, i]) - vpn**2 * np.sin(2 * theta[j, i])
#             dqx = vsz[j, i]**2 * np.cos(theta[j, i])**2 + vpz[j, i]**2 * np.sin(theta[j, i])**2  
#             dqz = vpz[j, i]**2 * np.cos(theta[j, i])**2 + vsz[j, i]**2 * np.sin(theta[j, i])**2
#             dqxz = vpz[j, i]**2 * np.sin(2 * theta[j, i]) - vsz[j, i]**2 * np.sin(2 * theta[j, i])

#             pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) + c2 * (Uc[j, i+2] + Uc[j, i-2]) +
#                    c3 * (Uc[j, i+3] + Uc[j, i-3]) + c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            
#             pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) + c2 * (Uc[j+2, i] + Uc[j-2, i]) +
#                    c3 * (Uc[j+3, i] + Uc[j-3, i]) + c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            
#             pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
#                     a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
#                     a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
#                     a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

#                     a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
#                     a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
#                     a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
#                     a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

#                     a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
#                     a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
#                     a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
#                     a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

#                     a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
#                     a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
#                     a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
#                     a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx)
            
#             qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + c2 * (Qc[j+2, i] + Qc[j-2, i]) +
#                    c3 * (Qc[j+3, i] + Qc[j-3, i]) + c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
            
#             qxx = (c0 * Qc[j, i] + c1 * (Qc[j, i+1] + Qc[j, i-1]) + c2 * (Qc[j, i+2] + Qc[j, i-2]) +
#                    c3 * (Qc[j, i+3] + Qc[j, i-3]) + c4 * (Qc[j, i+4] + Qc[j, i-4])) / (dx * dx)
            
#             qxz =  (a1*a1*(Qc[j+1,i+1] - Qc[j-1,i+1] + Qc[j-1,i-1] - Qc[j+1,i-1]) +
#                     a1*a2*(Qc[j+2,i+1] - Qc[j-2,i+1] + Qc[j-2,i-1] - Qc[j+2,i-1]) +
#                     a1*a3*(Qc[j+3,i+1] - Qc[j-3,i+1] + Qc[j-3,i-1] - Qc[j+3,i-1]) +
#                     a1*a4*(Qc[j+4,i+1] - Qc[j-4,i+1] + Qc[j-4,i-1] - Qc[j+4,i-1]) +

#                     a2*a1*(Qc[j+1,i+2] - Qc[j-1,i+2] + Qc[j-1,i-2] - Qc[j+1,i-2]) +
#                     a2*a2*(Qc[j+2,i+2] - Qc[j-2,i+2] + Qc[j-2,i-2] - Qc[j+2,i-2]) +
#                     a2*a3*(Qc[j+3,i+2] - Qc[j-3,i+2] + Qc[j-3,i-2] - Qc[j+3,i-2]) +
#                     a2*a4*(Qc[j+4,i+2] - Qc[j-4,i+2] + Qc[j-4,i-2] - Qc[j+4,i-2]) +

#                     a3*a1*(Qc[j+1,i+3] - Qc[j-1,i+3] + Qc[j-1,i-3] - Qc[j+1,i-3]) +
#                     a3*a2*(Qc[j+2,i+3] - Qc[j-2,i+3] + Qc[j-2,i-3] - Qc[j+2,i-3]) +
#                     a3*a3*(Qc[j+3,i+3] - Qc[j-3,i+3] + Qc[j-3,i-3] - Qc[j+3,i-3]) +
#                     a3*a4*(Qc[j+4,i+3] - Qc[j-4,i+3] + Qc[j-4,i-3] - Qc[j+4,i-3]) +

#                     a4*a1*(Qc[j+1,i+4] - Qc[j-1,i+4] + Qc[j-1,i-4] - Qc[j+1,i-4]) +
#                     a4*a2*(Qc[j+2,i+4] - Qc[j-2,i+4] + Qc[j-2,i-4] - Qc[j+2,i-4]) +
#                     a4*a3*(Qc[j+3,i+4] - Qc[j-3,i+4] + Qc[j-3,i-4] - Qc[j+3,i-4]) +
#                     a4*a4*(Qc[j+4,i+4] - Qc[j-4,i+4] + Qc[j-4,i-4] - Qc[j+4,i-4])) / (dz * dx) 
                       
#             psix = (a1 * (PsixF[j, i+1] - PsixF[j, i-1]) +
#                     a2 * (PsixF[j, i+2] - PsixF[j, i-2]) +
#                     a3 * (PsixF[j, i+3] - PsixF[j, i-3]) +
#                     a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / dx
            
#             psiz = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
#                     a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
#                     a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
#                     a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / dz          
            
#             psizx = (a1 * (PsizF[j, i+1] - PsizF[j, i-1]) +
#                     a2 * (PsizF[j, i+2] - PsizF[j, i-2]) +
#                     a3 * (PsizF[j, i+3] - PsizF[j, i-3]) +
#                     a4 * (PsizF[j, i+4] - PsizF[j, i-4])) /  dx

#             psiqx = (a1 * (PsixqF[j, i+1] - PsixqF[j, i-1]) +
#                     a2 * (PsixqF[j, i+2] - PsixqF[j, i-2]) +
#                     a3 * (PsixqF[j, i+3] - PsixqF[j, i-3]) +
#                     a4 * (PsixqF[j, i+4] - PsixqF[j, i-4])) / dx
            
#             psiqz = (a1 * (PsizqF[j+1, i] - PsizqF[j-1, i]) +
#                     a2 * (PsizqF[j+2, i] - PsizqF[j-2, i]) +
#                     a3 * (PsizqF[j+3, i] - PsizqF[j-3, i]) +
#                     a4 * (PsizqF[j+4, i] - PsizqF[j-4, i])) / dz

#             psiqzx = (a1 * (PsizqF[j, i+1] - PsizqF[j, i-1]) +
#                     a2 * (PsizqF[j, i+2] - PsizqF[j, i-2]) +
#                     a3 * (PsizqF[j, i+3] - PsizqF[j, i-3]) +
#                     a4 * (PsizqF[j, i+4] - PsizqF[j, i-4])) / dx
            
#             Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cpx * (pxx + psix + ZetaxF[j, i]) + cpz * (pzz + psiz + ZetazF[j, i]) + cpxz * (pxz + psizx + ZetaxzF[j,i]) + dpx * (qxx + psiqx + ZetaxqF[j, i]) + dpz * (qzz + psiqz + ZetazqF[j,i]) + dpxz * (qxz + psiqzx + ZetaxzqF[j,i]))
#             Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (cqx * (pxx + psix + ZetaxF[j, i]) + cqz * (pzz + psiz + ZetazF[j, i]) + cqxz * (pxz + psizx + ZetaxzF[j,i]) + dqx * (qxx + psiqx + ZetaxqF[j, i]) + dqz * (qzz + psiqz + ZetazqF[j,i]) + dqxz * (qxz + psiqzx + ZetaxzqF[j,i]))

#     return Uf, Qf

# @jit(nopython=True, parallel=True)
# def updatePsiTTI (PsixF, PsixqF, PsizF, PsizqF, nx_abc, nz_abc, a_z, a_x, b_z, b_x, Uc, Qc, dz, dx):

#     a1 = 672. / 840.
#     a2 = -168. / 840.
#     a3 = 32. / 840.
#     a4 = -3. / 840.

#     for j in prange(4, nz_abc - 4):
#         for i in prange(4, nx_abc - 4):
#             px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
#                 a2 * (Uc[j, i+2] - Uc[j, i-2]) +
#                 a3 * (Uc[j, i+3] - Uc[j, i-3]) +
#                 a4 * (Uc[j, i+4] - Uc[j, i-4])) / dx
#             pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
#                 a2 * (Uc[j+2, i] - Uc[j-2, i]) +
#                 a3 * (Uc[j+3, i] - Uc[j-3, i]) +
#                 a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
#             qz = (a1 * (Qc[j+1, i] - Qc[j-1, i]) +
#                 a2 * (Qc[j+2, i] - Qc[j-2, i]) +
#                 a3 * (Qc[j+3, i] - Qc[j-3, i]) +
#                 a4 * (Qc[j+4, i] - Qc[j-4, i])) / dz 
#             qx = (a1 * (Qc[j, i+1] - Qc[j, i-1]) +
#                 a2 * (Qc[j, i+2] - Qc[j, i-2]) +
#                 a3 * (Qc[j, i+3] - Qc[j, i-3]) +
#                 a4 * (Qc[j, i+4] - Qc[j, i-4])) / dx
        
#             PsixqF[j, i] = a_x[j,i] * PsixqF[j, i] + b_x[j,i] * qx
#             PsizqF[j, i] = a_z[j,i] * PsizqF[j, i] + b_z[j,i] * qz
#             PsixF[j, i] = a_x[j,i] * PsixF[j, i] + b_x[j,i] * px
#             PsizF[j, i] = a_z[j,i] * PsizF[j, i] + b_z[j,i] * pz

#     return PsixF, PsixqF, PsizF, PsizqF

@jit(nopython=True, parallel=True)
def updatePsiTTI (PsixF, PsizF, nx_abc, nz_abc,N_abc,vp,f_pico,d0, Uc, dz, dx, dt):

    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    for j in prange(4, nz_abc - 4):
        for i in prange(4, nx_abc - 4):
            ax, bx = horizontal_dampening_profiles(N_abc,nx_abc, dx, vp, f_pico, d0, dt, i, j)
            az,bz = vertical_dampening_profiles(N_abc,nz_abc, dz, vp, f_pico, d0, dt, i, j)
            px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
                a2 * (Uc[j, i+2] - Uc[j, i-2]) +
                a3 * (Uc[j, i+3] - Uc[j, i-3]) +
                a4 * (Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz

            PsixF[j, i] = ax * PsixF[j, i] + bx * px
            PsizF[j, i] = az * PsizF[j, i] + bz * pz

    return PsixF, PsizF

@jit(nopython=True, parallel=True)
def updateZetaTTI(PsixF, PsizF, ZetaxF, ZetazF, ZetaxzF, ZetazxF, nx_abc, nz_abc,N_abc,vp,f_pico,d0,dt, Uc, dz, dx):

    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    for i in prange(4, nx_abc - 4):
        for j in prange(4, nz_abc - 4):
            ax, bx = horizontal_dampening_profiles(N_abc,nx_abc, dx, vp, f_pico, d0, dt, i, j)
            az,bz = vertical_dampening_profiles(N_abc,nz_abc, dz, vp, f_pico, d0, dt, i, j)
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                    c2 * (Uc[j+2, i] + Uc[j-2, i]) +
                    c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
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
            psix = (a1 * (PsixF[j, i+1] - PsixF[j, i-1]) +
                    a2 * (PsixF[j, i+2] - PsixF[j, i-2]) +
                    a3 * (PsixF[j, i+3] - PsixF[j, i-3]) +
                    a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / dx
            psiz = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
                a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
                a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
                a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / dz
            psizx = (a1 * (PsizF[j, i+1] - PsizF[j, i-1]) +
                a2 * (PsizF[j, i+2] - PsizF[j, i-2]) +
                a3 * (PsizF[j, i+3] - PsizF[j, i-3]) +
                a4 * (PsizF[j, i+4] - PsizF[j, i-4])) / dx
            psixz = (a1 * (PsixF[j+1, i] - PsixF[j-1, i]) +
                a2 * (PsixF[j+2, i] - PsixF[j-2, i]) +
                a3 * (PsixF[j+3, i] - PsixF[j-3, i]) +
                a4 * (PsixF[j+4, i] - PsixF[j-4, i])) / dz

            ZetaxF[j, i] = ax * ZetaxF[j, i] + bx * (pxx + psix)
            ZetazF[j, i] = az * ZetazF[j, i] + bz * (pzz + psiz)
            ZetazxF[j, i] = az * ZetazxF[j, i] + bz *(pxz + psizx)
            ZetaxzF[j, i] = ax * ZetaxzF[j, i] + bx *(pxz + psixz)

    return ZetaxF, ZetazF, ZetaxzF, ZetazxF

# @jit(nopython=True, parallel=True)
# def updateZetaTTI(PsixF, PsizF, PsizqF, PsixqF, ZetaxF, ZetazF, ZetaxzF, ZetaxqF, ZetazqF, ZetaxzqF, nx_abc, nz_abc, a_z, a_x, b_z, b_x, Uc, Qc, dz, dx):

#     c0 = -205. / 72.
#     c1 = 8. / 5.
#     c2 = -1. / 5.
#     c3 = 8. / 315.
#     c4 = -1. / 560.
#     a1 = 672. / 840.
#     a2 = -168. / 840.
#     a3 = 32. / 840.
#     a4 = -3. / 840.

#     for i in prange(4, nx_abc - 4):
#         for j in prange(4, nz_abc - 4):
            
#             pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
#                 c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
#                 c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                 c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
#                     c2 * (Uc[j+2, i] + Uc[j-2, i]) +
#                     c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
#                     c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
#                 a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
#                 a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
#                 a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

#                 a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
#                 a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
#                 a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
#                 a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

#                 a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
#                 a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
#                 a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
#                 a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

#                 a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
#                 a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
#                 a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
#                 a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx)
#             qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + 
#                     c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
#                     c3 * (Qc[j+3, i] + Qc[j-3, i]) + 
#                     c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
#             qxx = (c0 * Qc[j, i] + c1 * (Qc[j, i+1] + Qc[j, i-1]) + 
#                     c2 * (Qc[j, i+2] + Qc[j, i-2]) +
#                     c3 * (Qc[j, i+3] + Qc[j, i-3]) + 
#                     c4 * (Qc[j, i+4] + Qc[j, i-4])) / (dx * dx)
#             qxz =  (a1*a1*(Qc[j+1,i+1] - Qc[j-1,i+1] + Qc[j-1,i-1] - Qc[j+1,i-1]) +
#                 a1*a2*(Qc[j+2,i+1] - Qc[j-2,i+1] + Qc[j-2,i-1] - Qc[j+2,i-1]) +
#                 a1*a3*(Qc[j+3,i+1] - Qc[j-3,i+1] + Qc[j-3,i-1] - Qc[j+3,i-1]) +
#                 a1*a4*(Qc[j+4,i+1] - Qc[j-4,i+1] + Qc[j-4,i-1] - Qc[j+4,i-1]) +

#                 a2*a1*(Qc[j+1,i+2] - Qc[j-1,i+2] + Qc[j-1,i-2] - Qc[j+1,i-2]) +
#                 a2*a2*(Qc[j+2,i+2] - Qc[j-2,i+2] + Qc[j-2,i-2] - Qc[j+2,i-2]) +
#                 a2*a3*(Qc[j+3,i+2] - Qc[j-3,i+2] + Qc[j-3,i-2] - Qc[j+3,i-2]) +
#                 a2*a4*(Qc[j+4,i+2] - Qc[j-4,i+2] + Qc[j-4,i-2] - Qc[j+4,i-2]) +

#                 a3*a1*(Qc[j+1,i+3] - Qc[j-1,i+3] + Qc[j-1,i-3] - Qc[j+1,i-3]) +
#                 a3*a2*(Qc[j+2,i+3] - Qc[j-2,i+3] + Qc[j-2,i-3] - Qc[j+2,i-3]) +
#                 a3*a3*(Qc[j+3,i+3] - Qc[j-3,i+3] + Qc[j-3,i-3] - Qc[j+3,i-3]) +
#                 a3*a4*(Qc[j+4,i+3] - Qc[j-4,i+3] + Qc[j-4,i-3] - Qc[j+4,i-3]) +

#                 a4*a1*(Qc[j+1,i+4] - Qc[j-1,i+4] + Qc[j-1,i-4] - Qc[j+1,i-4]) +
#                 a4*a2*(Qc[j+2,i+4] - Qc[j-2,i+4] + Qc[j-2,i-4] - Qc[j+2,i-4]) +
#                 a4*a3*(Qc[j+3,i+4] - Qc[j-3,i+4] + Qc[j-3,i-4] - Qc[j+3,i-4]) +
#                 a4*a4*(Qc[j+4,i+4] - Qc[j-4,i+4] + Qc[j-4,i-4] - Qc[j+4,i-4])) / (dz * dx) 
#             psix = (a1 * (PsixF[j, i+1] - PsixF[j, i-1]) +
#                     a2 * (PsixF[j, i+2] - PsixF[j, i-2]) +
#                     a3 * (PsixF[j, i+3] - PsixF[j, i-3]) +
#                     a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / dx
#             psiz = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
#                 a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
#                 a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
#                 a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / dz
#             psizx = (a1 * (PsizF[j, i+1] - PsizF[j, i-1]) +
#                 a2 * (PsizF[j, i+2] - PsizF[j, i-2]) +
#                 a3 * (PsizF[j, i+3] - PsizF[j, i-3]) +
#                 a4 * (PsizF[j, i+4] - PsizF[j, i-4])) / dx
#             psiqz = (a1 * (PsizqF[j+1, i] - PsizqF[j-1, i]) +
#                 a2 * (PsizqF[j+2, i] - PsizqF[j-2, i]) +
#                 a3 * (PsizqF[j+3, i] - PsizqF[j-3, i]) +
#                 a4 * (PsizqF[j+4, i] - PsizqF[j-4, i])) / dz
#             psiqx = (a1 * (PsixqF[j, i+1] - PsixqF[j, i-1]) +
#                 a2 * (PsixqF[j, i+2] - PsixqF[j, i-2]) +
#                 a3 * (PsixqF[j, i+3] - PsixqF[j, i-3]) +
#                 a4 * (PsixqF[j, i+4] - PsixqF[j, i-4])) / dx
#             psiqzx = (a1 * (PsizqF[j, i+1] - PsizqF[j, i-1]) +
#                 a2 * (PsizqF[j, i+2] - PsizqF[j, i-2]) +
#                 a3 * (PsizqF[j, i+3] - PsizqF[j, i-3]) +
#                 a4 * (PsizqF[j, i+4] - PsizqF[j, i-4])) / dx

#             ZetaxF[j, i] = a_x[j,i] * ZetaxF[j, i] + b_x[j,i] * (pxx + psix)
#             ZetazF[j, i] = a_z[j,i] * ZetazF[j, i] + b_z[j,i] * (pzz + psiz)
#             ZetaxzF[j, i] = a_x[j,i] * ZetaxzF[j, i] + b_x[j,i] *(pxz + psizx)
#             ZetaxzF[j, i] = a_x[j,i] * ZetaxzF[j, i] + b_x[j,i] *(pxz + psizx)
#             ZetaxqF[j, i] = a_x[j,i] * ZetaxqF[j, i] + b_x[j,i] * (qxx + psiqx)
#             ZetazqF[j, i] = a_z[j,i] * ZetazqF[j, i] + b_z[j,i] * (qzz + psiqz)
#             ZetaxzqF[j, i] = a_x[j,i] * ZetaxzqF[j, i] + b_x[j,i] *(qxz + psiqzx)

#     return ZetaxF, ZetazF, ZetaxzF, ZetaxqF, ZetazqF, ZetaxzqF

def AnalyticalModel(vpz, epsilon, delta, dt, f0, frame):
    theta = np.linspace(0, 2*np.pi, 500)
    vp = vpz * (1 + delta * np.sin(theta)**2 * np.cos(theta)**2 + epsilon * np.sin(theta)**4)
    tlag = 2 * np.sqrt(np.pi) / f0
    tt = (frame - 1) * dt - tlag
    Rp = tt * vp  
    return Rp

@jit(parallel=True, nopython=True)
def AbsorbingBoundary(N_abc, nz_abc, nx_abc, f, A):

    for i in prange(N_abc):
        for j in prange(nz_abc):
            f[j, i] *= A[i]
    
    for j in prange(N_abc):
        for i in prange(nx_abc):
            f[j, i] *= A[j]
    
    for i in prange(N_abc):
        for j in prange(nz_abc):
            f[j, nx_abc - i - 1] *= A[i]
    
    for j in prange(N_abc):
        for i in prange(nx_abc):
            f[nz_abc - j - 1, i] *= A[j]

    return f

@jit(nopython=True,parallel=True)
def updateWaveEquationTTI(Uf, Uc, nx, nz, dt, dx, dz, vp, epsilon, delta, theta):
    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.
    for i in prange(4,nx-4):
        for j in prange(4,nz-4):
            pxx = (c0 * Uc[j, i] + 
                   c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + 
                   c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
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
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            
            norm = np.sqrt(px*px + pz*pz)
            if norm > 1e-12:
                mx = px / norm
                mz = pz / norm
            else:
                mx, mz = 0.0, 0.0

            num = -2.0*(epsilon[j,i]-delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
            den = (1.0 + 2.0*epsilon[j,i])*(((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i])))**4) + ((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**4) + 2.0*(1.0 + delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
        #colocar o mais recorrente primeiro
            if abs(den) < 1e-12:
                Sd = 0.0
            else:
                Sd = num / den

            Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i])*(np.cos(theta[j,i])*np.cos(theta[j,i])) + (np.sin(theta[j,i])*np.sin(theta[j,i])) + Sd) * pxx + (vp[j, i] * vp[j, i]) * (dt * dt) *((1.+ 2.*epsilon[j,i])*(np.sin(theta[j,i])*np.sin(theta[j,i]))+ (np.cos(theta[j,i])*np.cos(theta[j,i])) + Sd) * pzz - 2. * epsilon[j,i]*(vp[j, i] * vp[j, i]) * (dt * dt) * np.sin(2.*theta[j,i]) * pxz

    return Uf

@jit(nopython=True, parallel=True)
def updateWaveEquationTTICPML(Uf, Uc, dt, dx, dz, vp, epsilon, delta,theta,
                               nx_abc, nz_abc,PsixF,PsizF, ZetaxF, ZetazF, ZetaxzF,ZetazxF):
    
    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    for i in prange(4,nx_abc-4):
        for j in prange(4,nz_abc-4):
            
                pxx = (c0 * Uc[j, i] + 
                        c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
                        c2 * (Uc[j, i+2] + Uc[j, i-2]) +
                        c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                        c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
                pzz = (c0 * Uc[j, i] + 
                        c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
                        c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                        c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
                        c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
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
                px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                        a2*(Uc[j, i+2] - Uc[j, i-2]) +
                        a3*(Uc[j, i+3] - Uc[j, i-3]) +
                        a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
                pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                        a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                        a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                        a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
                psix = (a1 * (PsixF[j, i+1] - PsixF[j, i-1]) +
                                a2 * (PsixF[j, i+2] - PsixF[j, i-2]) +
                                a3 * (PsixF[j, i+3] - PsixF[j, i-3]) +
                                a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / dx            
                psiz = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
                                a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
                                a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
                                a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / dz 
                psixz = (a1 * (PsixF[j+1, i] - PsixF[j-1, i]) +
                        a2 * (PsixF[j+2, i] - PsixF[j-2, i]) +
                        a3 * (PsixF[j+3, i] - PsixF[j-3, i]) +
                        a4 * (PsixF[j+4, i] - PsixF[j-4, i])) / dz
                psizx = (a1 * (PsizF[j, i+1] - PsizF[j, i-1]) +
                        a2 * (PsizF[j, i+2] - PsizF[j, i-2]) +
                        a3 * (PsizF[j, i+3] - PsizF[j, i-3]) +
                        a4 * (PsizF[j, i+4] - PsizF[j, i-4])) / dx
                                
                norm = np.sqrt((px)**2 + (pz)**2)
                if norm > 1e-12:
                        mx = (px) / norm
                        mz = (pz) / norm
                else:
                        mx, mz = 0.0, 0.0

                num = -2.0*(epsilon[j,i]-delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
                den = (1.0 + 2.0*epsilon[j,i])*(((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i])))**4) + ((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**4) + 2.0*(1.0 + delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
                
                if abs(den) < 1e-12:
                        Sd = 0.0
                else:
                        Sd = num / den

                Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i])*(np.cos(theta[j,i])*np.cos(theta[j,i])) + (np.sin(theta[j,i])*np.sin(theta[j,i])) + Sd) * (pxx + psix + ZetaxF[j,i]) + (vp[j, i] * vp[j, i]) * (dt * dt) *((1.+ 2.*epsilon[j,i])*(np.sin(theta[j,i])*np.sin(theta[j,i]))+ (np.cos(theta[j,i])*np.cos(theta[j,i])) + Sd) * (pzz + psiz + ZetazF[j,i]) - 2. * epsilon[j,i]*(vp[j, i] * vp[j, i]) * (dt * dt) * np.sin(2.*theta[j,i]) * (pxz)        
        
    return Uf

# @jit(nopython=True, parallel=True)
# def updateWaveEquationTTICPML(Uf, Uc, dt, dx, dz, vp, epsilon, delta,theta,
#                                nx_abc, nz_abc, PsixFR, PsixFL,PsizFU,PsizFD, ZetaxFR, ZetaxFL,ZetazFU, ZetazFD, ZetaxzFL,ZetaxzFR,ZetazxFU,ZetazxFD, N_abc):
    
#     c0 = -205. / 72.
#     c1 = 8. / 5.
#     c2 = -1. / 5.
#     c3 = 8. / 315.
#     c4 = -1. / 560.
#     a1 = 672. / 840.
#     a2 = -168. / 840.
#     a3 = 32. / 840.
#     a4 = -3. / 840.

#     # Região Interior
#     for i in prange(N_abc, nx_abc - N_abc):  
#         for j in prange(N_abc, nz_abc - N_abc):
#             pxx = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
#                    c2 * (Uc[j, i+2] + Uc[j, i-2]) +
#                    c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
#                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
#                    c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
#                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
#                     a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
#                     a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
#                     a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

#                     a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
#                     a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
#                     a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
#                     a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

#                     a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
#                     a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
#                     a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
#                     a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

#                     a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
#                     a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
#                     a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
#                     a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx)
#             px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
#                 a2*(Uc[j, i+2] - Uc[j, i-2]) +
#                 a3*(Uc[j, i+3] - Uc[j, i-3]) +
#                 a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
#             pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
#                 a2 * (Uc[j+2, i] - Uc[j-2, i]) +
#                 a3 * (Uc[j+3, i] - Uc[j-3, i]) +
#                 a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
            
#             norm = np.sqrt(px*px + pz*pz)
#             if norm > 1e-12:
#                 mx = px / norm
#                 mz = pz / norm
#             else:
#                 mx, mz = 0.0, 0.0

#             num = -2.0*(epsilon[j,i]-delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
#             den = (1.0 + 2.0*epsilon[j,i])*(((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i])))**4) + ((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**4) + 2.0*(1.0 + delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
                
#             if abs(den) < 1e-12:
#                 Sd = 0.0
#             else:
#                 Sd = num / den

#             Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i])*(np.cos(theta[j,i])*np.cos(theta[j,i])) + (np.sin(theta[j,i])*np.sin(theta[j,i])) + Sd) * pxx + (vp[j, i] * vp[j, i]) * (dt * dt) *((1.+ 2.*epsilon[j,i])*(np.sin(theta[j,i])*np.sin(theta[j,i]))+ (np.cos(theta[j,i])*np.cos(theta[j,i])) + Sd) * pzz - 2. * epsilon[j,i]*(vp[j, i] * vp[j, i]) * (dt * dt) * np.sin(2.*theta[j,i]) * pxz

#     # Região Esquerda
#     for i in prange(4, N_abc):
#         for j in range(N_abc, nz_abc - N_abc):
#             pxx = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
#                    c2 * (Uc[j, i+2] + Uc[j, i-2]) +
#                    c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
#                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
#                    c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
#                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
#                     a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
#                     a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
#                     a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

#                     a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
#                     a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
#                     a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
#                     a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

#                     a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
#                     a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
#                     a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
#                     a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

#                     a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
#                     a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
#                     a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
#                     a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx)
#             px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
#                 a2*(Uc[j, i+2] - Uc[j, i-2]) +
#                 a3*(Uc[j, i+3] - Uc[j, i-3]) +
#                 a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
#             pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
#                 a2 * (Uc[j+2, i] - Uc[j-2, i]) +
#                 a3 * (Uc[j+3, i] - Uc[j-3, i]) +
#                 a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
#             psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
#                     a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
#                     a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
#                     a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx
#             psixzL = (a1 * (PsixFL[j+1, i] - PsixFL[j-1, i]) +
#                 a2 * (PsixFL[j+2, i] - PsixFL[j-2, i]) +
#                 a3 * (PsixFL[j+3, i] - PsixFL[j-3, i]) +
#                 a4 * (PsixFL[j+4, i] - PsixFL[j-4, i])) / dz
                        
#             norm = np.sqrt((px + psix)**2 + (pz**2))
#             if norm > 1e-12:
#                 mx = (px + psix) / norm
#                 mz = pz / norm
#             else:
#                 mx, mz = 0.0, 0.0

#             num = -2.0*(epsilon[j,i]-delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
#             den = (1.0 + 2.0*epsilon[j,i])*(((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i])))**4) + ((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**4) + 2.0*(1.0 + delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
                
#             if abs(den) < 1e-12:
#                 Sd = 0.0
#             else:
#                 Sd = num / den

#             Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i])*(np.cos(theta[j,i])*np.cos(theta[j,i])) + (np.sin(theta[j,i])*np.sin(theta[j,i])) + Sd) * (pxx + psix + ZetaxFL[j,i]) + (vp[j, i] * vp[j, i]) * (dt * dt) *((1.+ 2.*epsilon[j,i])*(np.sin(theta[j,i])*np.sin(theta[j,i]))+ (np.cos(theta[j,i])*np.cos(theta[j,i])) + Sd) * pzz - 2. * epsilon[j,i]*(vp[j, i] * vp[j, i]) * (dt * dt) * np.sin(2.*theta[j,i]) * (pxz + psixzL + ZetaxzFL[j, i])

#     # Região Direita
#     for i in prange(nx_abc - N_abc, nx_abc - 4):
#         idx = i - (nx_abc - N_abc)
#         for j in range(N_abc, nz_abc - N_abc):
            
#             pxx = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
#                    c2 * (Uc[j, i+2] + Uc[j, i-2]) +
#                    c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
#                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
#                    c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
#                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
#                     a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
#                     a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
#                     a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

#                     a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
#                     a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
#                     a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
#                     a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

#                     a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
#                     a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
#                     a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
#                     a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

#                     a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
#                     a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
#                     a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
#                     a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx)
#             px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
#                 a2*(Uc[j, i+2] - Uc[j, i-2]) +
#                 a3*(Uc[j, i+3] - Uc[j, i-3]) +
#                 a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
#             pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
#                 a2 * (Uc[j+2, i] - Uc[j-2, i]) +
#                 a3 * (Uc[j+3, i] - Uc[j-3, i]) +
#                 a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
#             psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
#                     a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
#                     a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
#                     a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx  
#             psixzR = (a1 * (PsixFR[j+1, idx] - PsixFR[j-1, idx]) +
#                 a2 * (PsixFR[j+2, idx] - PsixFR[j-2, idx]) +
#                 a3 * (PsixFR[j+3, idx] - PsixFR[j-3, idx]) +
#                 a4 * (PsixFR[j+4, idx] - PsixFR[j-4, idx])) / dz
                        
#             norm = np.sqrt((px + psix)**2 + (pz**2))
#             if norm > 1e-12:
#                 mx = (px + psix) / norm
#                 mz = pz / norm
#             else:
#                 mx, mz = 0.0, 0.0

#             num = -2.0*(epsilon[j,i]-delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
#             den = (1.0 + 2.0*epsilon[j,i])*(((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i])))**4) + ((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**4) + 2.0*(1.0 + delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
                
#             if abs(den) < 1e-12:
#                 Sd = 0.0
#             else:
#                 Sd = num / den

#             Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i])*(np.cos(theta[j,i])*np.cos(theta[j,i])) + (np.sin(theta[j,i])*np.sin(theta[j,i])) + Sd) * (pxx + psix + ZetaxFR[j,idx]) + (vp[j, i] * vp[j, i]) * (dt * dt) *((1.+ 2.*epsilon[j,i])*(np.sin(theta[j,i])*np.sin(theta[j,i]))+ (np.cos(theta[j,i])*np.cos(theta[j,i])) + Sd) * pzz - 2. * epsilon[j,i]*(vp[j, i] * vp[j, i]) * (dt * dt) * np.sin(2.*theta[j,i]) * (pxz + psixzR + ZetaxzFR[j, idx])        
                     
#     # Região Superior
#     for i in prange(N_abc, nx_abc - N_abc):
#         for j in range(4, N_abc):
            
#             pxx = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
#                    c2 * (Uc[j, i+2] + Uc[j, i-2]) +
#                    c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
#                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
#                    c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
#                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
#                     a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
#                     a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
#                     a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

#                     a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
#                     a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
#                     a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
#                     a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

#                     a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
#                     a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
#                     a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
#                     a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

#                     a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
#                     a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
#                     a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
#                     a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx)
#             px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
#                 a2*(Uc[j, i+2] - Uc[j, i-2]) +
#                 a3*(Uc[j, i+3] - Uc[j, i-3]) +
#                 a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
#             pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
#                 a2 * (Uc[j+2, i] - Uc[j-2, i]) +
#                 a3 * (Uc[j+3, i] - Uc[j-3, i]) +
#                 a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
#             psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
#                     a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
#                     a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
#                     a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz 
#             psizxU = (a1 * (PsizFU[j, i+1] - PsizFU[j, i-1]) +
#                 a2 * (PsizFU[j, i+2] - PsizFU[j, i-2]) +
#                 a3 * (PsizFU[j, i+3] - PsizFU[j, i-3]) +
#                 a4 * (PsizFU[j, i+4] - PsizFU[j, i-4])) / dx
                        
#             norm = np.sqrt((px)**2 + (pz + psiz)**2)
#             if norm > 1e-12:
#                 mx = px / norm
#                 mz = (pz + psiz) / norm
#             else:
#                 mx, mz = 0.0, 0.0

#             num = -2.0*(epsilon[j,i]-delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
#             den = (1.0 + 2.0*epsilon[j,i])*(((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i])))**4) + ((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**4) + 2.0*(1.0 + delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
                
#             if abs(den) < 1e-12:
#                 Sd = 0.0
#             else:
#                 Sd = num / den

#             Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i])*(np.cos(theta[j,i])*np.cos(theta[j,i])) + (np.sin(theta[j,i])*np.sin(theta[j,i])) + Sd) * (pxx) + (vp[j, i] * vp[j, i]) * (dt * dt) *((1.+ 2.*epsilon[j,i])*(np.sin(theta[j,i])*np.sin(theta[j,i]))+ (np.cos(theta[j,i])*np.cos(theta[j,i])) + Sd) * (pzz + psiz + ZetazFU[j,i]) - 2. * epsilon[j,i]*(vp[j, i] * vp[j, i]) * (dt * dt) * np.sin(2.*theta[j,i]) * (pxz + psizxU + ZetazxFU[j, i])        
             
#     # Região Inferior
#     for i in prange(N_abc, nx_abc - N_abc):
#         for j in range(nz_abc - N_abc, nz_abc - 4):
#             jdx = j - (nz_abc - N_abc)
            
#             pxx = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
#                    c2 * (Uc[j, i+2] + Uc[j, i-2]) +
#                    c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
#                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
#                    c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
#                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
#                     a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
#                     a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
#                     a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

#                     a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
#                     a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
#                     a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
#                     a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

#                     a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
#                     a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
#                     a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
#                     a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

#                     a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
#                     a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
#                     a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
#                     a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx)
#             px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
#                 a2*(Uc[j, i+2] - Uc[j, i-2]) +
#                 a3*(Uc[j, i+3] - Uc[j, i-3]) +
#                 a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
#             pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
#                 a2 * (Uc[j+2, i] - Uc[j-2, i]) +
#                 a3 * (Uc[j+3, i] - Uc[j-3, i]) +
#                 a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
#             psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
#                     a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
#                     a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
#                     a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz 
#             psizxD = (a1 * (PsizFD[jdx, i+1] - PsizFD[jdx, i-1]) +
#                 a2 * (PsizFD[jdx, i+2] - PsizFD[jdx, i-2]) +
#                 a3 * (PsizFD[jdx, i+3] - PsizFD[jdx, i-3]) +
#                 a4 * (PsizFD[jdx, i+4] - PsizFD[jdx, i-4])) / dx
                        
#             norm = np.sqrt((px)**2 + (pz + psiz)**2)
#             if norm > 1e-12:
#                 mx = px / norm
#                 mz = (pz + psiz) / norm
#             else:
#                 mx, mz = 0.0, 0.0

#             num = -2.0*(epsilon[j,i]-delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
#             den = (1.0 + 2.0*epsilon[j,i])*(((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i])))**4) + ((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**4) + 2.0*(1.0 + delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
                
#             if abs(den) < 1e-12:
#                 Sd = 0.0
#             else:
#                 Sd = num / den

#             Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i])*(np.cos(theta[j,i])*np.cos(theta[j,i])) + (np.sin(theta[j,i])*np.sin(theta[j,i])) + Sd) * (pxx) + (vp[j, i] * vp[j, i]) * (dt * dt) *((1.+ 2.*epsilon[j,i])*(np.sin(theta[j,i])*np.sin(theta[j,i]))+ (np.cos(theta[j,i])*np.cos(theta[j,i])) + Sd) * (pzz + psiz + ZetazFD[jdx,i]) - 2. * epsilon[j,i]*(vp[j, i] * vp[j, i]) * (dt * dt) * np.sin(2.*theta[j,i]) * (pxz + psizxD + ZetazxFD[jdx, i])        
             
#     # Quina Superior Esquerda
#     for i in prange(4, N_abc):
#         for j in range(4, N_abc):
#             pxx = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
#                    c2 * (Uc[j, i+2] + Uc[j, i-2]) +
#                    c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
#                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
#                    c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
#                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
#                     a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
#                     a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
#                     a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

#                     a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
#                     a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
#                     a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
#                     a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

#                     a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
#                     a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
#                     a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
#                     a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

#                     a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
#                     a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
#                     a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
#                     a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx)
#             px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
#                 a2*(Uc[j, i+2] - Uc[j, i-2]) +
#                 a3*(Uc[j, i+3] - Uc[j, i-3]) +
#                 a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
#             pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
#                 a2 * (Uc[j+2, i] - Uc[j-2, i]) +
#                 a3 * (Uc[j+3, i] - Uc[j-3, i]) +
#                 a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
#             psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
#                     a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
#                     a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
#                     a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx            
#             psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
#                     a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
#                     a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
#                     a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz 
#             psixzL = (a1 * (PsixFL[j+1, i] - PsixFL[j-1, i]) +
#                 a2 * (PsixFL[j+2, i] - PsixFL[j-2, i]) +
#                 a3 * (PsixFL[j+3, i] - PsixFL[j-3, i]) +
#                 a4 * (PsixFL[j+4, i] - PsixFL[j-4, i])) / dz
#             psizxU = (a1 * (PsizFU[j, i+1] - PsizFU[j, i-1]) +
#                 a2 * (PsizFU[j, i+2] - PsizFU[j, i-2]) +
#                 a3 * (PsizFU[j, i+3] - PsizFU[j, i-3]) +
#                 a4 * (PsizFU[j, i+4] - PsizFU[j, i-4])) / dx
                        
#             norm = np.sqrt((px + psix)**2 + (pz + psiz)**2)
#             if norm > 1e-12:
#                 mx = (px + psix) / norm
#                 mz = (pz + psiz) / norm
#             else:
#                 mx, mz = 0.0, 0.0

#             num = -2.0*(epsilon[j,i]-delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
#             den = (1.0 + 2.0*epsilon[j,i])*(((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i])))**4) + ((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**4) + 2.0*(1.0 + delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
                
#             if abs(den) < 1e-12:
#                 Sd = 0.0
#             else:
#                 Sd = num / den

#             Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i])*(np.cos(theta[j,i])*np.cos(theta[j,i])) + (np.sin(theta[j,i])*np.sin(theta[j,i])) + Sd) * (pxx + psix + ZetaxFL[j,i]) + (vp[j, i] * vp[j, i]) * (dt * dt) *((1.+ 2.*epsilon[j,i])*(np.sin(theta[j,i])*np.sin(theta[j,i]))+ (np.cos(theta[j,i])*np.cos(theta[j,i])) + Sd) * (pzz + psiz + ZetazFU[j,i]) - 2. * epsilon[j,i]*(vp[j, i] * vp[j, i]) * (dt * dt) * np.sin(2.*theta[j,i]) * (pxz + psixzL + psizxU + ZetaxzFL[j, i] + ZetazxFU[j, i])        
             
#     # Quina Superior Direita
#     for i in prange(nx_abc - N_abc, nx_abc - 4):
#         idx = i - (nx_abc - N_abc)
#         for j in range(4, N_abc):

#             pxx = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
#                    c2 * (Uc[j, i+2] + Uc[j, i-2]) +
#                    c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
#                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
#                    c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
#                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
#                     a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
#                     a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
#                     a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

#                     a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
#                     a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
#                     a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
#                     a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

#                     a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
#                     a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
#                     a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
#                     a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

#                     a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
#                     a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
#                     a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
#                     a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx)
#             px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
#                 a2*(Uc[j, i+2] - Uc[j, i-2]) +
#                 a3*(Uc[j, i+3] - Uc[j, i-3]) +
#                 a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
#             pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
#                 a2 * (Uc[j+2, i] - Uc[j-2, i]) +
#                 a3 * (Uc[j+3, i] - Uc[j-3, i]) +
#                 a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
#             psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
#                     a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
#                     a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
#                     a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx          
#             psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
#                     a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
#                     a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
#                     a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz 
#             psixzR = (a1 * (PsixFR[j+1, idx] - PsixFR[j-1, idx]) +
#                 a2 * (PsixFR[j+2, idx] - PsixFR[j-2, idx]) +
#                 a3 * (PsixFR[j+3, idx] - PsixFR[j-3, idx]) +
#                 a4 * (PsixFR[j+4, idx] - PsixFR[j-4, idx])) / dz
#             psizxU = (a1 * (PsizFU[j, i+1] - PsizFU[j, i-1]) +
#                 a2 * (PsizFU[j, i+2] - PsizFU[j, i-2]) +
#                 a3 * (PsizFU[j, i+3] - PsizFU[j, i-3]) +
#                 a4 * (PsizFU[j, i+4] - PsizFU[j, i-4])) / dx
                        
#             norm = np.sqrt((px + psix)**2 + (pz + psiz)**2)
#             if norm > 1e-12:
#                 mx = (px + psix) / norm
#                 mz = (pz + psiz) / norm
#             else:
#                 mx, mz = 0.0, 0.0

#             num = -2.0*(epsilon[j,i]-delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
#             den = (1.0 + 2.0*epsilon[j,i])*(((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i])))**4) + ((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**4) + 2.0*(1.0 + delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
                
#             if abs(den) < 1e-12:
#                 Sd = 0.0
#             else:
#                 Sd = num / den

#             Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i])*(np.cos(theta[j,i])*np.cos(theta[j,i])) + (np.sin(theta[j,i])*np.sin(theta[j,i])) + Sd) * (pxx + psix + ZetaxFR[j,idx]) + (vp[j, i] * vp[j, i]) * (dt * dt) *((1.+ 2.*epsilon[j,i])*(np.sin(theta[j,i])*np.sin(theta[j,i]))+ (np.cos(theta[j,i])*np.cos(theta[j,i])) + Sd) * (pzz + psiz + ZetazFU[j,i]) - 2. * epsilon[j,i]*(vp[j, i] * vp[j, i]) * (dt * dt) * np.sin(2.*theta[j,i]) * (pxz + psixzR + psizxU + ZetaxzFR[j, idx] + ZetazxFU[j, i])                          
    
#     # Quina Inferior Esquerda
#     for i in prange(4, N_abc):
#         for j in range(nz_abc - N_abc, nz_abc - 4):
#             jdx = j - (nz_abc - N_abc)

#             pxx = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
#                    c2 * (Uc[j, i+2] + Uc[j, i-2]) +
#                    c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
#                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
#                    c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
#                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
#                     a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
#                     a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
#                     a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

#                     a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
#                     a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
#                     a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
#                     a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

#                     a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
#                     a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
#                     a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
#                     a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

#                     a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
#                     a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
#                     a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
#                     a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx)
#             px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
#                 a2*(Uc[j, i+2] - Uc[j, i-2]) +
#                 a3*(Uc[j, i+3] - Uc[j, i-3]) +
#                 a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
#             pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
#                 a2 * (Uc[j+2, i] - Uc[j-2, i]) +
#                 a3 * (Uc[j+3, i] - Uc[j-3, i]) +
#                 a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
#             psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
#                     a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
#                     a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
#                     a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx           
#             psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
#                     a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
#                     a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
#                     a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz 
#             psixzL = (a1 * (PsixFL[j+1, i] - PsixFL[j-1, i]) +
#                 a2 * (PsixFL[j+2, i] - PsixFL[j-2, i]) +
#                 a3 * (PsixFL[j+3, i] - PsixFL[j-3, i]) +
#                 a4 * (PsixFL[j+4, i] - PsixFL[j-4, i])) / dz
#             psizxD = (a1 * (PsizFD[jdx, i+1] - PsizFD[jdx, i-1]) +
#                 a2 * (PsizFD[jdx, i+2] - PsizFD[jdx, i-2]) +
#                 a3 * (PsizFD[jdx, i+3] - PsizFD[jdx, i-3]) +
#                 a4 * (PsizFD[jdx, i+4] - PsizFD[jdx, i-4])) / dx
                        
#             norm = np.sqrt((px + psix)**2 + (pz + psiz)**2)
#             if norm > 1e-12:
#                 mx = (px + psix) / norm
#                 mz = (pz + psiz) / norm
#             else:
#                 mx, mz = 0.0, 0.0

#             num = -2.0*(epsilon[j,i]-delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
#             den = (1.0 + 2.0*epsilon[j,i])*(((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i])))**4) + ((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**4) + 2.0*(1.0 + delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
                
#             if abs(den) < 1e-12:
#                 Sd = 0.0
#             else:
#                 Sd = num / den

#             Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i])*(np.cos(theta[j,i])*np.cos(theta[j,i])) + (np.sin(theta[j,i])*np.sin(theta[j,i])) + Sd) * (pxx + psix + ZetaxFL[j,i]) + (vp[j, i] * vp[j, i]) * (dt * dt) *((1.+ 2.*epsilon[j,i])*(np.sin(theta[j,i])*np.sin(theta[j,i]))+ (np.cos(theta[j,i])*np.cos(theta[j,i])) + Sd) * (pzz + psiz + ZetazFD[jdx,i]) - 2. * epsilon[j,i]*(vp[j, i] * vp[j, i]) * (dt * dt) * np.sin(2.*theta[j,i]) * (pxz + psixzL + psizxD + ZetaxzFL[j, i] + ZetazxFD[jdx, i])                          

#     # Quina Inferior Direita
#     for i in prange(nx_abc - N_abc, nx_abc - 4):
#         idx = i - (nx_abc - N_abc)
#         for j in range(nz_abc - N_abc, nz_abc - 4):
#             jdx = j - (nz_abc - N_abc)

#             pxx = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j, i+1] + Uc[j, i-1]) + 
#                    c2 * (Uc[j, i+2] + Uc[j, i-2]) +
#                    c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + 
#                    c1 * (Uc[j+1, i] + Uc[j-1, i]) + 
#                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
#                    c3 * (Uc[j+3, i] + Uc[j-3, i]) + 
#                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
#                     a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
#                     a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
#                     a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

#                     a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
#                     a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
#                     a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
#                     a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

#                     a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
#                     a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
#                     a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
#                     a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

#                     a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
#                     a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
#                     a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
#                     a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx)
#             px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
#                 a2*(Uc[j, i+2] - Uc[j, i-2]) +
#                 a3*(Uc[j, i+3] - Uc[j, i-3]) +
#                 a4*(Uc[j, i+4] - Uc[j, i-4])) / dx
#             pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
#                 a2 * (Uc[j+2, i] - Uc[j-2, i]) +
#                 a3 * (Uc[j+3, i] - Uc[j-3, i]) +
#                 a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz
#             psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
#                     a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
#                     a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
#                     a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx           
#             psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
#                     a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
#                     a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
#                     a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz 
#             psixzR = (a1 * (PsixFR[j+1, idx] - PsixFR[j-1, idx]) +
#                 a2 * (PsixFR[j+2, idx] - PsixFR[j-2, idx]) +
#                 a3 * (PsixFR[j+3, idx] - PsixFR[j-3, idx]) +
#                 a4 * (PsixFR[j+4, idx] - PsixFR[j-4, idx])) / dz
#             psizxD = (a1 * (PsizFD[jdx, i+1] - PsizFD[jdx, i-1]) +
#                 a2 * (PsizFD[jdx, i+2] - PsizFD[jdx, i-2]) +
#                 a3 * (PsizFD[jdx, i+3] - PsizFD[jdx, i-3]) +
#                 a4 * (PsizFD[jdx, i+4] - PsizFD[jdx, i-4])) / dx
                        
#             norm = np.sqrt((px + psix)**2 + (pz + psiz)**2)
#             if norm > 1e-12:
#                 mx = (px + psix) / norm
#                 mz = (pz + psiz) / norm
#             else:
#                 mx, mz = 0.0, 0.0

#             num = -2.0*(epsilon[j,i]-delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
#             den = (1.0 + 2.0*epsilon[j,i])*(((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i])))**4) + ((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**4) + 2.0*(1.0 + delta[j,i])*((mx*np.cos(theta[j,i]) - mz*np.sin(theta[j,i]))**2)*((mx*np.sin(theta[j,i]) + mz*np.cos(theta[j,i]))**2)
                
#             if abs(den) < 1e-12:
#                 Sd = 0.0
#             else:
#                 Sd = num / den

#             Uf[j, i] = 2. * Uc[j, i] - Uf[j, i] + (vp[j, i] * vp[j, i]) * (dt * dt) * ((1.+ 2.*epsilon[j,i])*(np.cos(theta[j,i])*np.cos(theta[j,i])) + (np.sin(theta[j,i])*np.sin(theta[j,i])) + Sd) * (pxx + psix + ZetaxFR[j,idx]) + (vp[j, i] * vp[j, i]) * (dt * dt) *((1.+ 2.*epsilon[j,i])*(np.sin(theta[j,i])*np.sin(theta[j,i]))+ (np.cos(theta[j,i])*np.cos(theta[j,i])) + Sd) * (pzz + psiz + ZetazFD[jdx,i]) - 2. * epsilon[j,i]*(vp[j, i] * vp[j, i]) * (dt * dt) * np.sin(2.*theta[j,i]) * (pxz + psixzR + psizxD + ZetaxzFR[j,idx] + ZetazxFD[jdx, i])                          

#     return Uf
 

# @jit(nopython=True, parallel=True)
# def updateZetaTTI(PsixFL, PsixFR, PsizFU, PsizFD, ZetaxzFL, ZetaxzFR, ZetazxFU, ZetazxFD, nx_abc, nz_abc, a_x, b_x, a_z, b_z, Uc, dx, dz, N_abc):
#     a1 = 672. / 840.
#     a2 = -168. / 840.
#     a3 = 32. / 840.
#     a4 = -3. / 840.
    
#     for i in prange(4, N_abc):
#         for j in prange(4, nz_abc - 4):

#             pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
#                 a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
#                 a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
#                 a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

#                 a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
#                 a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
#                 a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
#                 a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

#                 a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
#                 a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
#                 a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
#                 a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

#                 a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
#                 a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
#                 a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
#                 a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx) 
            
#             psixzL = (a1 * (PsixFL[j+1, i] - PsixFL[j-1, i]) +
#                 a2 * (PsixFL[j+2, i] - PsixFL[j-2, i]) +
#                 a3 * (PsixFL[j+3, i] - PsixFL[j-3, i]) +
#                 a4 * (PsixFL[j+4, i] - PsixFL[j-4, i])) / dz
            
#             ZetaxzFL[j, i] = a_z[j,i] * ZetaxzFL[j, i] + b_z[j,i] * (pxz + psixzL)

#     for i in prange(nx_abc - N_abc, nx_abc - 4):
#         idx = i - (nx_abc - N_abc)  
#         for j in prange(4, nz_abc - 4):

#             pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
#                 a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
#                 a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
#                 a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

#                 a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
#                 a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
#                 a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
#                 a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

#                 a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
#                 a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
#                 a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
#                 a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

#                 a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
#                 a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
#                 a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
#                 a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx) 
            
#             psixzR = (a1 * (PsixFR[j+1, idx] - PsixFR[j-1, idx]) +
#                 a2 * (PsixFR[j+2, idx] - PsixFR[j-2, idx]) +
#                 a3 * (PsixFR[j+3, idx] - PsixFR[j-3, idx]) +
#                 a4 * (PsixFR[j+4, idx] - PsixFR[j-4, idx])) / dz
            
#             ZetaxzFR[j, idx] = a_z[j,i] * ZetaxzFR[j, idx] + b_z[j,i] * (pxz + psixzR)

#     for j in prange(4, N_abc):
#         for i in prange(4, nx_abc - 4):

#             pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
#                 a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
#                 a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
#                 a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

#                 a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
#                 a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
#                 a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
#                 a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

#                 a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
#                 a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
#                 a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
#                 a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

#                 a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
#                 a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
#                 a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
#                 a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx) 
            
#             psizxU = (a1 * (PsizFU[j, i+1] - PsizFU[j, i-1]) +
#                 a2 * (PsizFU[j, i+2] - PsizFU[j, i-2]) +
#                 a3 * (PsizFU[j, i+3] - PsizFU[j, i-3]) +
#                 a4 * (PsizFU[j, i+4] - PsizFU[j, i-4])) / dx
            
#             ZetazxFU[j, i] = a_x[j,i] * ZetazxFU[j, i] + b_x[j,i] * (pxz + psizxU)

#     for j in prange(nz_abc - N_abc, nz_abc - 4):
#         jdx = j - (nz_abc - N_abc)  
#         for i in prange(4, nx_abc - 4):

#             pxz = (a1*a1*(Uc[j+1,i+1] - Uc[j-1,i+1] + Uc[j-1,i-1] - Uc[j+1,i-1]) +
#                 a1*a2*(Uc[j+2,i+1] - Uc[j-2,i+1] + Uc[j-2,i-1] - Uc[j+2,i-1]) +
#                 a1*a3*(Uc[j+3,i+1] - Uc[j-3,i+1] + Uc[j-3,i-1] - Uc[j+3,i-1]) +
#                 a1*a4*(Uc[j+4,i+1] - Uc[j-4,i+1] + Uc[j-4,i-1] - Uc[j+4,i-1]) +

#                 a2*a1*(Uc[j+1,i+2] - Uc[j-1,i+2] + Uc[j-1,i-2] - Uc[j+1,i-2]) +
#                 a2*a2*(Uc[j+2,i+2] - Uc[j-2,i+2] + Uc[j-2,i-2] - Uc[j+2,i-2]) +
#                 a2*a3*(Uc[j+3,i+2] - Uc[j-3,i+2] + Uc[j-3,i-2] - Uc[j+3,i-2]) +
#                 a2*a4*(Uc[j+4,i+2] - Uc[j-4,i+2] + Uc[j-4,i-2] - Uc[j+4,i-2]) +

#                 a3*a1*(Uc[j+1,i+3] - Uc[j-1,i+3] + Uc[j-1,i-3] - Uc[j+1,i-3]) +
#                 a3*a2*(Uc[j+2,i+3] - Uc[j-2,i+3] + Uc[j-2,i-3] - Uc[j+2,i-3]) +
#                 a3*a3*(Uc[j+3,i+3] - Uc[j-3,i+3] + Uc[j-3,i-3] - Uc[j+3,i-3]) +
#                 a3*a4*(Uc[j+4,i+3] - Uc[j-4,i+3] + Uc[j-4,i-3] - Uc[j+4,i-3]) +

#                 a4*a1*(Uc[j+1,i+4] - Uc[j-1,i+4] + Uc[j-1,i-4] - Uc[j+1,i-4]) +
#                 a4*a2*(Uc[j+2,i+4] - Uc[j-2,i+4] + Uc[j-2,i-4] - Uc[j+2,i-4]) +
#                 a4*a3*(Uc[j+3,i+4] - Uc[j-3,i+4] + Uc[j-3,i-4] - Uc[j+3,i-4]) +
#                 a4*a4*(Uc[j+4,i+4] - Uc[j-4,i+4] + Uc[j-4,i-4] - Uc[j+4,i-4])) / (dz * dx) 
            
#             psizxD = (a1 * (PsizFD[jdx, i+1] - PsizFD[jdx, i-1]) +
#                 a2 * (PsizFD[jdx, i+2] - PsizFD[jdx, i-2]) +
#                 a3 * (PsizFD[jdx, i+3] - PsizFD[jdx, i-3]) +
#                 a4 * (PsizFD[jdx, i+4] - PsizFD[jdx, i-4])) / dx
            
#             ZetazxFD[jdx, i] = a_x[j,i] * ZetazxFD[jdx, i] + b_x[j,i] * (pxz + psizxD)


#     return  ZetaxzFL, ZetaxzFR, ZetazxFU, ZetazxFD 
