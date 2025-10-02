import numpy as np
from numba import jit,njit,prange

def ricker(f0, t):
    pi = np.pi
    td  = t - 2 * np.sqrt(pi) / f0
    fcd = f0 / (np.sqrt(pi) * 3) 
    source = (1 - 2 * pi * (pi * fcd * td) * (pi * fcd * td)) * np.exp(-pi * (pi * fcd * td) * (pi * fcd * td))
    return source


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
    a1 = 4. / 5.
    a2 = -1. / 5.
    a3 = 4. / 105.
    a4 = -1. / 280.

    # Região Interior 
    for i in prange(N_abc, nx_abc - N_abc):
        for j in range(N_abc, nz_abc - N_abc):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz) + 2 * Uc[j, i] - Uf[j, i]

    # Região Esquerda 
    for i in prange(4, N_abc):
        for j in range(N_abc, nz_abc - N_abc):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                   c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                   c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                   c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                   c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / (2 * dx)

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
                        a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / (2 * dx)
    
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
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / (2*dz)          

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
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / (2*dz)
            
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
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / (2*dz)   
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / (2 * dx)
            
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
                        a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / (2 * dx)
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / (2*dz)          
            
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
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / (2*dx)
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / (2*dz)
            
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
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / (2 * dx)   
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / (2*dz)
            
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + psiz + ZetaxFR[j, idx] + ZetazFD[jdx, i]) + 2 * Uc[j, i] - Uf[j, i]

    return Uf

@jit(nopython=True, parallel=True)
def updatePsiRL(PsixFR, PsixFL, nx_abc, nz_abc, a_x, b_x, Uc, dx, N_abc):

    a1 = 4. / 5.
    a2 = -1. / 5.
    a3 = 4. / 105.
    a4 = -1. / 280.

    for i in prange(4, N_abc):
        idx = N_abc - i
        for j in prange(4, nz_abc - 4):

            px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
                a2 * (Uc[j, i+2] - Uc[j, i-2]) +
                a3 * (Uc[j, i+3] - Uc[j, i-3]) +
                a4 * (Uc[j, i+4] - Uc[j, i-4])) / (2 * dx)
            
            PsixFL[j, i] = a_x[idx] * PsixFL[j, i] + b_x[idx] * px

    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in prange(4, nz_abc - 4):

            px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
                a2 * (Uc[j, i+2] - Uc[j, i-2]) +
                a3 * (Uc[j, i+3] - Uc[j, i-3]) +
                a4 * (Uc[j, i+4] - Uc[j, i-4])) / (2 * dx)
            
            PsixFR[j, idx] = a_x[idx] * PsixFR[j, idx] + b_x[idx] * px

    return PsixFR, PsixFL

@jit(nopython=True, parallel=True)
def updatePsiUD (PsizFU, PsizFD, nx_abc, nz_abc, a_z, b_z, Uc, dz,N_abc):

    a1 = 4. / 5.
    a2 = -1. / 5.
    a3 = 4. / 105.
    a4 = -1. / 280.

    for i in prange(4, nx_abc - 4):
        for j in prange(4, N_abc):
            jdx = N_abc - j

            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / (2 * dz) 
            
            PsizFU[j, i] = a_z[jdx] * PsizFU[j, i] + b_z[jdx] * pz

    for i in prange(4, nx_abc - 4):
        for j in prange(nz_abc - N_abc, nz_abc - 4):
            jdx = j - (nz_abc - N_abc)

            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / (2 * dz) 
            
            PsizFD[jdx, i] = a_z[jdx] * PsizFD[jdx, i] + b_z[jdx] * pz

    return PsizFU, PsizFD

@jit(nopython=True, parallel=True)
def updateZetaRL(PsixFR, PsixFL, ZetaxFR, ZetaxFL, nx_abc, nz_abc, a_x, b_x, Uc, dx, N_abc):

    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 4. / 5.
    a2 = -1. / 5.
    a3 = 4. / 105.
    a4 = -1. / 280.

    for i in prange(4, N_abc - 4):
        idx = N_abc - i 
        for j in prange(4, nz_abc - 4):
            
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
        
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / (2 * dx)

            ZetaxFL[j, i] = a_x[idx] * ZetaxFL[j, i] + b_x[idx] * (pxx + psix)

    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc) 
        for j in prange(4, nz_abc - 4):
                
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
        
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / (2 * dx)

            ZetaxFR[j, idx] = a_x[idx] * ZetaxFR[j, idx] + b_x[idx] * (pxx + psix)  

    return ZetaxFR, ZetaxFL

@jit(nopython=True, parallel=True)
def updateZetaUD(PsizFU, PsizFD, ZetazFU, ZetazFD, nx_abc, nz_abc, a_z, b_z, Uc, dz, N_abc):

    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 4. / 5.
    a2 = -1. / 5.
    a3 = 4. / 105.
    a4 = -1. / 280.

    for i in prange(4, nx_abc - 4):
        for j in prange(4, N_abc- 4):
            jdx = N_abc - j 
                
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)               
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / (2*dz)
            
            ZetazFU[j, i] = a_z[jdx] * ZetazFU[j, i] + b_z[jdx] * (pzz + psiz)

    for i in prange(4, nx_abc - 4):
        for j in prange(nz_abc - N_abc, nz_abc - 4):
            jdx = j - (nz_abc - N_abc) 
            
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)               
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / (2*dz)
            
            ZetazFD[jdx, i] = a_z[jdx] * ZetazFD[jdx, i] + b_z[jdx] * (pzz + psiz)

    return ZetazFU, ZetazFD

@jit(parallel=True, nopython=True)
def updateWaveEquationVTI(Uf, Uc, Qc, Qf, nx, nz, dt, dx, dz, vpz, epsilon, delta):  
    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    for i in prange(4, nx - 4):  
        for j in prange(4, nz - 4):
            cx = vpz[j,i]**2 * (1 + 2 * epsilon[j,i])
            bx = vpz[j,i]**2 * (1 + 2 * delta[j,i])
            cz = bz = vpz[j,i]**2      
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) + c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) + c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                   c3 * (Qc[j+3, i] + Qc[j-3, i]) + c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * pxx  + cz * qzz)
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * pxx  + bz * qzz)

    return Uf, Qf

@jit(nopython=True, parallel=True)
def updateWaveEquationVTICPML(Uf, Uc, Qc, Qf, dt, dx, dz, vpz, epsilon, delta,
                               nx_abc, nz_abc, PsixFR, PsixFL, PsizqFU, PsizqFD, ZetaxFR, ZetaxFL, ZetazqFU, ZetazqFD, N_abc):
    
    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 4. / 5.
    a2 = -1. / 5.
    a3 = 4. / 105.
    a4 = -1. / 280.

    # Região Interior
    for i in prange(N_abc, nx_abc - N_abc):  
        for j in prange(N_abc, nz_abc - N_abc):
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
    for i in prange(4, N_abc):
        for j in range(N_abc, nz_abc - N_abc):
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
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / (2 * dx)           
                  
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * (pxx + psix + ZetaxFL[j,i] ) + cz * qzz)
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * (pxx + psix + ZetaxFL[j,i]) + bz * qzz)

    # Região Direita
    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in range(N_abc, nz_abc - N_abc):
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
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / (2 * dx)           
                  
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * (pxx + psix + ZetaxFR[j,idx] ) + cz * qzz)
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * (pxx + psix + ZetaxFR[j,idx]) + bz * qzz)
    
    # Região Superior
    for i in prange(N_abc, nx_abc - N_abc):
        for j in range(4, N_abc):
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
                    a4 * (PsizqFU[j+4, i] - PsizqFU[j-4, i])) / (2*dz)           
                  
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * pxx + cz *(qzz + psiqz + ZetazqFU[j,i]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * pxx + bz *(qzz + psiqz + ZetazqFU[j,i]))

    # Região Inferior
    for i in prange(N_abc, nx_abc - N_abc):
        for j in range(nz_abc - N_abc, nz_abc - 4):
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
                    a4 * (PsizqFD[jdx+4, i] - PsizqFD[jdx-4, i])) / (2*dz)           
                  
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * pxx  + cz *(qzz + psiqz + ZetazqFD[jdx,i]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * pxx  + bz *(qzz + psiqz + ZetazqFD[jdx,i]))

    # Quina Superior Esquerda
    for i in prange(4, N_abc):
        for j in range(4, N_abc):
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
                    a4 * (PsizqFU[j+4, i] - PsizqFU[j-4, i])) / (2*dz)
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / (2 * dx)           
                  
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * (pxx + psix + ZetaxFL[j,i]) + cz *(qzz + psiqz + ZetazqFU[j,i]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * (pxx + psix + ZetaxFL[j,i]) + bz *(qzz + psiqz + ZetazqFU[j,i]))

    # Quina Superior Direita
    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in range(4, N_abc):
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
                    a4 * (PsizqFU[j+4, i] - PsizqFU[j-4, i])) / (2*dz)
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / (2 * dx)           
                  
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * (pxx + psix + ZetaxFR[j,idx]) + cz *(qzz + psiqz + ZetazqFU[j,i]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * (pxx + psix + ZetaxFR[j,idx]) + bz *(qzz + psiqz + ZetazqFU[j,i]))
    
    # Quina Inferior Esquerda
    for i in prange(4, N_abc):
        for j in range(nz_abc - N_abc, nz_abc - 4):
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
                    a4 * (PsizqFD[jdx+4, i] - PsizqFD[jdx-4, i])) / (2*dz)
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / (2 * dx)           
                  
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * (pxx + psix + ZetaxFL[j,i]) + cz *(qzz + psiqz + ZetazqFD[jdx,i]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * (pxx + psix + ZetaxFL[j,i]) + bz *(qzz + psiqz + ZetazqFD[jdx,i]))
    
    # Quina Inferior Direita
    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in range(nz_abc - N_abc, nz_abc - 4):
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
                    a4 * (PsizqFD[jdx+4, i] - PsizqFD[jdx-4, i])) / (2*dz)
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / (2 * dx)           
                  
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * (pxx + psix + ZetaxFR[j,idx]) + cz *(qzz + psiqz + ZetazqFD[jdx,i]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * (pxx + psix + ZetaxFR[j,idx]) + bz *(qzz + psiqz + ZetazqFD[jdx,i]))


    return Uf, Qf

@jit(nopython=True, parallel=True)
def updatePsiVTIUD (PsizqFU, PsizqFD, nx_abc, nz_abc, a_z, b_z, Qc, dz, N_abc):

    a1 = 4. / 5.
    a2 = -1. / 5.
    a3 = 4. / 105.
    a4 = -1. / 280.

    for i in prange(4, nx_abc - 4):
        for j in prange(4, N_abc):
            jdx = N_abc - j

            qz = (a1 * (Qc[j+1, i] - Qc[j-1, i]) +
                a2 * (Qc[j+2, i] - Qc[j-2, i]) +
                a3 * (Qc[j+3, i] - Qc[j-3, i]) +
                a4 * (Qc[j+4, i] - Qc[j-4, i])) / (2 * dz) 
            
            PsizqFU[j, i] = a_z[jdx] * PsizqFU[j, i] + b_z[jdx] * qz

    for i in prange(4, nx_abc - 4):
        for j in prange(nz_abc - N_abc, nz_abc - 4):
            jdx = j - (nz_abc - N_abc)

            qz = (a1 * (Qc[j+1, i] - Qc[j-1, i]) +
                a2 * (Qc[j+2, i] - Qc[j-2, i]) +
                a3 * (Qc[j+3, i] - Qc[j-3, i]) +
                a4 * (Qc[j+4, i] - Qc[j-4, i])) / (2 * dz) 
            
            PsizqFD[jdx, i] = a_z[jdx] * PsizqFD[jdx, i] + b_z[jdx] * qz

    return PsizqFU, PsizqFD

@jit(nopython=True, parallel=True)
def updateZetaVTIUD (PsizqFU, PsizqFD, ZetazqFU, ZetazqFD, nx_abc, nz_abc, a_z, b_z, Qc, dz, N_abc):

    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 4. / 5.
    a2 = -1. / 5.
    a3 = 4. / 105.
    a4 = -1. / 280.

    for i in prange(4, nx_abc - 4):
        for j in prange(4, N_abc- 4):
            jdx = N_abc - j

            qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + 
                    c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                    c3 * (Qc[j+3, i] + Qc[j-3, i]) + 
                    c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
            psiqz = (a1 * (PsizqFU[j+1, i] - PsizqFU[j-1, i]) +
                    a2 * (PsizqFU[j+2, i] - PsizqFU[j-2, i]) +
                    a3 * (PsizqFU[j+3, i] - PsizqFU[j-3, i]) +
                    a4 * (PsizqFU[j+4, i] - PsizqFU[j-4, i])) / (2*dz)

            ZetazqFU[j, i] = a_z[jdx] * ZetazqFU[j, i] + b_z[jdx] * (qzz + psiqz)

    for i in prange(4, nx_abc - 4):
        for j in prange(nz_abc - N_abc, nz_abc - 4):
            jdx = j - (nz_abc - N_abc) 

            qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + 
                    c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                    c3 * (Qc[j+3, i] + Qc[j-3, i]) + 
                    c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
            psiqz = (a1 * (PsizqFD[jdx+1, i] - PsizqFD[jdx-1, i]) +
                    a2 * (PsizqFD[jdx+2, i] - PsizqFD[jdx-2, i]) +
                    a3 * (PsizqFD[jdx+3, i] - PsizqFD[jdx-3, i]) +
                    a4 * (PsizqFD[jdx+4, i] - PsizqFD[jdx-4, i])) / (2*dz)

            ZetazqFD[jdx, i] = a_z[jdx] * ZetazqFD[jdx, i] + b_z[jdx] * (qzz + psiqz)

    return ZetazqFU, ZetazqFD

@jit(nopython=True, parallel=True)
def updateWaveEquationTTI(Uf, Uc, Qc, Qf, nx, nz, dt, dx, dz, vpz, vsz, epsilon, delta, theta):
    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 4. / 5.
    a2 = -1. / 5.
    a3 = 4./105.
    a4 = -1./280.
    for i in prange(4, nx - 4):
        for j in prange(4, nz - 4):
            vpx = vpz[j, i] * np.sqrt(1 + 2*epsilon[j, i])
            vpn = vpz[j, i] * np.sqrt(1 + 2*delta[j, i])
            cpx = vpx**2 * np.cos(theta[j, i])**2 + vsz[j, i]**2 * np.sin(theta[j, i])**2
            cpz = vpx**2 * np.sin(theta[j, i])**2 + vsz[j, i]**2 * np.cos(theta[j, i])**2
            cpxz = vsz[j, i]**2 * np.sin(2 * theta[j, i]) - vpx**2 * np.sin(2 * theta[j, i])
            dpx = vpz[j, i]**2 * np.sin(theta[j, i])**2 - vsz[j, i]**2 * np.sin(theta[j, i])**2
            dpz = vpz[j, i]**2 * np.cos(theta[j, i])**2 - vsz[j, i]**2 * np.cos(theta[j, i])**2
            dpxz = vpz[j, i]**2 * np.sin(2 * theta[j, i]) - vsz[j, i]**2 * np.sin(2 * theta[j, i])
            cqx = vpn**2 * np.cos(theta[j, i])**2 - vsz[j, i]**2 * np.cos(theta[j, i])**2
            cqz = vpn**2 * np.sin(theta[j, i])**2 - vsz[j, i]**2 * np.sin(theta[j, i])**2  
            cqxz = vsz[j, i]**2 * np.sin(2 * theta[j, i]) - vpn**2 * np.sin(2 * theta[j, i])
            dqx = vsz[j, i]**2 * np.cos(theta[j, i])**2 + vpz[j, i]**2 * np.sin(theta[j, i])**2  
            dqz = vpz[j, i]**2 * np.cos(theta[j, i])**2 + vsz[j, i]**2 * np.sin(theta[j, i])**2
            dqxz = vpz[j, i]**2 * np.sin(2 * theta[j, i]) - vsz[j, i]**2 * np.sin(2 * theta[j, i])

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

    return Uf, Qf



# @jit(nopython=True, parallel=True)
# def updateWaveEquationTTICPML(Uf, Uc, Qc, Qf, nx_abc, nz_abc, dt, dx, dz, vpz, vsz, epsilon, delta, theta,PsixF,PsizF,PsixqF,PsizqF,ZetaxF,ZetazF,ZetaxzF,ZetaxqF,ZetazqF,ZetaxzqF, N_abc):
#     c0 = -205. / 72.
#     c1 = 8. / 5.
#     c2 = -1. / 5.
#     c3 = 8. / 315.
#     c4 = -1. / 560.
#     a1 = 4. / 5.
#     a2 = -1. / 5.
#     a3 = 4./105.
#     a4 = -1./280.

#     # Região Interior
#     for i in prange(N_abc, nx_abc - N_abc):
#         for j in prange(N_abc, nz_abc - N_abc):
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

#             Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cpx * pxx + cpz * pzz + cpxz * pxz + dpx * qxx + dpz * qzz + dpxz * qxz)
#             Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (cqx * pxx + cqz * pzz + cqxz * pxz + dqx * qxx + dqz * qzz + dqxz * qxz)
    
#     # Região Esquerda
#     for i in prange(4, N_abc):
#         for j in prange(N_abc, nz_abc - N_abc):
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
                       
#             psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
#                     a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
#                     a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
#                     a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / (2 * dx)                   
#             psizx = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
#                     a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
#                     a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
#                     a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / (2*dx)
#             psiqx = (a1 * (PsixqF[j+1, i] - PsixqF[j-1, i]) +
#                     a2 * (PsixqF[j+2, i] - PsixqF[j-2, i]) +
#                     a3 * (PsixqF[j+3, i] - PsixqF[j-3, i]) +
#                     a4 * (PsixqF[j+4, i] - PsixqF[j-4, i])) / (2*dx)
            
#             psiqz = (a1 * (PsizqF[j+1, i] - PsizqF[j-1, i]) +
#                     a2 * (PsizqF[j+2, i] - PsizqF[j-2, i]) +
#                     a3 * (PsizqF[j+3, i] - PsizqF[j-3, i]) +
#                     a4 * (PsizqF[j+4, i] - PsizqF[j-4, i])) / (2*dz)

#             psiqzx = (a1 * (PsizqF[j+1, i] - PsizqF[j-1, i]) +
#                     a2 * (PsizqF[j+2, i] - PsizqF[j-2, i]) +
#                     a3 * (PsizqF[j+3, i] - PsizqF[j-3, i]) +
#                     a4 * (PsizqF[j+4, i] - PsizqF[j-4, i])) / (2*dx)
            
#             Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cpx * (pxx + psix + ZetaxF[j, i]) + cpz * pzz + cpxz * (pxz + psizx + ZetaxzF[j,i]) + dpx * (qxx + psiqx + ZetaxqF[j, i]) + dpz * qzz + dpxz * (qxz + psiqzx + ZetaxzqF[j,i]))
#             Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (cqx * (pxx + psix + ZetaxF[j, i]) + cqz * pzz + cqxz * (pxz + psizx + ZetaxzF[j,i]) + dqx * (qxx + psiqx + ZetaxqF[j, i]) + dqz * qzz  + dqxz * (qxz + psiqzx + ZetaxzqF[j,i]))


#     for i in prange(4, N_abc):
#         for j in prange(N_abc, nz_abc - N_abc):
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
#                     a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / (2 * dx)
            
#             psiz = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
#                     a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
#                     a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
#                     a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / (2*dz)          
            
#             psizx = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
#                     a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
#                     a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
#                     a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / (2*dx)

#             psiqx = (a1 * (PsixqF[j+1, i] - PsixqF[j-1, i]) +
#                     a2 * (PsixqF[j+2, i] - PsixqF[j-2, i]) +
#                     a3 * (PsixqF[j+3, i] - PsixqF[j-3, i]) +
#                     a4 * (PsixqF[j+4, i] - PsixqF[j-4, i])) / (2*dx)
            
#             psiqz = (a1 * (PsizqF[j+1, i] - PsizqF[j-1, i]) +
#                     a2 * (PsizqF[j+2, i] - PsizqF[j-2, i]) +
#                     a3 * (PsizqF[j+3, i] - PsizqF[j-3, i]) +
#                     a4 * (PsizqF[j+4, i] - PsizqF[j-4, i])) / (2*dz)

#             psiqzx = (a1 * (PsizqF[j+1, i] - PsizqF[j-1, i]) +
#                     a2 * (PsizqF[j+2, i] - PsizqF[j-2, i]) +
#                     a3 * (PsizqF[j+3, i] - PsizqF[j-3, i]) +
#                     a4 * (PsizqF[j+4, i] - PsizqF[j-4, i])) / (2*dx)
            
#             Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cpx * (pxx + psix + ZetaxF[j, i]) + cpz * (pzz + psiz + ZetazF[j, i]) + cpxz * (pxz + psizx + ZetaxzF[j,i]) + dpx * (qxx + psiqx + ZetaxqF[j, i]) + dpz * (qzz + psiqz + ZetazqF[j,i]) + dpxz * (qxz + psiqzx + ZetaxzqF[j,i]))
#             Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (cqx * (pxx + psix + ZetaxF[j, i]) + cqz * (pzz + psiz + ZetazF[j, i]) + cqxz * (pxz + psizx + ZetaxzF[j,i]) + dqx * (qxx + psiqx + ZetaxqF[j, i]) + dqz * (qzz + psiqz + ZetazqF[j,i]) + dqxz * (qxz + psiqzx + ZetaxzqF[j,i]))


#     return Uf, Qf

# @jit(nopython=True, parallel=True)
# def updatePsiTTIRL (PsixqFR, PsixqFL, nx_abc, nz_abc, a_x, b_x, Qc, dx, N_abc):

#     a1 = 4. / 5.
#     a2 = -1. / 5.
#     a3 = 4. / 105.
#     a4 = -1. / 280.

#     for i in prange(4, N_abc):
#         idx = N_abc - i
#         for j in prange(4, nz_abc - 4):

#             qx = (a1 * (Qc[j, i+1] - Qc[j, i-1]) +
#                 a2 * (Qc[j, i+2] - Qc[j, i-2]) +
#                 a3 * (Qc[j, i+3] - Qc[j, i-3]) +
#                 a4 * (Qc[j, i+4] - Qc[j, i-4])) / (2 * dx)
        
#             PsixqFL[j, i] = a_x[idx] * PsixqFL[j, i] + b_x[idx] * qx

#     for i in prange(nx_abc - N_abc, nx_abc - 4):
#         idx = i - (nx_abc - N_abc)
#         for j in prange(4, nz_abc - 4):

#             qx = (a1 * (Qc[j, i+1] - Qc[j, i-1]) +
#                 a2 * (Qc[j, i+2] - Qc[j, i-2]) +
#                 a3 * (Qc[j, i+3] - Qc[j, i-3]) +
#                 a4 * (Qc[j, i+4] - Qc[j, i-4])) / (2 * dx)
        
#             PsixqFR[j, idx] = a_x[idx] * PsixqFR[j, idx] + b_x[idx] * qx

#     return PsixqFR, PsixqFL

# @jit(nopython=True, parallel=True)
# def updateZetaTTIRL(PsixqFR, PsixqFL, ZetaxqFR, ZetaxqFL, nx_abc, nz_abc, a_x, b_x, Qc, dx, N_abc):

#     c0 = -205. / 72.
#     c1 = 8. / 5.
#     c2 = -1. / 5.
#     c3 = 8. / 315.
#     c4 = -1. / 560.
#     a1 = 4. / 5.
#     a2 = -1. / 5.
#     a3 = 4. / 105.
#     a4 = -1. / 280.

#     for i in prange(4, N_abc - 4):
#         idx = N_abc - i 
#         for j in prange(4, nz_abc - 4):

#             qxx = (c0 * Qc[j, i] + c1 * (Qc[j, i+1] + Qc[j, i-1]) + 
#                     c2 * (Qc[j, i+2] + Qc[j, i-2]) +
#                     c3 * (Qc[j, i+3] + Qc[j, i-3]) + 
#                     c4 * (Qc[j, i+4] + Qc[j, i-4])) / (dx * dx)
            
#             psiqx = (a1 * (PsixqFL[j+1, i] - PsixqFL[j-1, i]) +
#                 a2 * (PsixqFL[j+2, i] - PsixqFL[j-2, i]) +
#                 a3 * (PsixqFL[j+3, i] - PsixqFL[j-3, i]) +
#                 a4 * (PsixqFL[j+4, i] - PsixqFL[j-4, i])) / (2*dx)


#             ZetaxqFL[j, i] = a_x[idx] * ZetaxqFL[j, i] + b_x[idx] * (qxx + psiqx)

#     for i in prange(nx_abc - N_abc, nx_abc - 4):
#         idx = i - (nx_abc - N_abc) 
#         for j in prange(4, nz_abc - 4):

#             qxx = (c0 * Qc[j, i] + c1 * (Qc[j, i+1] + Qc[j, i-1]) + 
#                     c2 * (Qc[j, i+2] + Qc[j, i-2]) +
#                     c3 * (Qc[j, i+3] + Qc[j, i-3]) + 
#                     c4 * (Qc[j, i+4] + Qc[j, i-4])) / (dx * dx)
            
#             psiqx = (a1 * (PsixqFR[j+1, idx] - PsixqFR[j-1, idx]) +
#                 a2 * (PsixqFR[j+2, idx] - PsixqFR[j-2, idx]) +
#                 a3 * (PsixqFR[j+3, idx] - PsixqFR[j-3, idx]) +
#                 a4 * (PsixqFR[j+4, idx] - PsixqFR[j-4, idx])) / (2*dx)


#             ZetaxqFR[j, idx] = a_x[idx] * ZetaxqFR[j, idx] + b_x[idx] * (qxx + psiqx)

#     return ZetaxqFL, ZetaxqFR


# @jit(nopython=True, parallel=True)
# def updateZetaTTIUD(PsizFU, PsizFD, ZetaxzFU, ZetaxzFD, nx_abc, nz_abc, a_x, b_x, Uc, dz, dx, N_abc):

#     a1 = 4. / 5.
#     a2 = -1. / 5.
#     a3 = 4. / 105.
#     a4 = -1. / 280.

#     for i in prange(4, N_abc - 4):
#         idx = N_abc - i 
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

#             psizx = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
#                 a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
#                 a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
#                 a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / (2*dx)

            
#             ZetaxzFU[j, i] = a_x[idx] * ZetaxzFU[j, i] + b_x[idx] * (pxz + psizx)
    
#     for i in prange(nx_abc - N_abc, nx_abc - 4):
#         idx = i - (nx_abc - N_abc) 
#         for j in prange(4, nz_abc - 4):
#             jdx = j - (nz_abc - N_abc)

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

#             psizx = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
#                 a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
#                 a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
#                 a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / (2*dx)

            
#             ZetaxzFD[jdx, idx] = a_x[idx] * ZetaxzFD[jdx, idx] + b_x[idx] * (pxz + psizx)

#     return ZetaxzFU,ZetaxzFD

# @jit(nopython=True, parallel=True)
# def updateZetaQTTIUD(PsizqFU, PsizqFD, ZetaxzqFU, ZetaxzqFD, nx_abc, nz_abc, a_x, b_x, Qc, dz, dx, N_abc):

#     a1 = 4. / 5.
#     a2 = -1. / 5.
#     a3 = 4. / 105.
#     a4 = -1. / 280.

#     for i in prange(4, N_abc - 4):
#         idx = N_abc - i 
#         for j in prange(4, nz_abc - 4):
            
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

#             psiqzx = (a1 * (PsizqFU[j+1, i] - PsizqFU[j-1, i]) +
#                 a2 * (PsizqFU[j+2, i] - PsizqFU[j-2, i]) +
#                 a3 * (PsizqFU[j+3, i] - PsizqFU[j-3, i]) +
#                 a4 * (PsizqFU[j+4, i] - PsizqFU[j-4, i])) / (2*dx)

#             ZetaxzqFU[j, i] = a_x[idx] * ZetaxzqFU[j, i] + b_x[idx] * (qxz + psiqzx)

#     for i in prange(nx_abc - N_abc, nx_abc - 4):
#         idx = i - (nx_abc - N_abc) 
#         for j in prange(4, nz_abc - 4):
#             jdx = j - (nz_abc - N_abc)
            
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

#             psiqzx = (a1 * (PsizqFD[jdx+1, i] - PsizqFD[jdx-1, i]) +
#                 a2 * (PsizqFD[jdx+2, i] - PsizqFD[jdx-2, i]) +
#                 a3 * (PsizqFD[jdx+3, i] - PsizqFD[jdx-3, i]) +
#                 a4 * (PsizqFD[jdx+4, i] - PsizqFD[jdx-4, i])) / (2*dx)

#             ZetaxzqFD[jdx, idx] = a_x[idx] * ZetaxzqFD[jdx, idx] + b_x[idx] * (qxz + psiqzx)

#     return  ZetaxzqFU, ZetaxzqFD

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

