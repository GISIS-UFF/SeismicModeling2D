import numpy as np
from numba import jit,prange, njit, cuda

def ricker(f0, t):
    pi = np.pi
    td  = t - 2 * np.sqrt(pi) / f0
    fcd = f0 / (np.sqrt(pi) * 3) 
    source = (1 - 2 * pi * (pi * fcd * td) * (pi * fcd * td)) * np.exp(-pi * (pi * fcd * td) * (pi * fcd * td))
    return source

@njit(inline = "always")
def horizontal_dampening_profiles(N_abc,nx_abc, dx, vp, f_pico, d0, dt, i, j):
    if i < N_abc:
        points_CPML = (N_abc - i - 1)*dx
        posicao_relativa = points_CPML / (N_abc*dx)
        d = d0 * (posicao_relativa**2) * vp[j,i]
        alpha = np.pi* f_pico * (1 - posicao_relativa**2)

        a = np.exp(-(d + alpha) * dt)
        b = (d / (d + alpha)) * (a - 1)

    elif i >= nx_abc - N_abc:
        points_CPML = (i - nx_abc + N_abc)*dx
        posicao_relativa = points_CPML / (N_abc*dx)
        d = d0 * (posicao_relativa**2) * vp[j,i]
        alpha = np.pi* f_pico * (1 - posicao_relativa**2)

        a = np.exp(-(d + alpha) * dt)
        b = (d / (d + alpha)) * (a - 1) 

    return a, b

@njit(inline = "always")
def vertical_dampening_profiles(N_abc,nz_abc, dz, vp, f_pico, d0, dt, i, j):
    if j < N_abc:
        points_CPML = (N_abc - j - 1)*dz
        posicao_relativa = points_CPML / (N_abc*dz)
        d = d0 * (posicao_relativa**2) * vp[j,i]
        alpha = np.pi* f_pico * (1 - posicao_relativa**2)

        a = np.exp(-(d + alpha) * dt)
        b = (d / (d + alpha)) * (a - 1)

    elif j >= nz_abc - N_abc:
        points_CPML = (j - nz_abc + N_abc)*dz
        posicao_relativa = points_CPML / (N_abc*dz)
        d = d0 * (posicao_relativa**2) * vp[j,i]
        alpha = np.pi* f_pico * (1 - posicao_relativa**2)

        a = np.exp(-(d + alpha) * dt)
        b = (d / (d + alpha)) * (a - 1) 
        
    return a, b

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
def updatePsi(PsixFR, PsixFL, PsizFU, PsizFD, nx_abc, nz_abc, Uc, dx,dz, N_abc,ax,bx,az,bz, f_pico, d0, dt, vp):

    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    for j in prange(4, nz_abc - 4):
        for i in prange(4, N_abc):

            px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
                a2 * (Uc[j, i+2] - Uc[j, i-2]) +
                a3 * (Uc[j, i+3] - Uc[j, i-3]) +
                a4 * (Uc[j, i+4] - Uc[j, i-4])) / dx
            
            PsixFL[j, i] = ax[j,i] * PsixFL[j, i] + bx[j,i] * px

    for j in prange(4, nz_abc - 4):
        for i in prange(nx_abc - N_abc, nx_abc - 4):
            idx = i - (nx_abc - N_abc)

            px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
                a2 * (Uc[j, i+2] - Uc[j, i-2]) +
                a3 * (Uc[j, i+3] - Uc[j, i-3]) +
                a4 * (Uc[j, i+4] - Uc[j, i-4])) / dx
            
            PsixFR[j, idx] = ax[j,i] * PsixFR[j, idx] + bx[j,i] * px

    for j in prange(4, N_abc):
        for i in prange(4, nx_abc - 4):

            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz 
            
            PsizFU[j, i] = az[j,i] * PsizFU[j, i] + bz[j,i] * pz

    for j in prange(nz_abc - N_abc, nz_abc - 4):  
        jdx = j - (nz_abc - N_abc)
        for i in prange(4, nx_abc - 4):

            pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                a4 * (Uc[j+4, i] - Uc[j-4, i])) / dz 
            
            PsizFD[jdx, i] = az[j,i] * PsizFD[jdx, i] + bz[j,i] * pz

    return PsixFR, PsixFL, PsizFU, PsizFD

@jit(nopython=True, parallel=True)
def updateZeta(PsixFR, PsixFL, ZetaxFR, ZetaxFL,PsizFU, PsizFD, ZetazFU, ZetazFD, nx_abc, nz_abc, Uc, dx, dz, N_abc,ax,bx,az,bz, f_pico, d0, dt, vp):

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
        
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
        
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx

            ZetaxFL[j, i] = ax[j,i] * ZetaxFL[j, i] + bx[j,i] * (pxx + psix)

    for j in prange(4, nz_abc - 4):
        for i in prange(nx_abc - N_abc, nx_abc - 4):
            idx = i - (nx_abc - N_abc) 
                
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
        
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx

            ZetaxFR[j, idx] = ax[j,i] * ZetaxFR[j, idx] + bx[j,i] * (pxx + psix)

    for j in prange(4, N_abc):
        for i in prange(4, nx_abc - 4):
                
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)               
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz
            
            ZetazFU[j, i] = az[j,i] * ZetazFU[j, i] + bz[j,i] * (pzz + psiz)

    for j in prange(nz_abc - N_abc, nz_abc - 4):
        jdx = j - (nz_abc - N_abc) 
        for i in prange(4, nx_abc - 4):
            
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)               
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz
            
            ZetazFD[jdx, i] = az[j,i] * ZetazFD[jdx, i] + bz[j,i] * (pzz + psiz) 

    return ZetaxFR, ZetaxFL, ZetazFU, ZetazFD

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
    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

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
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx           
                  
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
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx           
                  
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
                    a4 * (PsizqFU[j+4, i] - PsizqFU[j-4, i])) / dz           
                  
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
                    a4 * (PsizqFD[jdx+4, i] - PsizqFD[jdx-4, i])) / dz           
                  
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
                    a4 * (PsizqFU[j+4, i] - PsizqFU[j-4, i])) / dz
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx           
                  
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
                    a4 * (PsizqFU[j+4, i] - PsizqFU[j-4, i])) / dz
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx           
                  
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
                    a4 * (PsizqFD[jdx+4, i] - PsizqFD[jdx-4, i])) / dz
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx           
                  
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
                    a4 * (PsizqFD[jdx+4, i] - PsizqFD[jdx-4, i])) / dz
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx           
                  
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * (pxx + psix + ZetaxFR[j,idx]) + cz *(qzz + psiqz + ZetazqFD[jdx,i]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * (pxx + psix + ZetaxFR[j,idx]) + bz *(qzz + psiqz + ZetazqFD[jdx,i]))


    return Uf, Qf

@jit(nopython=True, parallel=True)
def updatePsiVTI (PsizqFU, PsizqFD, nx_abc, nz_abc, a_z, b_z, Qc, dz, N_abc):

    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    for i in prange(4, nx_abc - 4):
        for j in prange(4, N_abc):

            qz = (a1 * (Qc[j+1, i] - Qc[j-1, i]) +
                a2 * (Qc[j+2, i] - Qc[j-2, i]) +
                a3 * (Qc[j+3, i] - Qc[j-3, i]) +
                a4 * (Qc[j+4, i] - Qc[j-4, i])) / dz 
            
            PsizqFU[j, i] = a_z[j, i] * PsizqFU[j, i] + b_z[j, i] * qz

    for i in prange(4, nx_abc - 4):
        for j in prange(nz_abc - N_abc, nz_abc - 4):
            jdx = j - (nz_abc - N_abc)

            qz = (a1 * (Qc[j+1, i] - Qc[j-1, i]) +
                a2 * (Qc[j+2, i] - Qc[j-2, i]) +
                a3 * (Qc[j+3, i] - Qc[j-3, i]) +
                a4 * (Qc[j+4, i] - Qc[j-4, i])) / dz 
            
            PsizqFD[jdx, i] = a_z[j, i] * PsizqFD[jdx, i] + b_z[j, i] * qz

    return PsizqFU, PsizqFD

@jit(nopython=True, parallel=True)
def updateZetaVTI (PsizqFU, PsizqFD, ZetazqFU, ZetazqFD, nx_abc, nz_abc, a_z, b_z, Qc, dz, N_abc):

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
        for j in prange(4, N_abc):

            qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + 
                    c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                    c3 * (Qc[j+3, i] + Qc[j-3, i]) + 
                    c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
            psiqz = (a1 * (PsizqFU[j+1, i] - PsizqFU[j-1, i]) +
                    a2 * (PsizqFU[j+2, i] - PsizqFU[j-2, i]) +
                    a3 * (PsizqFU[j+3, i] - PsizqFU[j-3, i]) +
                    a4 * (PsizqFU[j+4, i] - PsizqFU[j-4, i])) / dz

            ZetazqFU[j, i] = a_z[j, i] * ZetazqFU[j, i] + b_z[j, i] * (qzz + psiqz)

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
                    a4 * (PsizqFD[jdx+4, i] - PsizqFD[jdx-4, i])) / dz

            ZetazqFD[jdx, i] = a_z[j, i] * ZetazqFD[jdx, i] + b_z[j, i] * (qzz + psiqz)

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

@jit(nopython=True, parallel=True)
def updateWaveEquationTTICPML(Uf, Uc, Qc, Qf, nx_abc, nz_abc, dt, dx, dz, vpz, vsz, epsilon, delta, theta, PsixFR, PsixFL,PsizFU, PsizFD,PsixqFR, PsixqFL,PsizqFU, PsizqFD,PsiauxFL,PsiauxFR,PsiauxqFL,PsiauxqFR, ZetaxFR, ZetaxFL,ZetazFU, ZetazFD,ZetaxqFL, ZetaxqFR,ZetazqFU, ZetazqFD, ZetaxzFLU,ZetaxzFLD,ZetaxzFRU,ZetaxzFRD,ZetaxzqFLU,ZetaxzqFLD, ZetaxzqFRU,ZetaxzqFRD,ZetaxzFL, ZetaxzFR, ZetaxzqFL, ZetaxzqFR, N_abc):
    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 4. / 5.
    a2 = -1. / 5.
    a3 = 4./105.
    a4 = -1./280.

    # Região Interior
    for i in prange(N_abc, nx_abc - N_abc):
        for j in prange(N_abc, nz_abc - N_abc):
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
    
    # Região Esquerda
    for i in prange(4, N_abc):
        for j in prange(N_abc, nz_abc - N_abc):
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
                       
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx 
            psiqx = (a1 * (PsixqFL[j, i+1] - PsixqFL[j, i-1]) +
                    a2 * (PsixqFL[j, i+2] - PsixqFL[j, i-2]) +
                    a3 * (PsixqFL[j, i+3] - PsixqFL[j, i-3]) +
                    a4 * (PsixqFL[j, i+4] - PsixqFL[j, i-4])) / dx    
            psizx = (a1 * (PsiauxFL[j, i+1] - PsiauxFL[j, i-1]) +
                a2 * (PsiauxFL[j, i+2] - PsiauxFL[j, i-2]) +
                a3 * (PsiauxFL[j, i+3] - PsiauxFL[j, i-3]) +
                a4 * (PsiauxFL[j, i+4] - PsiauxFL[j, i-4])) / dx
            psiqzx = (a1 * (PsiauxqFL[j, i+1] - PsiauxqFL[j, i-1]) +
                a2 * (PsiauxqFL[j, i+2] - PsiauxqFL[j, i-2]) +
                a3 * (PsiauxqFL[j, i+3] - PsiauxqFL[j, i-3]) +
                a4 * (PsiauxqFL[j, i+4] - PsiauxqFL[j, i-4])) / dx
                      
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cpx * (pxx + psix + ZetaxFL[j, i]) + cpz * pzz + cpxz * (pxz + psizx + ZetaxzFL[j,i]) + dpx * (qxx + psiqx + ZetaxqFL[j, i]) + dpz * qzz + dpxz * (qxz + psiqzx + ZetaxzqFL[j,i]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (cqx * (pxx + psix + ZetaxFL[j, i]) + cqz * pzz + cqxz * (pxz + psizx + ZetaxzFL[j,i]) + dqx * (qxx + psiqx + ZetaxqFL[j, i]) + dqz * qzz  + dqxz * (qxz + psiqzx + ZetaxzqFL[j,i]))

    # Região Direita
    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in range(N_abc, nz_abc - N_abc):
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
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx               
            psiqx = (a1 * (PsixqFR[j, idx+1] - PsixqFR[j, idx-1]) +
                    a2 * (PsixqFR[j, idx+2] - PsixqFR[j, idx-2]) +
                    a3 * (PsixqFR[j, idx+3] - PsixqFR[j, idx-3]) +
                    a4 * (PsixqFR[j, idx+4] - PsixqFR[j, idx-4])) / dx
            psizx = (a1 * (PsiauxFR[j, idx+1] - PsiauxFR[j, idx-1]) +
                a2 * (PsiauxFR[j, idx+2] - PsiauxFR[j, idx-2]) +
                a3 * (PsiauxFR[j, idx+3] - PsiauxFR[j, idx-3]) +
                a4 * (PsiauxFR[j, idx+4] - PsiauxFR[j, idx-4])) / dx
            psiqzx = (a1 * (PsiauxqFR[j, idx+1] - PsiauxqFR[j, idx-1]) +
                a2 * (PsiauxqFR[j, idx+2] - PsiauxqFR[j, idx-2]) +
                a3 * (PsiauxqFR[j, idx+3] - PsiauxqFR[j, idx-3]) +
                a4 * (PsiauxqFR[j, idx+4] - PsiauxqFR[j, idx-4])) / dx

            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cpx * (pxx + psix + ZetaxFR[j, idx]) + cpz * pzz + cpxz * (pxz + psizx + ZetaxzFR[j,idx]) + dpx * (qxx + psiqx + ZetaxqFR[j, idx]) + dpz * qzz + dpxz * (qxz + psiqzx + ZetaxzqFR[j,idx]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (cqx * (pxx + psix + ZetaxFR[j, idx]) + cqz * pzz + cqxz * (pxz + psizx + ZetaxzFR[j,idx]) + dqx * (qxx + psiqx + ZetaxqFR[j, idx]) + dqz * qzz  + dqxz * (qxz + psiqzx + ZetaxzqFR[j,idx]))

    # Região Superior
    for j in prange(4, N_abc):
        for i in range(N_abc, nx_abc - N_abc):
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
            
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz             
            psiqz = (a1 * (PsizqFU[j+1, i] - PsizqFU[j-1, i])+
                    a2 * (PsizqFU[j+2, i] - PsizqFU[j-2, i]) +
                    a3 * (PsizqFU[j+3, i] - PsizqFU[j-3, i]) +
                    a4 * (PsizqFU[j+4, i] - PsizqFU[j-4, i])) / dz
        
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cpx * pxx + cpz * (pzz + psiz + ZetazFU[j, i]) + cpxz * (pxz) + dpx * qxx  + dpz * (qzz + psiqz + ZetazqFU[j,i]) + dpxz * (qxz))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (cqx * pxx + cqz * (pzz + psiz + ZetazFU[j, i]) + cqxz * (pxz) + dqx * qxx  + dqz * (qzz + psiqz + ZetazqFU[j,i]) + dqxz * (qxz))
   
    # Região inferior
    for j in prange(nz_abc - N_abc, nz_abc - 4):
        jdx = j - (nz_abc - N_abc)
        for i in range(N_abc, nx_abc - N_abc):
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
            
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz          
            psiqz = (a1 * (PsizqFD[jdx+1, i] - PsizqFD[jdx-1, i])+
                    a2 * (PsizqFD[jdx+2, i] - PsizqFD[jdx-2, i]) +
                    a3 * (PsizqFD[jdx+3, i] - PsizqFD[jdx-3, i]) +
                    a4 * (PsizqFD[jdx+4, i] - PsizqFD[jdx-4, i])) / dz
            
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cpx * pxx + cpz * (pzz + psiz + ZetazFD[jdx, i]) + cpxz * (pxz) + dpx * qxx  + dpz * (qzz + psiqz + ZetazqFD[jdx,i]) + dpxz * (qxz))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (cqx * pxx + cqz * (pzz + psiz + ZetazFD[jdx, i]) + cqxz * (pxz) + dqx * qxx  + dqz * (qzz + psiqz + ZetazqFD[jdx,i]) + dqxz * (qxz))

    # Quina Superior Esquerda
    for i in prange(4, N_abc):
        for j in range(4, N_abc):
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
            
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx 
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz                     
            psiqx = (a1 * (PsixqFL[j, i+1] - PsixqFL[j, i-1]) +
                    a2 * (PsixqFL[j, i+2] - PsixqFL[j, i-2]) +
                    a3 * (PsixqFL[j, i+3] - PsixqFL[j, i-3]) +
                    a4 * (PsixqFL[j, i+4] - PsixqFL[j, i-4])) / dx            
            psiqz = (a1 * (PsizqFU[j+1, i] - PsizqFU[j-1, i])+
                    a2 * (PsizqFU[j+2, i] - PsizqFU[j-2, i]) +
                    a3 * (PsizqFU[j+3, i] - PsizqFU[j-3, i]) +
                    a4 * (PsizqFU[j+4, i] - PsizqFU[j-4, i])) / dz
            psizx = (a1 * (PsizFU[j, i+1] - PsizFU[j, i-1]) +
                a2 * (PsizFU[j, i+2] - PsizFU[j, i-2]) +
                a3 * (PsizFU[j, i+3] - PsizFU[j, i-3]) +
                a4 * (PsizFU[j, i+4] - PsizFU[j, i-4])) / dx
            psiqzx = (a1 * (PsizqFU[j, i+1] - PsizqFU[j, i-1]) +
                a2 * (PsizqFU[j, i+2] - PsizqFU[j, i-2]) +
                a3 * (PsizqFU[j, i+3] - PsizqFU[j, i-3]) +
                a4 * (PsizqFU[j, i+4] - PsizqFU[j, i-4])) / dx
             
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cpx * (pxx + psix + ZetaxFL[j, i]) + cpz * (pzz + psiz + ZetazFU[j, i]) + cpxz * (pxz + psizx + ZetaxzFLU[j,i]) + dpx * (qxx + psiqx + ZetaxqFL[j, i]) + dpz * (qzz + psiqz + ZetazqFU[j,i]) + dpxz * (qxz + psiqzx + ZetaxzqFLU[j,i]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (cqx * (pxx + psix + ZetaxFL[j, i]) + cqz * (pzz + psiz + ZetazFU[j, i]) + cqxz * (pxz + psizx + ZetaxzFLU[j,i]) + dqx * (qxx + psiqx + ZetaxqFL[j, i]) + dqz * (qzz + psiqz + ZetazqFU[j,i]) + dqxz * (qxz + psiqzx + ZetaxzqFLU[j,i]))

    # Quina Superior Direita 
    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in range(4, N_abc):
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
            
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx 
            psiz = (a1 * (PsizFU[j+1, i] - PsizFU[j-1, i]) +
                    a2 * (PsizFU[j+2, i] - PsizFU[j-2, i]) +
                    a3 * (PsizFU[j+3, i] - PsizFU[j-3, i]) +
                    a4 * (PsizFU[j+4, i] - PsizFU[j-4, i])) / dz                     
            psiqx = (a1 * (PsixqFR[j, idx+1] - PsixqFR[j, idx-1]) +
                    a2 * (PsixqFR[j, idx+2] - PsixqFR[j, idx-2]) +
                    a3 * (PsixqFR[j, idx+3] - PsixqFR[j, idx-3]) +
                    a4 * (PsixqFR[j, idx+4] - PsixqFR[j, idx-4])) / dx
            psiqz = (a1 * (PsizqFU[j+1, i] - PsizqFU[j-1, i])+
                    a2 * (PsizqFU[j+2, i] - PsizqFU[j-2, i]) +
                    a3 * (PsizqFU[j+3, i] - PsizqFU[j-3, i]) +
                    a4 * (PsizqFU[j+4, i] - PsizqFU[j-4, i])) / dz
            psizx = (a1 * (PsizFU[j, i+1] - PsizFU[j, i-1]) +
                a2 * (PsizFU[j, i+2] - PsizFU[j, i-2]) +
                a3 * (PsizFU[j, i+3] - PsizFU[j, i-3]) +
                a4 * (PsizFU[j, i+4] - PsizFU[j, i-4])) / dx
            psiqzx = (a1 * (PsizqFU[j, i+1] - PsizqFU[j, i-1]) +
                a2 * (PsizqFU[j, i+2] - PsizqFU[j, i-2]) +
                a3 * (PsizqFU[j, i+3] - PsizqFU[j, i-3]) +
                a4 * (PsizqFU[j, i+4] - PsizqFU[j, i-4])) / dx

            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cpx * (pxx + psix + ZetaxFR[j, idx]) + cpz * (pzz + psiz + ZetazFU[j, i]) + cpxz * (pxz  + psizx + ZetaxzFRU[j,idx]) + dpx * (qxx + psiqx + ZetaxqFR[j, idx]) + dpz * (qzz + psiqz + ZetazqFU[j,i]) + dpxz * (qxz + psiqzx  + ZetaxzqFRU[j,idx]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (cqx * (pxx + psix + ZetaxFR[j, idx]) + cqz * (pzz + psiz + ZetazFU[j, i]) + cqxz * (pxz  + psizx + ZetaxzFRU[j,idx] ) + dqx * (qxx + psiqx + ZetaxqFR[j, idx])  + dqz * (qzz + psiqz + ZetazqFU[j,i]) + dqxz * (qxz + psiqzx + ZetaxzqFRU[j,idx]))

    # Quina Inferior Esquerda 
    for i in prange(4, N_abc):
        for j in range(nz_abc - N_abc, nz_abc - 4):
            jdx = j - (nz_abc - N_abc)

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
            
            psix = (a1 * (PsixFL[j, i+1] - PsixFL[j, i-1]) +
                    a2 * (PsixFL[j, i+2] - PsixFL[j, i-2]) +
                    a3 * (PsixFL[j, i+3] - PsixFL[j, i-3]) +
                    a4 * (PsixFL[j, i+4] - PsixFL[j, i-4])) / dx 
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz             
            psiqx = (a1 * (PsixqFL[j, i+1] - PsixqFL[j, i-1]) +
                    a2 * (PsixqFL[j, i+2] - PsixqFL[j, i-2]) +
                    a3 * (PsixqFL[j, i+3] - PsixqFL[j, i-3]) +
                    a4 * (PsixqFL[j, i+4] - PsixqFL[j, i-4])) / dx            
            psiqz = (a1 * (PsizqFD[jdx+1, i] - PsizqFD[jdx-1, i])+
                    a2 * (PsizqFD[jdx+2, i] - PsizqFD[jdx-2, i]) +
                    a3 * (PsizqFD[jdx+3, i] - PsizqFD[jdx-3, i]) +
                    a4 * (PsizqFD[jdx+4, i] - PsizqFD[jdx-4, i])) / dz
            psizx = (a1 * (PsizFD[jdx, i+1] - PsizFD[jdx, i-1]) +
                a2 * (PsizFD[jdx, i+2] - PsizFD[jdx, i-2]) +
                a3 * (PsizFD[jdx, i+3] - PsizFD[jdx, i-3]) +
                a4 * (PsizFD[jdx, i+4] - PsizFD[jdx, i-4])) / dx
            psiqzx = (a1 * (PsizqFD[jdx, i+1] - PsizqFD[jdx, i-1]) +
                a2 * (PsizqFD[jdx, i+2] - PsizqFD[jdx, i-2]) +
                a3 * (PsizqFD[jdx, i+3] - PsizqFD[jdx, i-3]) +
                a4 * (PsizqFD[jdx, i+4] - PsizqFD[jdx, i-4])) / dx
                           
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cpx * (pxx + psix + ZetaxFL[j, i]) + cpz * (pzz + psiz + ZetazFD[jdx, i]) + cpxz * (pxz + psizx + ZetaxzFLD[jdx,i]) + dpx * (qxx + psiqx + ZetaxqFL[j, i]) + dpz * (qzz + psiqz + ZetazqFD[jdx,i]) + dpxz * (qxz +  psiqzx  + ZetaxzqFLD[jdx,i]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (cqx * (pxx + psix + ZetaxFL[j, i]) + cqz * (pzz + psiz + ZetazFD[jdx, i]) + cqxz * (pxz + psizx + ZetaxzFLD[jdx,i]) + dqx * (qxx + psiqx + ZetaxqFL[j, i])  + dqz * (qzz + psiqz + ZetazqFD[jdx,i]) + dqxz * (qxz + psiqzx + ZetaxzqFLD[jdx,i]))

    # Quina Inferior Direita 
    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in range(nz_abc - N_abc, nz_abc - 4):
            jdx = j - (nz_abc - N_abc)

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
            
            psix = (a1 * (PsixFR[j, idx+1] - PsixFR[j, idx-1]) +
                    a2 * (PsixFR[j, idx+2] - PsixFR[j, idx-2]) +
                    a3 * (PsixFR[j, idx+3] - PsixFR[j, idx-3]) +
                    a4 * (PsixFR[j, idx+4] - PsixFR[j, idx-4])) / dx    
            psiz = (a1 * (PsizFD[jdx+1, i] - PsizFD[jdx-1, i]) +
                    a2 * (PsizFD[jdx+2, i] - PsizFD[jdx-2, i]) +
                    a3 * (PsizFD[jdx+3, i] - PsizFD[jdx-3, i]) +
                    a4 * (PsizFD[jdx+4, i] - PsizFD[jdx-4, i])) / dz          
            psiqx = (a1 * (PsixqFR[j, idx+1] - PsixqFR[j, idx-1]) +
                    a2 * (PsixqFR[j, idx+2] - PsixqFR[j, idx-2]) +
                    a3 * (PsixqFR[j, idx+3] - PsixqFR[j, idx-3]) +
                    a4 * (PsixqFR[j, idx+4] - PsixqFR[j, idx-4])) / dx            
            psiqz = (a1 * (PsizqFD[jdx+1, i] - PsizqFD[jdx-1, i])+
                    a2 * (PsizqFD[jdx+2, i] - PsizqFD[jdx-2, i]) +
                    a3 * (PsizqFD[jdx+3, i] - PsizqFD[jdx-3, i]) +
                    a4 * (PsizqFD[jdx+4, i] - PsizqFD[jdx-4, i])) / dz
            psizx = (a1 * (PsizFD[jdx, i+1] - PsizFD[jdx, i-1]) +
                a2 * (PsizFD[jdx, i+2] - PsizFD[jdx, i-2]) +
                a3 * (PsizFD[jdx, i+3] - PsizFD[jdx, i-3]) +
                a4 * (PsizFD[jdx, i+4] - PsizFD[jdx, i-4])) / dx
            psiqzx = (a1 * (PsizqFD[jdx, i+1] - PsizqFD[jdx, i-1]) +
                a2 * (PsizqFD[jdx, i+2] - PsizqFD[jdx, i-2]) +
                a3 * (PsizqFD[jdx, i+3] - PsizqFD[jdx, i-3]) +
                a4 * (PsizqFD[jdx, i+4] - PsizqFD[jdx, i-4])) / dx
          
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cpx * (pxx + psix + ZetaxFR[j, idx]) + cpz * (pzz + psiz + ZetazFD[jdx, i]) + cpxz * (pxz + psizx + ZetaxzFRD[jdx,idx]) + dpx * (qxx + psiqx + ZetaxqFR[j, idx]) + dpz * (qzz + psiqz + ZetazqFD[jdx,i]) + dpxz * (qxz + psiqzx + ZetaxzqFRD[jdx,idx]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (cqx * (pxx + psix + ZetaxFR[j, idx]) + cqz * (pzz + psiz + ZetazFD[jdx, i]) + cqxz * (pxz + psizx + ZetaxzFRD[jdx,idx]) + dqx * (qxx + psiqx + ZetaxqFR[j, idx]) + dqz * (qzz + psiqz + ZetazqFD[jdx,i]) + dqxz * (qxz + psiqzx + ZetaxzqFRD[jdx,idx]))

    return Uf, Qf

@jit(nopython=True, parallel=True)
def updatePsiTTI(PsixqFR, PsixqFL,PsizFU,PsizFD,PsizqFU,PsizqFD,PsiauxFL,PsiauxFR,PsiauxqFL,PsiauxqFR, nx_abc, nz_abc, a_x, b_x, Qc,Uc, dx,dz, N_abc):

    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    for i in prange(4, N_abc):
        for j in prange(4, nz_abc - 4):

            qx = (a1 * (Qc[j, i+1] - Qc[j, i-1]) +
                a2 * (Qc[j, i+2] - Qc[j, i-2]) +
                a3 * (Qc[j, i+3] - Qc[j, i-3]) +
                a4 * (Qc[j, i+4] - Qc[j, i-4])) / dx
        
            PsixqFL[j, i] = a_x[j,i] * PsixqFL[j, i] + b_x[j,i] * qx

    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in prange(4, nz_abc - 4):

            qx = (a1 * (Qc[j, i+1] - Qc[j, i-1]) +
                a2 * (Qc[j, i+2] - Qc[j, i-2]) +
                a3 * (Qc[j, i+3] - Qc[j, i-3]) +
                a4 * (Qc[j, i+4] - Qc[j, i-4])) / dx
        
            PsixqFR[j, idx] = a_x[j,i] * PsixqFR[j, idx] + b_x[j,i] * qx
    
    PsiauxFL[:N_abc + 4,:N_abc] = PsizFU[:N_abc + 4,:N_abc]
    PsiauxqFL[:N_abc + 4,:N_abc] = PsizqFU[:N_abc + 4,:N_abc]
    PsiauxFL[nz_abc - N_abc - 4:,:N_abc] = PsizFD[:,:N_abc]
    PsiauxqFL[nz_abc - N_abc - 4:,:N_abc] = PsizqFD[:,:N_abc]
    for i in prange(4, N_abc):
        for j in range(N_abc, nz_abc - N_abc):
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx

            qx = (a1*(Qc[j, i+1] - Qc[j, i-1]) +
                a2*(Qc[j, i+2] - Qc[j, i-2]) +
                a3*(Qc[j, i+3] - Qc[j, i-3]) +
                a4*(Qc[j, i+4] - Qc[j, i-4])) / dx
            
            PsiauxFL[j, i]  = a_x[j, i]* PsiauxFL[j, i] + b_x[j, i]*px
            PsiauxqFL[j, i] = a_x[j, i]* PsiauxqFL[j, i] + b_x[j, i]*qx

    PsiauxFR[:N_abc + 4, :N_abc] = PsizFU[:N_abc + 4, nx_abc - N_abc :]
    PsiauxqFR[:N_abc + 4, :N_abc] = PsizqFU[:N_abc + 4, nx_abc - N_abc :]
    PsiauxFR[nz_abc - N_abc - 4:, :N_abc] = PsizFD[:, nx_abc - N_abc :]
    PsiauxqFR[nz_abc - N_abc - 4:, :N_abc] = PsizqFD[:, nx_abc - N_abc :]
    
    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)  
        for j in range(N_abc, nz_abc - N_abc):
            px = (a1*(Uc[j, i+1] - Uc[j, i-1]) +
                a2*(Uc[j, i+2] - Uc[j, i-2]) +
                a3*(Uc[j, i+3] - Uc[j, i-3]) +
                a4*(Uc[j, i+4] - Uc[j, i-4])) / dx

            qx = (a1*(Qc[j, i+1] - Qc[j, i-1]) +
                a2*(Qc[j, i+2] - Qc[j, i-2]) +
                a3*(Qc[j, i+3] - Qc[j, i-3]) +
                a4*(Qc[j, i+4] - Qc[j, i-4])) / dx

            PsiauxFR[j, idx]  = a_x[j, i]*PsiauxFR[j, idx]  + b_x[j, i]*px
            PsiauxqFR[j, idx] = a_x[j, i]*PsiauxqFR[j, idx] + b_x[j, i]*qx

    return PsixqFR, PsixqFL, PsiauxFL,PsiauxFR,PsiauxqFL,PsiauxqFR

@jit(nopython=True, parallel=True)
def updateZetaTTI(PsixqFR, PsixqFL, PsizFU, PsizFD, PsizqFU, PsizqFD,PsiauxFL,PsiauxFR,PsiauxqFL,PsiauxqFR, ZetaxqFL, ZetaxqFR, ZetaxzFLU,ZetaxzFLD,ZetaxzFRU,ZetaxzFRD,ZetaxzqFLU,ZetaxzqFLD, ZetaxzqFRU,ZetaxzqFRD,ZetaxzFL, ZetaxzFR, ZetaxzqFL, ZetaxzqFR, nx_abc, nz_abc, a_x, b_x, Qc, Uc, dx, dz, N_abc):

    c0 = -205. / 72.
    c1 = 8. / 5.
    c2 = -1. / 5.
    c3 = 8. / 315.
    c4 = -1. / 560.
    a1 = 672. / 840.
    a2 = -168. / 840.
    a3 = 32. / 840.
    a4 = -3. / 840.

    for i in prange(4, N_abc):
        for j in prange(4, nz_abc - 4):

            qxx = (c0 * Qc[j, i] + c1 * (Qc[j, i+1] + Qc[j, i-1]) + 
                    c2 * (Qc[j, i+2] + Qc[j, i-2]) +
                    c3 * (Qc[j, i+3] + Qc[j, i-3]) + 
                    c4 * (Qc[j, i+4] + Qc[j, i-4])) / (dx * dx)
            
            psiqx = (a1 * (PsixqFL[j, i+1] - PsixqFL[j, i-1]) +
                a2 * (PsixqFL[j, i+2] - PsixqFL[j, i-2]) +
                a3 * (PsixqFL[j, i+3] - PsixqFL[j, i-3]) +
                a4 * (PsixqFL[j, i+4] - PsixqFL[j, i-4])) / dx
            
            ZetaxqFL[j, i] = a_x[j,i] * ZetaxqFL[j, i] + b_x[j,i] * (qxx + psiqx)

    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc) 
        for j in prange(4, nz_abc - 4):

            qxx = (c0 * Qc[j, i] + c1 * (Qc[j, i+1] + Qc[j, i-1]) + 
                    c2 * (Qc[j, i+2] + Qc[j, i-2]) +
                    c3 * (Qc[j, i+3] + Qc[j, i-3]) + 
                    c4 * (Qc[j, i+4] + Qc[j, i-4])) / (dx * dx)
            
            psiqx = (a1 * (PsixqFR[j, idx+1] - PsixqFR[j, idx-1]) +
                a2 * (PsixqFR[j, idx+2] - PsixqFR[j, idx-2]) +
                a3 * (PsixqFR[j, idx+3] - PsixqFR[j, idx-3]) +
                a4 * (PsixqFR[j, idx+4] - PsixqFR[j, idx-4])) / dx

            ZetaxqFR[j, idx] = a_x[j,i] * ZetaxqFR[j, idx] + b_x[j,i] * (qxx + psiqx)
    
    for i in prange(4, N_abc):
        for j in prange(4, N_abc):

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

            psizx = (a1 * (PsizFU[j, i+1] - PsizFU[j, i-1]) +
                a2 * (PsizFU[j, i+2] - PsizFU[j, i-2]) +
                a3 * (PsizFU[j, i+3] - PsizFU[j, i-3]) +
                a4 * (PsizFU[j, i+4] - PsizFU[j, i-4])) / dx
           
            ZetaxzFLU[j, i] = a_x[j,i] * ZetaxzFLU[j, i] + b_x[j,i] * (pxz + psizx)

    for i in prange(4, N_abc):
        for j in prange(nz_abc - N_abc, nz_abc - 4):
            jdx = j - (nz_abc - N_abc)

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

            psizx = (a1 * (PsizFD[jdx, i+1] - PsizFD[jdx, i-1]) +
                a2 * (PsizFD[jdx, i+2] - PsizFD[jdx, i-2]) +
                a3 * (PsizFD[jdx, i+3] - PsizFD[jdx, i-3]) +
                a4 * (PsizFD[jdx, i+4] - PsizFD[jdx, i-4])) / dx
           
            ZetaxzFLD[jdx, i] = a_x[j,i] * ZetaxzFLD[jdx, i] + b_x[j,i] * (pxz + psizx)

    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in prange(4, N_abc):

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

            psizx = (a1 * (PsizFU[j, i+1] - PsizFU[j, i-1]) +
                a2 * (PsizFU[j, i+2] - PsizFU[j, i-2]) +
                a3 * (PsizFU[j, i+3] - PsizFU[j, i-3]) +
                a4 * (PsizFU[j, i+4] - PsizFU[j, i-4])) / dx
           
            ZetaxzFRU[j, idx] = a_x[j,i] * ZetaxzFRU[j, idx] + b_x[j,i] * (pxz + psizx)

    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in prange(nz_abc - N_abc, nz_abc - 4):
            jdx = j - (nz_abc - N_abc)

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

            psizx = (a1 * (PsizFD[jdx, i+1] - PsizFD[jdx, i-1]) +
                a2 * (PsizFD[jdx, i+2] - PsizFD[jdx, i-2]) +
                a3 * (PsizFD[jdx, i+3] - PsizFD[jdx, i-3]) +
                a4 * (PsizFD[jdx, i+4] - PsizFD[jdx, i-4])) / dx
           
            ZetaxzFRD[jdx, idx] = a_x[j,i] * ZetaxzFRD[jdx, idx] + b_x[j,i] * (pxz + psizx)

    for i in prange(4, N_abc):
        for j in prange(4, N_abc):
            
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

            psiqzx = (a1 * (PsizqFU[j, i+1] - PsizqFU[j, i-1]) +
                a2 * (PsizqFU[j, i+2] - PsizqFU[j, i-2]) +
                a3 * (PsizqFU[j, i+3] - PsizqFU[j, i-3]) +
                a4 * (PsizqFU[j, i+4] - PsizqFU[j, i-4])) / dx

            ZetaxzqFLU[j, i] = a_x[j, i] * ZetaxzqFLU[j, i] + b_x[j, i] * (qxz + psiqzx)

    for i in prange(4, N_abc):
        for j in prange(nz_abc - N_abc, nz_abc - 4):
            jdx = j - (nz_abc - N_abc)
            
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

            psiqzx = (a1 * (PsizqFD[jdx, i+1] - PsizqFD[jdx, i-1]) +
                a2 * (PsizqFD[jdx, i+2] - PsizqFD[jdx, i-2]) +
                a3 * (PsizqFD[jdx, i+3] - PsizqFD[jdx, i-3]) +
                a4 * (PsizqFD[jdx, i+4] - PsizqFD[jdx, i-4])) / dx

            ZetaxzqFLD[jdx, i] = a_x[j, i] * ZetaxzqFLD[jdx, i] + b_x[j, i] * (qxz + psiqzx)

    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in prange(4, N_abc):
            
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

            psiqzx = (a1 * (PsizqFU[j, i+1] - PsizqFU[j, i-1]) +
                a2 * (PsizqFU[j, i+2] - PsizqFU[j, i-2]) +
                a3 * (PsizqFU[j, i+3] - PsizqFU[j, i-3]) +
                a4 * (PsizqFU[j, i+4] - PsizqFU[j, i-4])) / dx

            ZetaxzqFRU[j, idx] = a_x[j, i] * ZetaxzqFRU[j, idx] + b_x[j, i] * (qxz + psiqzx)

    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)
        for j in prange(nz_abc - N_abc, nz_abc - 4):
            jdx = j - (nz_abc - N_abc)
            
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

            psiqzx = (a1 * (PsizqFD[jdx, i+1] - PsizqFD[jdx, i-1]) +
                a2 * (PsizqFD[jdx, i+2] - PsizqFD[jdx, i-2]) +
                a3 * (PsizqFD[jdx, i+3] - PsizqFD[jdx, i-3]) +
                a4 * (PsizqFD[jdx, i+4] - PsizqFD[jdx, i-4])) / dx

            ZetaxzqFRD[jdx, idx] = a_x[j, i] * ZetaxzqFRD[jdx, idx] + b_x[j, i] * (qxz + psiqzx)
    
    for i in prange(4, N_abc):
        for j in prange(4, nz_abc - 4):

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
            
            psizx = (a1 * (PsiauxFL[j, i+1] - PsiauxFL[j, i-1]) +
                a2 * (PsiauxFL[j, i+2] - PsiauxFL[j, i-2]) +
                a3 * (PsiauxFL[j, i+3] - PsiauxFL[j, i-3]) +
                a4 * (PsiauxFL[j, i+4] - PsiauxFL[j, i-4])) / dx
            
            ZetaxzFL[j, i] = a_x[j,i] * ZetaxzFL[j, i] + b_x[j,i] * (pxz + psizx)

    for i in prange(4, N_abc):
        for j in prange(4, nz_abc - 4):

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
            
            psiqzx = (a1 * (PsiauxqFL[j, i+1] - PsiauxqFL[j, i-1]) +
                a2 * (PsiauxqFL[j, i+2] - PsiauxqFL[j, i-2]) +
                a3 * (PsiauxqFL[j, i+3] - PsiauxqFL[j, i-3]) +
                a4 * (PsiauxqFL[j, i+4] - PsiauxqFL[j, i-4])) / dx
            
            ZetaxzqFL[j, i] = a_x[j,i] * ZetaxzqFL[j, i] + b_x[j,i] * (qxz + psiqzx)

    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc)  
        for j in prange(4, nz_abc - 4):

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
            
            psizx = (a1 * (PsiauxFR[j, idx+1] - PsiauxFR[j, idx-1]) +
                a2 * (PsiauxFR[j, idx+2] - PsiauxFR[j, idx-2]) +
                a3 * (PsiauxFR[j, idx+3] - PsiauxFR[j, idx-3]) +
                a4 * (PsiauxFR[j, idx+4] - PsiauxFR[j, idx-4])) / dx
            
            ZetaxzFR[j, idx] = a_x[j,i] * ZetaxzFR[j, idx] + b_x[j,i] * (pxz + psizx)

    for i in prange(nx_abc - N_abc, nx_abc - 4):
        idx = i - (nx_abc - N_abc) 
        for j in prange(4, nz_abc - 4):

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
            
            psiqzx = (a1 * (PsiauxqFR[j, idx+1] - PsiauxqFR[j, idx-1]) +
                a2 * (PsiauxqFR[j, idx+2] - PsiauxqFR[j, idx-2]) +
                a3 * (PsiauxqFR[j, idx+3] - PsiauxqFR[j, idx-3]) +
                a4 * (PsiauxqFR[j, idx+4] - PsiauxqFR[j, idx-4])) / dx
            
            ZetaxzqFR[j, idx] = a_x[j,i] * ZetaxzqFR[j, idx] + b_x[j,i] * (qxz + psiqzx)

    return ZetaxqFL, ZetaxqFR, ZetaxzFLU, ZetaxzFLD, ZetaxzFRU, ZetaxzFRD, ZetaxzqFLU, ZetaxzqFLD, ZetaxzqFRU, ZetaxzqFRD, ZetaxzFL, ZetaxzFR, ZetaxzqFL, ZetaxzqFR 

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
