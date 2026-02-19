import numpy as np
from numba import jit,prange, njit, cuda
import math

def ricker(f0, t, t_lag):
    pi = np.pi
    f = f0 / (np.sqrt(pi) * 3) 
    td  = t - t_lag
    source = (1 - 2 * pi * (pi * f * td) * (pi * f * td)) * np.exp(-pi * (pi * f * td) * (pi * f * td)) 
    return source

@jit(parallel=True)
def Mute(seismogram, shot, rec_x, rec_z, shot_x, shot_z, dt,window = 0.2 ,v0=1500): 
    result = np.zeros_like(seismogram)
    Nt = seismogram.shape[0]
    Nrec = seismogram.shape[1]  
    dist = np.sqrt((rec_z - shot_z[shot])**2 + (rec_x - shot_x[shot])**2)
    traveltimes = dist/v0
    for rec in prange(Nrec):
        t1 = traveltimes[rec]
        t2 = traveltimes[rec] + window
        t3 = (Nt - 1) * dt - window
        t4 = (Nt - 1) * dt
        for i in prange(Nt):
            t = (i-1)*dt
            if t <=t1:
                result[i,rec] = 0.0
            elif t>t1 and t<t2:
                result[i,rec] = (t-t1)/(t2-t1)*seismogram[i,rec]
            elif t>=t2 and t<t3:
                result[i,rec] = seismogram[i,rec]
            elif t>=t3 and t<t4:
                result[i,rec] = (t4-t)/(t4-t3)*seismogram[i,rec]
            elif t>t4:
                result[i,rec] = 0.0
    return result

@njit(inline = "always")
def horizontal_dampening_profiles(N_abc,nx_abc, dx, vp, f_pico, d0, dt, i, j):
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

@jit(parallel=True, nopython=True)
def AbsorbingBoundary(N_abc, nz_abc, nx_abc, f, A):
    for y in prange(nz_abc):
        for x in range(N_abc):
            f[y, x] *= A[x]
        for x in range(nx_abc - N_abc, nx_abc):
            f[y, x] *= A[nx_abc - 1 - x] 
    for x in prange(nx_abc):
        for y in range(N_abc):
            f[y, x] *= A[y]
        for y in range(nz_abc - N_abc, nz_abc):
            f[y, x] *= A[nz_abc - 1 - y]  

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