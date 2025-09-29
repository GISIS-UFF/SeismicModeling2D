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
    c0 = -205 / 72
    c1 = 8 / 5
    c2 = -1 / 5
    c3 = 8 / 315
    c4 = -1 / 560
    for i in prange(4,nx-4):
        for j in prange(4,nz-4):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) + c2 * (Uc[j, i+2] + Uc[j, i-2]) +c3 * (Uc[j, i+3] + Uc[j, i-3]) +c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) + c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) + c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz) + 2 * Uc[j, i] - Uf[j, i]

    return Uf

@jit(nopython=True, parallel=True)
def updateWaveEquationCPML(Uf, Uc, vp, nx_abc, nz_abc, dz, dx, dt, PsixF, PsizF, ZetaxF, ZetazF):
    
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
        for j in prange(4, nz_abc - 4):
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                    c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                    c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)                                
            psix = (a1 * (PsixF[j, i+1] - PsixF[j, i-1]) +
                    a2 * (PsixF[j, i+2] - PsixF[j, i-2]) +
                    a3 * (PsixF[j, i+3] - PsixF[j, i-3]) +
                    a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / (2 * dx)
            psiz = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
                    a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
                    a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
                    a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / (2*dz)
            
            Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + psiz + ZetaxF[j, i] + ZetazF[j, i]) + 2 * Uc[j, i] - Uf[j, i]

    return Uf

@jit(nopython=True, parallel=True)
def updatePsi (PsixF, PsizF, nx_abc, nz_abc, a_z, a_x, b_z, b_x, Uc, dz, dx):

    a1 = 4. / 5.
    a2 = -1. / 5.
    a3 = 4. / 105.
    a4 = -1. / 280.

    for i in prange(4, nx_abc - 4):
            for j in prange(4, nz_abc - 4):

                px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
                    a2 * (Uc[j, i+2] - Uc[j, i-2]) +
                    a3 * (Uc[j, i+3] - Uc[j, i-3]) +
                    a4 * (Uc[j, i+4] - Uc[j, i-4])) / (2 * dx)
                pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                    a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                    a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                    a4 * (Uc[j+4, i] - Uc[j-4, i])) / (2 * dz) 
                
                PsizF[j, i] = a_z[j] * PsizF[j, i] + b_z[j] * pz
                PsixF[j, i] = a_x[i] * PsixF[j, i] + b_x[i] * px

    return PsixF, PsizF

@jit(nopython=True, parallel=True)
def updateZeta(PsixF, PsizF, ZetaxF, ZetazF, nx_abc, nz_abc, a_z, a_x, b_z, b_x, Uc, dz, dx):

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
            for j in prange(4, nz_abc - 4):
                
                pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                    c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                    c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
                pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + 
                    c3 * (Uc[j+3, i] + Uc[j-3, i]) +
                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)               
                psix = (a1 * (PsixF[j, i+1] - PsixF[j, i-1]) +
                        a2 * (PsixF[j, i+2] - PsixF[j, i-2]) +
                        a3 * (PsixF[j, i+3] - PsixF[j, i-3]) +
                        a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / (2 * dx)
                psiz = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
                        a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
                        a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
                        a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / (2*dz)
                
                ZetazF[j, i] = a_z[j] * ZetazF[j, i] + b_z[j] * (pzz + psiz)
                ZetaxF[j, i] = a_x[i] * ZetaxF[j, i] + b_x[i] * (pxx + psix)  

    return ZetaxF, ZetazF

# @jit(nopython=True, parallel=True)
# def updateWaveEquationCPML(Uf, Uc, vp, nx_abc, nz_abc, N_abc, dz, dx, dt,
#                           PsixF, PsizF, ZetaxF, ZetazF, a_x, a_z, b_x, b_z):
    
#     c0 = -205 / 72
#     c1 = 8 / 5
#     c2 = -1 / 5
#     c3 = 8 / 315
#     c4 = -1 / 560
#     a1 = 4 / 5
#     a2 = -1 / 5
#     a3 = 4 / 105
#     a4 = -1 / 280

#     # Região Interior 
#     for i in prange(N_abc, nx_abc - N_abc):
#         for j in range(N_abc, nz_abc - N_abc):
#             pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
#                    c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
#                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
#                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz) + 2 * Uc[j, i] - Uf[j, i]

#     # Região Esquerda 
#     for i in prange(4, N_abc):
#         for j in range(N_abc, nz_abc - N_abc):
#             pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
#                    c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
#                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
#                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
#                   a2 * (Uc[j, i+2] - Uc[j, i-2]) +
#                   a3 * (Uc[j, i+3] - Uc[j, i-3]) +
#                   a4 * (Uc[j, i+4] - Uc[j, i-4])) / (2 * dx)
#             psix = (a1 * (PsixF[j, i+1] - PsixF[j, i-1]) +
#                     a2 * (PsixF[j, i+2] - PsixF[j, i-2]) +
#                     a3 * (PsixF[j, i+3] - PsixF[j, i-3]) +
#                     a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / (2 * dx)
            
#             PsixF[j, i] = a_x[i] * PsixF[j, i] + b_x[i] * px
#             ZetaxF[j, i] = a_x[i] * ZetaxF[j, i] + b_x[i] * (pxx + psix)
#             Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + ZetaxF[j, i]) + 2 * Uc[j, i] - Uf[j, i]
            
#     # Região Direita
#     for i in prange(nx_abc - N_abc, nx_abc - 4):
#             for j in range(N_abc, nz_abc - N_abc):
#                 pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
#                     c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                     c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#                 pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
#                     c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
#                     c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#                 px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
#                     a2 * (Uc[j, i+2] - Uc[j, i-2]) +
#                     a3 * (Uc[j, i+3] - Uc[j, i-3]) +
#                     a4 * (Uc[j, i+4] - Uc[j, i-4])) / (2 * dx)
#                 psix = (a1 * (PsixF[j, i+1] - PsixF[j, i-1]) +
#                         a2 * (PsixF[j, i+2] - PsixF[j, i-2]) +
#                         a3 * (PsixF[j, i+3] - PsixF[j, i-3]) +
#                         a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / (2 * dx)
                
#                 PsixF[j, i] = a_x[i] * PsixF[j, i] + b_x[i] * px
#                 ZetaxF[j, i] = a_x[i] * ZetaxF[j, i] + b_x[i] * (pxx + psix)
#                 Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + ZetaxF[j, i]) + 2 * Uc[j, i] - Uf[j, i]

#     # Região Superior 
#     for j in prange(4, N_abc):
#         for i in range(N_abc, nx_abc - N_abc):
#             pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
#                    c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
#                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
#                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
#                   a2 * (Uc[j+2, i] - Uc[j-2, i]) +
#                   a3 * (Uc[j+3, i] - Uc[j-3, i]) +
#                   a4 * (Uc[j+4, i] - Uc[j-4, i])) / (2 * dz)
#             psiz = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
#                     a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
#                     a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
#                     a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / (2*dz)          

#             PsizF[j, i] = a_z[j] * PsizF[j, i] + b_z[j] * pz
#             ZetazF[j, i] = a_z[j] * ZetazF[j, i] + b_z[j] * (pzz + psiz)
#             Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psiz + ZetazF[j, i]) + 2 * Uc[j, i] - Uf[j, i]

#     # Região Inferior
#     for j in prange(nz_abc - N_abc, nz_abc - 4):
#         for i in range(N_abc, nx_abc - N_abc):
#             pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
#                    c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
#                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
#                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
#                   a2 * (Uc[j+2, i] - Uc[j-2, i]) +
#                   a3 * (Uc[j+3, i] - Uc[j-3, i]) +
#                   a4 * (Uc[j+4, i] - Uc[j-4, i])) / (2 * dz)
#             psiz = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
#                     a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
#                     a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
#                     a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / (2*dz)
            
#             PsizF[j, i] = a_z[j] * PsizF[j, i] + b_z[j] * pz
#             ZetazF[j, i] = a_z[j] * ZetazF[j, i] + b_z[j] * (pzz + psiz)
#             Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psiz + ZetazF[j, i]) + 2 * Uc[j, i] - Uf[j, i]

#     # Quina Superior Esquerda
#     for i in prange(4, N_abc):
#         for j in range(4, N_abc):
#             pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
#                    c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
#                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
#                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
#                   a2 * (Uc[j, i+2] - Uc[j, i-2]) +
#                   a3 * (Uc[j, i+3] - Uc[j, i-3]) +
#                   a4 * (Uc[j, i+4] - Uc[j, i-4])) / (2 * dx)
#             pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
#                   a2 * (Uc[j+2, i] - Uc[j-2, i]) +
#                   a3 * (Uc[j+3, i] - Uc[j-3, i]) +
#                   a4 * (Uc[j+4, i] - Uc[j-4, i])) / (2 * dz)
#             psiz = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
#                     a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
#                     a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
#                     a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / (2*dz)   
#             psix = (a1 * (PsixF[j, i+1] - PsixF[j, i-1]) +
#                     a2 * (PsixF[j, i+2] - PsixF[j, i-2]) +
#                     a3 * (PsixF[j, i+3] - PsixF[j, i-3]) +
#                     a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / (2 * dx)
            
#             PsixF[j, i] = a_x[i] * PsixF[j, i] + b_x[i] * px
#             ZetaxF[j, i] = a_x[i] * ZetaxF[j, i] + b_x[i] * (pxx + psix)
#             PsizF[j, i] = a_z[j] * PsizF[j, i] + b_z[j] * pz
#             ZetazF[j, i] = a_z[j] * ZetazF[j, i] + b_z[j] * (pzz + psiz)
            
#             Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + psiz + ZetaxF[j, i] + ZetazF[j, i]) + 2 * Uc[j, i] - Uf[j, i]

#     # Quina Superior Direita 
#     for i in prange(nx_abc - N_abc, nx_abc - 4):
#         for j in range(4, N_abc):
#             pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
#                    c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
#                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
#                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
#                   a2 * (Uc[j, i+2] - Uc[j, i-2]) +
#                   a3 * (Uc[j, i+3] - Uc[j, i-3]) +
#                   a4 * (Uc[j, i+4] - Uc[j, i-4])) / (2 * dx)
#             pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
#                   a2 * (Uc[j+2, i] - Uc[j-2, i]) +
#                   a3 * (Uc[j+3, i] - Uc[j-3, i]) +
#                   a4 * (Uc[j+4, i] - Uc[j-4, i])) / (2 * dz)
#             psix = (a1 * (PsixF[j, i+1] - PsixF[j, i-1]) +
#                         a2 * (PsixF[j, i+2] - PsixF[j, i-2]) +
#                         a3 * (PsixF[j, i+3] - PsixF[j, i-3]) +
#                         a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / (2 * dx)
#             psiz = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
#                     a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
#                     a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
#                     a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / (2*dz)          

#             PsixF[j, i] = a_x[i] * PsixF[j, i] + b_x[i] * px
#             ZetaxF[j, i] = a_x[i] * ZetaxF[j, i] + b_x[i] * (pxx + psix)
#             PsizF[j, i] = a_z[j] * PsizF[j, i] + b_z[j] * pz
#             ZetazF[j, i] = a_z[j] * ZetazF[j, i] + b_z[j] * (pzz + psiz)
            
#             Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + psiz + ZetaxF[j, i] + ZetazF[j, i]) + 2 * Uc[j, i] - Uf[j, i]

#     # Quina Inferior Esquerda 
#     for i in prange(4, N_abc):
#         for j in range(nz_abc - N_abc, nz_abc - 4):

#             pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
#                    c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
#                    c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
#                    c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
#                   a2 * (Uc[j, i+2] - Uc[j, i-2]) +
#                   a3 * (Uc[j, i+3] - Uc[j, i-3]) +
#                   a4 * (Uc[j, i+4] - Uc[j, i-4])) / (2 * dx)
#             psix = (a1 * (PsixF[j, i+1] - PsixF[j, i-1]) +
#                     a2 * (PsixF[j, i+2] - PsixF[j, i-2]) +
#                     a3 * (PsixF[j, i+3] - PsixF[j, i-3]) +
#                     a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / (2 * dx)
#             pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
#                   a2 * (Uc[j+2, i] - Uc[j-2, i]) +
#                   a3 * (Uc[j+3, i] - Uc[j-3, i]) +
#                   a4 * (Uc[j+4, i] - Uc[j-4, i])) / (2 * dz)
#             psiz = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
#                     a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
#                     a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
#                     a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / (2*dz)

#             PsixF[j, i] = a_x[i] * PsixF[j, i] + b_x[i] * px
#             ZetaxF[j, i] = a_x[i] * ZetaxF[j, i] + b_x[i] * (pxx + psix)
#             PsizF[j, i] = a_z[j] * PsizF[j, i] + b_z[j] * pz
#             ZetazF[j, i] = a_z[j] * ZetazF[j, i] + b_z[j] * (pzz + psiz)
            
#             Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + psiz + ZetaxF[j, i] + ZetazF[j, i]) + 2 * Uc[j, i] - Uf[j, i]

#     # Quina Inferior Direita 
#     for i in prange(nx_abc - N_abc, nx_abc - 4):
#         for j in range(nz_abc - N_abc, nz_abc - 4):

#             pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
#                     c2 * (Uc[j, i+2] + Uc[j, i-2]) + c3 * (Uc[j, i+3] + Uc[j, i-3]) +
#                     c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
#             pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) +
#                 c2 * (Uc[j+2, i] + Uc[j-2, i]) + c3 * (Uc[j+3, i] + Uc[j-3, i]) +
#                 c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dz)
#             px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
#                 a2 * (Uc[j, i+2] - Uc[j, i-2]) +
#                 a3 * (Uc[j, i+3] - Uc[j, i-3]) +
#                 a4 * (Uc[j, i+4] - Uc[j, i-4])) / (2 * dx)
#             psix = (a1 * (PsixF[j, i+1] - PsixF[j, i-1]) +
#                     a2 * (PsixF[j, i+2] - PsixF[j, i-2]) +
#                     a3 * (PsixF[j, i+3] - PsixF[j, i-3]) +
#                     a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / (2 * dx)
#             pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
#                   a2 * (Uc[j+2, i] - Uc[j-2, i]) +
#                   a3 * (Uc[j+3, i] - Uc[j-3, i]) +
#                   a4 * (Uc[j+4, i] - Uc[j-4, i])) / (2 * dz)

#             psiz = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
#                     a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
#                     a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
#                     a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / (2*dz)
            
#             PsizF[j, i] = a_z[j] * PsizF[j, i] + b_z[j] * pz
#             ZetazF[j, i] = a_z[j] * ZetazF[j, i] + b_z[j] * (pzz + psiz)
#             PsixF[j, i] = a_x[i] * PsixF[j, i] + b_x[i] * px
#             ZetaxF[j, i] = a_x[i] * ZetaxF[j, i] + b_x[i] * (pxx + psix)
            
#             Uf[j, i] = (vp[j, i] ** 2) * (dt ** 2) * (pxx + pzz + psix + psiz + ZetaxF[j, i] + ZetazF[j, i]) + 2 * Uc[j, i] - Uf[j, i]

#     return Uf

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
                               nx_abc, nz_abc, PsixF, PsizqF, ZetaxF, ZetazqF):
    
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
        for j in range(4, nz_abc - 4):
            cx = vpz[j,i]**2 * (1 + 2 * epsilon[j,i])
            bx = vpz[j,i]**2 * (1 + 2 * delta[j,i])
            cz = bz = vpz[j,i]**2
            pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) + c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) + c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
            qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                   c3 * (Qc[j+3, i] + Qc[j-3, i]) + c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
            psiqz = (a1 * (PsizqF[j+1, i] - PsizqF[j-1, i]) +
                    a2 * (PsizqF[j+2, i] - PsizqF[j-2, i]) +
                    a3 * (PsizqF[j+3, i] - PsizqF[j-3, i]) +
                    a4 * (PsizqF[j+4, i] - PsizqF[j-4, i])) / (2*dz)
            psix = (a1 * (PsixF[j, i+1] - PsixF[j, i-1]) +
                    a2 * (PsixF[j, i+2] - PsixF[j, i-2]) +
                    a3 * (PsixF[j, i+3] - PsixF[j, i-3]) +
                    a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / (2 * dx)           
                  
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cx * (pxx + psix + ZetaxF[j,i] ) + cz *(qzz + psiqz + ZetazqF[j,i]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (bx * (pxx + psix + ZetaxF[j,i]) + bz *(qzz + psiqz + ZetazqF[j,i]))

    return Uf, Qf

@jit(nopython=True, parallel=True)
def updatePsiVTI (PsixF, PsizqF, nx_abc, nz_abc, a_z, a_x, b_z, b_x, Uc, Qc, dz, dx):

    a1 = 4. / 5.
    a2 = -1. / 5.
    a3 = 4. / 105.
    a4 = -1. / 280.

    for i in prange(4, nx_abc - 4):
            for j in prange(4, nz_abc - 4):

                px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
                    a2 * (Uc[j, i+2] - Uc[j, i-2]) +
                    a3 * (Uc[j, i+3] - Uc[j, i-3]) +
                    a4 * (Uc[j, i+4] - Uc[j, i-4])) / (2 * dx)
                qz = (a1 * (Qc[j+1, i] - Qc[j-1, i]) +
                    a2 * (Qc[j+2, i] - Qc[j-2, i]) +
                    a3 * (Qc[j+3, i] - Qc[j-3, i]) +
                    a4 * (Qc[j+4, i] - Qc[j-4, i])) / (2 * dz) 
                
                PsizqF[j, i] = a_z[j] * PsizqF[j, i] + b_z[j] * qz
                PsixF[j, i] = a_x[i] * PsixF[j, i] + b_x[i] * px

    return PsixF, PsizqF

@jit(nopython=True, parallel=True)
def updateZetaVTI(PsixF, PsizqF, ZetaxF, ZetazqF, nx_abc, nz_abc, a_z, a_x, b_z, b_x, Uc, Qc, dz, dx):

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
            for j in prange(4, nz_abc - 4):
                
                pxx = (c0 * Uc[j, i] + c1 * (Uc[j, i+1] + Uc[j, i-1]) +
                    c2 * (Uc[j, i+2] + Uc[j, i-2]) + 
                    c3 * (Uc[j, i+3] + Uc[j, i-3]) +
                    c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dx * dx)
                qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + 
                        c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                        c3 * (Qc[j+3, i] + Qc[j-3, i]) + 
                        c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
                psix = (a1 * (PsixF[j, i+1] - PsixF[j, i-1]) +
                        a2 * (PsixF[j, i+2] - PsixF[j, i-2]) +
                        a3 * (PsixF[j, i+3] - PsixF[j, i-3]) +
                        a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / (2 * dx)
                psiqz = (a1 * (PsizqF[j+1, i] - PsizqF[j-1, i]) +
                    a2 * (PsizqF[j+2, i] - PsizqF[j-2, i]) +
                    a3 * (PsizqF[j+3, i] - PsizqF[j-3, i]) +
                    a4 * (PsizqF[j+4, i] - PsizqF[j-4, i])) / (2*dz)

                ZetazqF[j, i] = a_z[j] * ZetazqF[j, i] + b_z[j] * (qzz + psiqz)
                ZetaxF[j, i] = a_x[i] * ZetaxF[j, i] + b_x[i] * (pxx + psix)  

    return ZetaxF, ZetazqF

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
def updateWaveEquationTTICPML(Uf, Uc, Qc, Qf, nx, nz, dt, dx, dz, vpz, vsz, epsilon, delta, theta,PsixF,PsizF,PsixqF,PsizqF,ZetaxF,ZetazF,ZetaxzF,ZetaxqF,ZetazqF,ZetaxzqF):
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
                       
            psix = (a1 * (PsixF[j, i+1] - PsixF[j, i-1]) +
                    a2 * (PsixF[j, i+2] - PsixF[j, i-2]) +
                    a3 * (PsixF[j, i+3] - PsixF[j, i-3]) +
                    a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / (2 * dx)
            
            psiz = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
                    a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
                    a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
                    a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / (2*dz)          
            
            psizx = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
                    a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
                    a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
                    a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / (2*dx)

            psiqx = (a1 * (PsixqF[j+1, i] - PsixqF[j-1, i]) +
                    a2 * (PsixqF[j+2, i] - PsixqF[j-2, i]) +
                    a3 * (PsixqF[j+3, i] - PsixqF[j-3, i]) +
                    a4 * (PsixqF[j+4, i] - PsixqF[j-4, i])) / (2*dx)
            
            psiqz = (a1 * (PsizqF[j+1, i] - PsizqF[j-1, i]) +
                    a2 * (PsizqF[j+2, i] - PsizqF[j-2, i]) +
                    a3 * (PsizqF[j+3, i] - PsizqF[j-3, i]) +
                    a4 * (PsizqF[j+4, i] - PsizqF[j-4, i])) / (2*dz)

            psiqzx = (a1 * (PsizqF[j+1, i] - PsizqF[j-1, i]) +
                    a2 * (PsizqF[j+2, i] - PsizqF[j-2, i]) +
                    a3 * (PsizqF[j+3, i] - PsizqF[j-3, i]) +
                    a4 * (PsizqF[j+4, i] - PsizqF[j-4, i])) / (2*dx)
            
            Uf[j, i] = 2 * Uc[j, i] - Uf[j, i] + (dt**2) * (cpx * (pxx + psix + ZetaxF[j, i]) + cpz * (pzz + psiz + ZetazF[j, i]) + cpxz * (pxz + psizx + ZetaxzF[j,i]) + dpx * (qxx + psiqx + ZetaxqF[j, i]) + dpz * (qzz + psiqz + ZetazqF[j,i]) + dpxz * (qxz + psiqzx + ZetaxzqF[j,i]))
            Qf[j, i] = 2 * Qc[j, i] - Qf[j, i] + (dt**2) * (cqx * (pxx + psix + ZetaxF[j, i]) + cqz * (pzz + psiz + ZetazF[j, i]) + cqxz * (pxz + psizx + ZetaxzF[j,i]) + dqx * (qxx + psiqx + ZetaxqF[j, i]) + dqz * (qzz + psiqz + ZetazqF[j,i]) + dqxz * (qxz + psiqzx + ZetaxzqF[j,i]))

    return Uf, Qf

@jit(nopython=True, parallel=True)
def updatePsiTTI (PsixF, PsixqF, PsizF, PsizqF, nx_abc, nz_abc, a_z, a_x, b_z, b_x, Uc, Qc, dz, dx):

    a1 = 4. / 5.
    a2 = -1. / 5.
    a3 = 4. / 105.
    a4 = -1. / 280.

    for i in prange(4, nx_abc - 4):
            for j in prange(4, nz_abc - 4):

                px = (a1 * (Uc[j, i+1] - Uc[j, i-1]) +
                    a2 * (Uc[j, i+2] - Uc[j, i-2]) +
                    a3 * (Uc[j, i+3] - Uc[j, i-3]) +
                    a4 * (Uc[j, i+4] - Uc[j, i-4])) / (2 * dx)
                pz = (a1 * (Uc[j+1, i] - Uc[j-1, i]) +
                    a2 * (Uc[j+2, i] - Uc[j-2, i]) +
                    a3 * (Uc[j+3, i] - Uc[j-3, i]) +
                    a4 * (Uc[j+4, i] - Uc[j-4, i])) / (2 * dz)
                qz = (a1 * (Qc[j+1, i] - Qc[j-1, i]) +
                    a2 * (Qc[j+2, i] - Qc[j-2, i]) +
                    a3 * (Qc[j+3, i] - Qc[j-3, i]) +
                    a4 * (Qc[j+4, i] - Qc[j-4, i])) / (2 * dz) 
                qx = (a1 * (Qc[j, i+1] - Qc[j, i-1]) +
                    a2 * (Qc[j, i+2] - Qc[j, i-2]) +
                    a3 * (Qc[j, i+3] - Qc[j, i-3]) +
                    a4 * (Qc[j, i+4] - Qc[j, i-4])) / (2 * dx)
            
                PsixqF[j, i] = a_x[i] * PsixqF[j, i] + b_x[i] * qx
                PsizqF[j, i] = a_z[j] * PsizqF[j, i] + b_z[j] * qz
                PsixF[j, i] = a_x[i] * PsixF[j, i] + b_x[i] * px
                PsizF[j, i] = a_z[i] * PsizF[j, i] + b_z[i] * pz

    return PsixF, PsixqF, PsizF, PsizqF

@jit(nopython=True, parallel=True)
def updateZetaTTI(PsixF, PsizF, PsizqF, PsixqF, ZetaxF, ZetazF, ZetaxzF, ZetaxqF, ZetazqF, ZetaxzqF, nx_abc, nz_abc, a_z, a_x, b_z, b_x, Uc, Qc, dz, dx):

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
            for j in prange(4, nz_abc - 4):
                
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
                qzz = (c0 * Qc[j, i] + c1 * (Qc[j+1, i] + Qc[j-1, i]) + 
                        c2 * (Qc[j+2, i] + Qc[j-2, i]) + 
                        c3 * (Qc[j+3, i] + Qc[j-3, i]) + 
                        c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dz)
                qxx = (c0 * Qc[j, i] + c1 * (Qc[j, i+1] + Qc[j, i-1]) + 
                        c2 * (Qc[j, i+2] + Qc[j, i-2]) +
                        c3 * (Qc[j, i+3] + Qc[j, i-3]) + 
                        c4 * (Qc[j, i+4] + Qc[j, i-4])) / (dx * dx)
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
                psix = (a1 * (PsixF[j, i+1] - PsixF[j, i-1]) +
                        a2 * (PsixF[j, i+2] - PsixF[j, i-2]) +
                        a3 * (PsixF[j, i+3] - PsixF[j, i-3]) +
                        a4 * (PsixF[j, i+4] - PsixF[j, i-4])) / (2 * dx)
                psiz = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
                    a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
                    a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
                    a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / (2*dz)
                psizx = (a1 * (PsizF[j+1, i] - PsizF[j-1, i]) +
                    a2 * (PsizF[j+2, i] - PsizF[j-2, i]) +
                    a3 * (PsizF[j+3, i] - PsizF[j-3, i]) +
                    a4 * (PsizF[j+4, i] - PsizF[j-4, i])) / (2*dx)
                psiqz = (a1 * (PsizqF[j+1, i] - PsizqF[j-1, i]) +
                    a2 * (PsizqF[j+2, i] - PsizqF[j-2, i]) +
                    a3 * (PsizqF[j+3, i] - PsizqF[j-3, i]) +
                    a4 * (PsizqF[j+4, i] - PsizqF[j-4, i])) / (2*dz)
                psiqx = (a1 * (PsixqF[j+1, i] - PsixqF[j-1, i]) +
                    a2 * (PsixqF[j+2, i] - PsixqF[j-2, i]) +
                    a3 * (PsixqF[j+3, i] - PsixqF[j-3, i]) +
                    a4 * (PsixqF[j+4, i] - PsixqF[j-4, i])) / (2*dx)
                psiqzx = (a1 * (PsizqF[j+1, i] - PsizqF[j-1, i]) +
                    a2 * (PsizqF[j+2, i] - PsizqF[j-2, i]) +
                    a3 * (PsizqF[j+3, i] - PsizqF[j-3, i]) +
                    a4 * (PsizqF[j+4, i] - PsizqF[j-4, i])) / (2*dx)

                ZetaxF[j, i] = a_x[i] * ZetaxF[j, i] + b_x[i] * (pxx + psix)
                ZetazF[j, i] = a_z[i] * ZetazF[j, i] + b_z[i] * (pzz + psiz)
                ZetaxzF[j, i] = a_x[i] * ZetaxzF[j, i] + b_x[i] *(pxz + psizx)
                ZetaxqF[j, i] = a_x[i] * ZetaxqF[j, i] + b_x[i] * (qxx + psiqx)
                ZetazqF[j, i] = a_z[j] * ZetazqF[j, i] + b_z[j] * (qzz + psiqz)
                ZetaxzqF[j, i] = a_x[i] * ZetaxzqF[j, i] + b_x[i] *(qxz + psiqzx)

    return ZetaxF, ZetazF, ZetaxzF, ZetaxqF, ZetazqF, ZetaxzqF

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

