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

@jit(parallel=True, nopython=True)
def updateWaveEquationVTI(Uf, Uc, Qc, Qf, nx, nz, dt, dx, dz, vpz, epsilon, delta):  
    c0 = -205 / 72
    c1 = 8 / 5
    c2 = -1 / 5
    c3 = 8 / 315
    c4 = -1 / 560
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
def updateWaveEquationTTI(Uf, Uc, Qc, Qf, nx, nz, dt, dx, dz, vpz, vsz, epsilon, delta, theta):
    c0 = -205 / 72
    c1 = 8 / 5
    c2 = -1 / 5
    c3 = 8 / 315
    c4 = -1 / 560
    a1 = 4 / 5
    a2 = -1 / 5
    a3 = 4/105
    a4 = -1/280
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
                   c3 * (Uc[j, i+3] + Uc[j, i-3]) + c4 * (Uc[j, i+4] + Uc[j, i-4])) / (dz * dx)
            
            pzz = (c0 * Uc[j, i] + c1 * (Uc[j+1, i] + Uc[j-1, i]) + c2 * (Uc[j+2, i] + Uc[j-2, i]) +
                   c3 * (Uc[j+3, i] + Uc[j-3, i]) + c4 * (Uc[j+4, i] + Uc[j-4, i])) / (dz * dx)
            
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
                   c3 * (Qc[j+3, i] + Qc[j-3, i]) + c4 * (Qc[j+4, i] + Qc[j-4, i])) / (dz * dx)
            
            qxx = (c0 * Qc[j, i] + c1 * (Qc[j, i+1] + Qc[j, i-1]) + c2 * (Qc[j, i+2] + Qc[j, i-2]) +
                   c3 * (Qc[j, i+3] + Qc[j, i-3]) + c4 * (Qc[j, i+4] + Qc[j, i-4])) / (dz * dx)
            
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

