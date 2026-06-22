import cupy as cp

ABC_Cuda = r'''
extern "C" __global__
void AbsorbingBoundaryCuda(float* Uf , float* Uc, int N_abc, int nz, int nx, float* A) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = nz * nx;
    if (i >= total_size) return;

    // Get 2D coordinates from linear index 'i'
    int iz = i / nx;
    int ix = i % nx;

    if (ix < N_abc){
    Uf[i] =  Uf[i] * A[ix];
    Uc[i] =  Uc[i] * A[ix];
    }
    if (ix >=  nx - N_abc){
    Uf[i] =  Uf[i] * A[nx - 1 - ix];
    Uc[i] =  Uc[i] * A[nx - 1 - ix];
    }
    if (iz < N_abc){
    Uf[i] =  Uf[i] * A[iz];
    Uc[i] =  Uc[i] * A[iz];
    }
    if (iz >= nz - N_abc){
    Uf[i] =  Uf[i] * A[nz - 1 - iz];
    Uc[i] =  Uc[i] * A[nz - 1 - iz];
    }
}
'''
AbsorbingBoundaryCudaKernel = cp.RawKernel(ABC_Cuda, 'AbsorbingBoundaryCuda')

horizontal_dampening_profilesDevice = r'''
__device__ __forceinline__
void horizontal_dampening_profilesCuda(const int N_abc,const int nx_abc,const float dx,const float* vp,const float f_pico,const float d0,const float dt,const int i,const int j,float* ax,float* bx){
    float d = 0.0f;
    float alpha = 0.0f;
    float points_CPML = 0.0f;
    float posicao_relativa = 0.0f;

    if (i < N_abc){
        points_CPML = (N_abc - i - 1.0f) * dx;
        posicao_relativa = points_CPML / (N_abc * dx);
        d = d0 / (2.0f * N_abc * dx) * (posicao_relativa * posicao_relativa) * vp[j * nx_abc + i];
        alpha = 3.1415926535f * f_pico * (1.0f - (posicao_relativa * posicao_relativa));
    }
    else if (i >= nx_abc - N_abc){
        points_CPML = (i - nx_abc + N_abc) * dx;
        posicao_relativa = points_CPML / (N_abc * dx);
        d = d0 / (2.0f * N_abc * dx) * (posicao_relativa * posicao_relativa) * vp[j * nx_abc + i];
        alpha = 3.1415926535f * f_pico * (1.0f - (posicao_relativa * posicao_relativa));
    }

    *ax = expf(-(d + alpha) * dt);

    *bx = 0.0f;
    if (fabsf(d + alpha) > 1e-10f){
        *bx = (d / (d + alpha)) * ((*ax) - 1.0f);
    }
}
'''

vertical_dampening_profilesDevice = r'''
__device__ __forceinline__
void vertical_dampening_profilesCuda(const int N_abc,const int nx_abc,const int nz_abc,const float dz,const float* vp,const float f_pico,const float d0,const float dt,const int i,const int j,float* az,float* bz){
    float d = 0.0f;
    float alpha = 0.0f;
    float points_CPML = 0.0f;
    float posicao_relativa = 0.0f;

    if (j < N_abc){
        points_CPML = (N_abc - j - 1.0f) * dz;
        posicao_relativa = points_CPML / (N_abc * dz);
        d = d0 / (2.0f * N_abc * dz) * (posicao_relativa * posicao_relativa) * vp[j * nx_abc + i];
        alpha = 3.1415926535f * f_pico * (1.0f - (posicao_relativa * posicao_relativa));
    }
    else if (j >= nz_abc - N_abc){
        points_CPML = (j - nz_abc + N_abc) * dz;
        posicao_relativa = points_CPML / (N_abc * dz);
        d = d0 / (2.0f * N_abc * dz) * (posicao_relativa * posicao_relativa) * vp[j * nx_abc + i];
        alpha = 3.1415926535f * f_pico * (1.0f - (posicao_relativa * posicao_relativa));
    }

    *az = expf(-(d + alpha) * dt);

    *bz = 0.0f;
    if (fabsf(d + alpha) > 1e-10f){
        *bz = (d / (d + alpha)) * ((*az) - 1.0f);
    }
}
'''
updatePsiCuda = r'''
extern "C" __global__
void updatePsiCuda(float* PsixFR,float* PsixFL, float* PsizFU, float* PsizFD,const int nx_abc,const int nz_abc,const float* Uc,const float dx,const float dz,const int N_abc, const float f_pico, const float d0, const float dt, const float* vp)
{
    const float a1 =  4.0f / 5.0f;
    const float a2 = -1.0f / 5.0f;
    const float a3 =  4.0f / 105.0f;
    const float a4 = -1.0f / 280.0f;
    float ax;
    float bx;
    float az;
    float bz;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = nz_abc * nx_abc;

    if (i >= total_size) return;

    int iz = i/nx_abc;
    int ix = i%nx_abc;

    if (ix >= 4 && ix < N_abc && iz >= 4 && iz < nz_abc - 4){
        int idx = iz * N_abc + ix;
        horizontal_dampening_profilesCuda(N_abc,nx_abc,dx,vp,f_pico,d0,dt,ix,iz,&ax,&bx);

        float px = (a1*(Uc[i+1] - Uc[i-1]) +
                    a2*(Uc[i+2] - Uc[i-2]) +
                    a3*(Uc[i+3] - Uc[i-3]) +
                    a4*(Uc[i+4] - Uc[i-4])) / dx;

        PsixFL[idx] = ax * PsixFL[idx] + bx * px;
    }

    if (ix >= nx_abc - N_abc && ix < nx_abc - 4 && iz >= 4 && iz < nz_abc - 4){
        int idx = iz * N_abc + (ix - (nx_abc - N_abc));
        horizontal_dampening_profilesCuda(N_abc,nx_abc,dx,vp,f_pico,d0,dt,ix,iz,&ax,&bx);

        float px = (a1*(Uc[i+1] - Uc[i-1]) +
                    a2*(Uc[i+2] - Uc[i-2]) +
                    a3*(Uc[i+3] - Uc[i-3]) +
                    a4*(Uc[i+4] - Uc[i-4])) / dx;

        PsixFR[idx] = ax * PsixFR[idx] + bx * px;
    }

    if (ix >= 4 && ix < nx_abc - 4 && iz >= 4 && iz < N_abc){
        int jdx =  iz * nx_abc + ix;
        vertical_dampening_profilesCuda(N_abc,nx_abc,nz_abc,dz,vp,f_pico,d0,dt,ix,iz,&az,&bz);

        float pz = (a1 * (Uc[i + nx_abc] - Uc[i - nx_abc]) +
                    a2 * (Uc[i + 2*nx_abc] - Uc[i - 2*nx_abc]) +
                    a3 * (Uc[i + 3*nx_abc] - Uc[i - 3*nx_abc]) +
                    a4 * (Uc[i + 4*nx_abc] - Uc[i - 4*nx_abc])) / dz;

        PsizFU[jdx] = az * PsizFU[jdx] + bz * pz;
    }

    if (ix >= 4 && ix < nx_abc - 4 && iz >= nz_abc - N_abc && iz < nz_abc - 4){
        int jdx = (iz - (nz_abc - N_abc)) * nx_abc + ix;
        vertical_dampening_profilesCuda(N_abc,nx_abc,nz_abc,dz,vp,f_pico,d0,dt,ix,iz,&az,&bz);

        float pz = (a1 * (Uc[i + nx_abc] - Uc[i - nx_abc]) +
                    a2 * (Uc[i + 2*nx_abc] - Uc[i - 2*nx_abc]) +
                    a3 * (Uc[i + 3*nx_abc] - Uc[i - 3*nx_abc]) +
                    a4 * (Uc[i + 4*nx_abc] - Uc[i - 4*nx_abc])) / dz;

        PsizFD[jdx] = az * PsizFD[jdx] + bz * pz;
    }
}
'''
updatePsiCode = horizontal_dampening_profilesDevice + vertical_dampening_profilesDevice + updatePsiCuda
updatePsiKernel = cp.RawKernel(updatePsiCode, 'updatePsiCuda')

updateZetaCuda = r'''
extern "C" __global__
void updateZetaCuda(float* PsixFR,float* PsixFL,float*ZetaxFR,float* ZetaxFL,float* PsizFU, float* PsizFD,float* ZetazFU,float* ZetazFD,const int nx_abc,const int nz_abc,const float* Uc,const float dx,const float dz,const int N_abc, const float f_pico, const float d0, const float dt, const float* vp)
{
    const float c0 = -1435.0f / 504.0f;
    const float c1 =  8.0f / 5.0f;
    const float c2 = -1.0f / 5.0f;
    const float c3 =  8.0f / 315.0f;
    const float c4 = -1.0f / 560.0f;
    const float a1 =  4.0f / 5.0f;
    const float a2 = -1.0f / 5.0f;
    const float a3 =  4.0f / 105.0f;
    const float a4 = -1.0f / 280.0f;

    float ax;
    float bx;
    float az;
    float bz;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = nz_abc * nx_abc;

    if (i >= total_size) return;

    int iz = i/nx_abc;
    int ix = i%nx_abc;

    if (ix >= 4 && ix < N_abc && iz >= 4 && iz < nz_abc - 4){
        int idx = iz * N_abc + ix;
        horizontal_dampening_profilesCuda(N_abc,nx_abc,dx,vp,f_pico,d0,dt,ix,iz,&ax,&bx);

        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float psix = (a1 * (PsixFL[idx+1] - PsixFL[idx-1]) +
            a2 * (PsixFL[idx+2] - PsixFL[idx-2]) +
            a3 * (PsixFL[idx+3] - PsixFL[idx-3]) +
            a4 * (PsixFL[idx+4] - PsixFL[idx-4])) / dx;

        ZetaxFL[idx] = ax * ZetaxFL[idx] + bx * (pxx + psix);
    }

    if (ix >= nx_abc - N_abc && ix < nx_abc - 4 && iz >= 4 && iz < nz_abc - 4){
        int idx = iz * N_abc + (ix - (nx_abc - N_abc));
        horizontal_dampening_profilesCuda(N_abc,nx_abc,dx,vp,f_pico,d0,dt,ix,iz,&ax,&bx);

        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float psix = (a1 * (PsixFR[idx+1] - PsixFR[idx-1]) +
            a2 * (PsixFR[idx+2] - PsixFR[idx-2]) +
            a3 * (PsixFR[idx+3] - PsixFR[idx-3]) +
            a4 * (PsixFR[idx+4] - PsixFR[idx-4])) / dx;

        ZetaxFR[idx] = ax * ZetaxFR[idx] + bx * (pxx + psix);
    }

    if (ix >= 4 && ix < nx_abc - 4 && iz >= 4 && iz < N_abc){
        int jdx =  iz * nx_abc + ix;
        vertical_dampening_profilesCuda(N_abc,nx_abc,nz_abc,dz,vp,f_pico,d0,dt,ix,iz,&az,&bz);

        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);

        float psiz = (a1 * (PsizFU[jdx + nx_abc] - PsizFU[jdx - nx_abc]) +
            a2 * (PsizFU[jdx + 2*nx_abc] - PsizFU[jdx - 2*nx_abc]) +
            a3 * (PsizFU[jdx + 3*nx_abc] - PsizFU[jdx - 3*nx_abc]) +
            a4 * (PsizFU[jdx + 4*nx_abc] - PsizFU[jdx - 4*nx_abc])) / dz;

        ZetazFU[jdx] = az * ZetazFU[jdx] + bz * (pzz + psiz);
    }

    if (ix >= 4 && ix < nx_abc - 4 && iz >= nz_abc - N_abc && iz < nz_abc - 4){
        int jdx = (iz - (nz_abc - N_abc)) * nx_abc + ix;
        vertical_dampening_profilesCuda(N_abc,nx_abc,nz_abc,dz,vp,f_pico,d0,dt,ix,iz,&az,&bz);

        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);

        float psiz = (a1 * (PsizFD[jdx + nx_abc] - PsizFD[jdx - nx_abc]) +
            a2 * (PsizFD[jdx + 2*nx_abc] - PsizFD[jdx - 2*nx_abc]) +
            a3 * (PsizFD[jdx + 3*nx_abc] - PsizFD[jdx - 3*nx_abc]) +
            a4 * (PsizFD[jdx + 4*nx_abc] - PsizFD[jdx - 4*nx_abc])) / dz;

        ZetazFD[jdx] = az * ZetazFD[jdx] + bz * (pzz + psiz);
    }
}
'''
updateZetaCode = horizontal_dampening_profilesDevice + vertical_dampening_profilesDevice + updateZetaCuda
updateZetaKernel = cp.RawKernel(updateZetaCode, 'updateZetaCuda')

updateWaveEquationCuda = r'''
extern "C" __global__
void updateWaveEquationCuda(float* Uf,const float* Uc,const float* vp,const int nz,const int nx,const float dz,const float dx,const float dt)
{
    const float c0 = -1435.0f / 504.0f;
    const float c1 =  8.0f / 5.0f;
    const float c2 = -1.0f / 5.0f;
    const float c3 =  8.0f / 315.0f;
    const float c4 = -1.0f / 560.0f;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = nz * nx;

    if (i >= total_size) return;

    int iz = i/nx;
    int ix = i%nx;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4) 
    {
        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);

        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx] + Uc[i - nx])
            + c2 * (Uc[i + 2*nx] + Uc[i - 2*nx])
            + c3 * (Uc[i + 3*nx] + Uc[i - 3*nx])
            + c4 * (Uc[i + 4*nx] + Uc[i - 4*nx])) / (dz * dz);

        Uf[i] = (vp[i] * vp[i]) * (dt * dt) * (pxx + pzz) + 2.0f * Uc[i] - Uf[i];
    }
}
'''

updateWaveEquationKernel = cp.RawKernel(updateWaveEquationCuda, 'updateWaveEquationCuda')

updateWaveEquationVTICuda = r'''
extern "C" __global__
void updateWaveEquationVTICuda(float* Uf,const float* Uc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* vp,const float* epsilon,const float* delta )
{
    const float c0 = -1435.0f / 504.0f;
    const float c1 =  8.0f / 5.0f;
    const float c2 = -1.0f / 5.0f;
    const float c3 =  8.0f / 315.0f;
    const float c4 = -1.0f / 560.0f;
    const float a1 =  4.0f / 5.0f;
    const float a2 = -1.0f / 5.0f;
    const float a3 =  4.0f / 105.0f;
    const float a4 = -1.0f / 280.0f;
    float Sd;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = nz * nx;

    if (i >= total_size) return;

    int iz = i/nx;
    int ix = i%nx;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4) 
    {
        float pxx = (c0 * Uc[i]
                    + c1 * (Uc[i + 1] + Uc[i - 1])
                    + c2 * (Uc[i + 2] + Uc[i - 2])
                    + c3 * (Uc[i + 3] + Uc[i - 3])
                    + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);

        float pzz = (c0 * Uc[i]
                    + c1 * (Uc[i + nx] + Uc[i - nx])
                    + c2 * (Uc[i + 2*nx] + Uc[i - 2*nx])
                    + c3 * (Uc[i + 3*nx] + Uc[i - 3*nx])
                    + c4 * (Uc[i + 4*nx] + Uc[i - 4*nx])) / (dz * dz);

        float px = (a1*(Uc[i+1] - Uc[i-1]) +
                    a2*(Uc[i+2] - Uc[i-2]) +
                    a3*(Uc[i+3] - Uc[i-3]) +
                    a4*(Uc[i+4] - Uc[i-4])) / dx;

        float pz = (a1 * (Uc[i + nx] - Uc[i - nx]) +
                    a2 * (Uc[i + 2*nx] - Uc[i - 2*nx]) +
                    a3 * (Uc[i + 3*nx] - Uc[i - 3*nx]) +
                    a4 * (Uc[i + 4*nx] - Uc[i - 4*nx])) / dz;
        
        float num = -2.0f*(epsilon[i]-delta[i])*(px*px)*(pz*pz);
        float den = (1.0f + 2.0f*epsilon[i])*(px*px*px*px) + (pz*pz*pz*pz) + 2.0f*(1.0f + delta[i])*(px*px)*(pz*pz);

        if (fabsf(den)<1e-12f){
            Sd = 0.0f;
        }
        else{
            Sd = num / den;
        }
        Uf[i] = 2.0f * Uc[i] - Uf[i] + (vp[i] * vp[i]) * (dt * dt) * ((1.0f+ 2.0f*epsilon[i]) + Sd) * pxx + (vp[i] * vp[i]) * (dt * dt) *(1.0f + Sd) * pzz;
    }
}
'''

updateWaveEquationVTIKernel = cp.RawKernel(updateWaveEquationVTICuda, 'updateWaveEquationVTICuda')

updateWaveEquationTTICuda = r'''
extern "C" __global__
void updateWaveEquationTTICuda(float* Uf,const float* Uc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* vp,const float* epsilon,const float* delta,const float* theta)
{
    const float c0 = -1435.0f / 504.0f;
    const float c1 =  8.0f / 5.0f;
    const float c2 = -1.0f / 5.0f;
    const float c3 =  8.0f / 315.0f;
    const float c4 = -1.0f / 560.0f;

    const float a1 =  4.0f / 5.0f;
    const float a2 = -1.0f / 5.0f;
    const float a3 =  4.0f / 105.0f;
    const float a4 = -1.0f / 280.0f;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = nz * nx;

    if (i >= total_size) return;

    int iz = i / nx;
    int ix = i % nx;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4)
    {
        float pxx = (c0 * Uc[i]
                   + c1 * (Uc[i + 1] + Uc[i - 1])
                   + c2 * (Uc[i + 2] + Uc[i - 2])
                   + c3 * (Uc[i + 3] + Uc[i - 3])
                   + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);

        float pzz = (c0 * Uc[i]
                   + c1 * (Uc[i + nx]     + Uc[i - nx])
                   + c2 * (Uc[i + 2*nx] + Uc[i - 2*nx])
                   + c3 * (Uc[i + 3*nx] + Uc[i - 3*nx])
                   + c4 * (Uc[i + 4*nx] + Uc[i - 4*nx])) / (dz * dz);

        float pxz = (
            a1*a1*(Uc[i + nx + 1]     - Uc[i - nx + 1]     + Uc[i - nx - 1]     - Uc[i + nx - 1]) +
            a1*a2*(Uc[i + 2*nx + 1]   - Uc[i - 2*nx + 1]   + Uc[i - 2*nx - 1]   - Uc[i + 2*nx - 1]) +
            a1*a3*(Uc[i + 3*nx + 1]   - Uc[i - 3*nx + 1]   + Uc[i - 3*nx - 1]   - Uc[i + 3*nx - 1]) +
            a1*a4*(Uc[i + 4*nx + 1]   - Uc[i - 4*nx + 1]   + Uc[i - 4*nx - 1]   - Uc[i + 4*nx - 1]) +

            a2*a1*(Uc[i + nx + 2]     - Uc[i - nx + 2]     + Uc[i - nx - 2]     - Uc[i + nx - 2]) +
            a2*a2*(Uc[i + 2*nx + 2]   - Uc[i - 2*nx + 2]   + Uc[i - 2*nx - 2]   - Uc[i + 2*nx - 2]) +
            a2*a3*(Uc[i + 3*nx + 2]   - Uc[i - 3*nx + 2]   + Uc[i - 3*nx - 2]   - Uc[i + 3*nx - 2]) +
            a2*a4*(Uc[i + 4*nx + 2]   - Uc[i - 4*nx + 2]   + Uc[i - 4*nx - 2]   - Uc[i + 4*nx - 2]) +

            a3*a1*(Uc[i + nx + 3]     - Uc[i - nx + 3]     + Uc[i - nx - 3]     - Uc[i + nx - 3]) +
            a3*a2*(Uc[i + 2*nx + 3]   - Uc[i - 2*nx + 3]   + Uc[i - 2*nx - 3]   - Uc[i + 2*nx - 3]) +
            a3*a3*(Uc[i + 3*nx + 3]   - Uc[i - 3*nx + 3]   + Uc[i - 3*nx - 3]   - Uc[i + 3*nx - 3]) +
            a3*a4*(Uc[i + 4*nx + 3]   - Uc[i - 4*nx + 3]   + Uc[i - 4*nx - 3]   - Uc[i + 4*nx - 3]) +

            a4*a1*(Uc[i + nx + 4]     - Uc[i - nx + 4]     + Uc[i - nx - 4]     - Uc[i + nx - 4]) +
            a4*a2*(Uc[i + 2*nx + 4]   - Uc[i - 2*nx + 4]   + Uc[i - 2*nx - 4]   - Uc[i + 2*nx - 4]) +
            a4*a3*(Uc[i + 3*nx + 4]   - Uc[i - 3*nx + 4]   + Uc[i - 3*nx - 4]   - Uc[i + 3*nx - 4]) +
            a4*a4*(Uc[i + 4*nx + 4]   - Uc[i - 4*nx + 4]   + Uc[i - 4*nx - 4]   - Uc[i + 4*nx - 4])
        ) / (dz * dx);

        float px = (a1 * (Uc[i + 1] - Uc[i - 1]) +
                    a2 * (Uc[i + 2] - Uc[i - 2]) +
                    a3 * (Uc[i + 3] - Uc[i - 3]) +
                    a4 * (Uc[i + 4] - Uc[i - 4])) / dx;

        float pz = (a1 * (Uc[i + nx]     - Uc[i - nx]) +
                    a2 * (Uc[i + 2 * nx] - Uc[i - 2 * nx]) +
                    a3 * (Uc[i + 3 * nx] - Uc[i - 3 * nx]) +
                    a4 * (Uc[i + 4 * nx] - Uc[i - 4 * nx])) / dz;

        float norm = sqrtf(px * px + pz * pz);
        float mx, mz;
        if (norm > 1e-12f) {
            mx = px / norm;
            mz = pz / norm;
        } else {
            mx = 0.0f;
            mz = 0.0f;
        }

        float th = theta[i];

        float num = -2.0f * (epsilon[i] - delta[i]) * ((mx * cosf(th) - mz * sinf(th)) * (mx * cosf(th) - mz * sinf(th))) * ((mx * sinf(th) + mz * cosf(th)) * (mx * sinf(th) + mz * cosf(th)));
        float den = (1.0f + 2.0f * epsilon[i]) * ((mx * cosf(th) - mz * sinf(th)) * (mx * cosf(th) - mz * sinf(th)) * (mx * cosf(th) - mz * sinf(th)) * (mx * cosf(th) - mz * sinf(th))) + ((mx * sinf(th) + mz * cosf(th)) * (mx * sinf(th) + mz * cosf(th)) *(mx * sinf(th) + mz * cosf(th)) * (mx * sinf(th) + mz * cosf(th))) + 2.0f * (1.0f + delta[i]) * ((mx * cosf(th) - mz * sinf(th)) * (mx * cosf(th) - mz * sinf(th))) * ((mx * sinf(th) + mz * cosf(th)) * (mx * sinf(th) + mz * cosf(th)));

        float Sd;
        if (fabsf(den) < 1e-12f) {
            Sd = 0.0f;
        } else {
            Sd = num / den;
        }

        Uf[i] = 2.0f * Uc[i] - Uf[i]+ (vp[i] * vp[i]) * (dt * dt) * (((1.0f + 2.0f * epsilon[i]) * (cosf(th) * cosf(th)) + (sinf(th) * sinf(th)) + Sd) * pxx) + (vp[i] * vp[i]) * (dt * dt) * (((1.0f + 2.0f * epsilon[i]) * (sinf(th) * sinf(th)) + (cosf(th) * cosf(th)) + Sd) * pzz) - 2.0f * epsilon[i] * (vp[i] * vp[i]) * (dt * dt) * sinf(2.0f * th) * pxz;
    }
}
'''

updateWaveEquationTTIKernel = cp.RawKernel(updateWaveEquationTTICuda, 'updateWaveEquationTTICuda')

updateWaveEquationCPMLCuda = r'''
extern "C" __global__
void updateWaveEquationCPMLCuda(float* Uf,const float* Uc,const float* vp,const int nx_abc,const int nz_abc,const float dz,const float dx,const float dt,float* PsixFR,float* PsixFL,float* PsizFU,float* PsizFD,float* ZetaxFR,float* ZetaxFL,float* ZetazFU,float* ZetazFD,const int N_abc)
{
    const float c0 = -1435.0f / 504.0f;
    const float c1 =  8.0f / 5.0f;
    const float c2 = -1.0f / 5.0f;
    const float c3 =  8.0f / 315.0f;
    const float c4 = -1.0f / 560.0f;

    const float a1 =  4.0f / 5.0f;
    const float a2 = -1.0f / 5.0f;
    const float a3 =  4.0f / 105.0f;
    const float a4 = -1.0f / 280.0f;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = nz_abc * nx_abc;

    if (i >= total_size) return;

    int iz = i/nx_abc;
    int ix = i%nx_abc;

    // Regiao Interior
    if (iz >= N_abc && iz < nz_abc - N_abc && ix >= N_abc && ix < nx_abc - N_abc)
    {
        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);

        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);

        Uf[i] = (vp[i] * vp[i]) * (dt * dt) * (pxx + pzz) + 2.0f * Uc[i] - Uf[i];
    }

    // Regiao Esquerda
    if (iz >= N_abc && iz < nz_abc - N_abc && ix >= 4 && ix < N_abc)
    {
        int idx = iz * N_abc + ix;

        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);

        float psix = (a1 * (PsixFL[idx + 1] - PsixFL[idx - 1]) +
                      a2 * (PsixFL[idx + 2] - PsixFL[idx - 2]) +
                      a3 * (PsixFL[idx + 3] - PsixFL[idx - 3]) +
                      a4 * (PsixFL[idx + 4] - PsixFL[idx - 4])) / dx;

        Uf[i] = (vp[i] * vp[i]) * (dt * dt) * (pxx + pzz + psix + ZetaxFL[idx]) + 2.0f * Uc[i] - Uf[i];
    }

    // Regiao Direita
    if (iz >= N_abc && iz < nz_abc - N_abc && ix >= nx_abc - N_abc && ix < nx_abc - 4)
    {
        int idx = iz * N_abc + (ix - (nx_abc - N_abc));

        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);

        float psix = (a1 * (PsixFR[idx + 1] - PsixFR[idx - 1]) +
                      a2 * (PsixFR[idx + 2] - PsixFR[idx - 2]) +
                      a3 * (PsixFR[idx + 3] - PsixFR[idx - 3]) +
                      a4 * (PsixFR[idx + 4] - PsixFR[idx - 4])) / dx;

        Uf[i] = (vp[i] * vp[i]) * (dt * dt) * (pxx + pzz + psix + ZetaxFR[idx]) + 2.0f * Uc[i] - Uf[i];
    }

    // Regiao Superior
    if (iz >= 4 && iz < N_abc && ix >= N_abc && ix < nx_abc - N_abc)
    {
        int jdx = iz * nx_abc + ix;
    
        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);

        float psiz = (a1 * (PsizFU[jdx + nx_abc]     - PsizFU[jdx - nx_abc]) +
                      a2 * (PsizFU[jdx + 2 * nx_abc] - PsizFU[jdx - 2 * nx_abc]) +
                      a3 * (PsizFU[jdx + 3 * nx_abc] - PsizFU[jdx - 3 * nx_abc]) +
                      a4 * (PsizFU[jdx + 4 * nx_abc] - PsizFU[jdx - 4 * nx_abc])) / dz;

        Uf[i] = (vp[i] * vp[i]) * (dt * dt) * (pxx + pzz + psiz + ZetazFU[jdx]) + 2.0f * Uc[i] - Uf[i];
    }

    // Regiao Inferior
    if (iz >= nz_abc - N_abc && iz < nz_abc - 4 && ix >= N_abc && ix < nx_abc - N_abc)
    {
        int jdx = (iz - (nz_abc - N_abc)) * nx_abc + ix;
    
        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);

        float psiz = (a1 * (PsizFD[jdx + nx_abc]     - PsizFD[jdx - nx_abc]) +
                      a2 * (PsizFD[jdx + 2 * nx_abc] - PsizFD[jdx - 2 * nx_abc]) +
                      a3 * (PsizFD[jdx + 3 * nx_abc] - PsizFD[jdx - 3 * nx_abc]) +
                      a4 * (PsizFD[jdx + 4 * nx_abc] - PsizFD[jdx - 4 * nx_abc])) / dz;

        Uf[i] = (vp[i] * vp[i]) * (dt * dt) * (pxx + pzz + psiz + ZetazFD[jdx]) + 2.0f * Uc[i] - Uf[i];
    }

    // Quina Superior Esquerda
    if (ix >= 4 && ix < N_abc && iz >= 4 && iz < N_abc)
    {
        int idx = iz * N_abc + ix;
        int jdx = iz * nx_abc + ix;

        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);

        float psix = (a1 * (PsixFL[idx + 1] - PsixFL[idx - 1]) +
                      a2 * (PsixFL[idx + 2] - PsixFL[idx - 2]) +
                      a3 * (PsixFL[idx + 3] - PsixFL[idx - 3]) +
                      a4 * (PsixFL[idx + 4] - PsixFL[idx - 4])) / dx;

        float psiz = (a1 * (PsizFU[jdx + nx_abc]     - PsizFU[jdx - nx_abc]) +
                      a2 * (PsizFU[jdx + 2 * nx_abc] - PsizFU[jdx - 2 * nx_abc]) +
                      a3 * (PsizFU[jdx + 3 * nx_abc] - PsizFU[jdx - 3 * nx_abc]) +
                      a4 * (PsizFU[jdx + 4 * nx_abc] - PsizFU[jdx - 4 * nx_abc])) / dz;

        Uf[i] = (vp[i] * vp[i]) * (dt * dt) * (pxx + pzz + psix + psiz + ZetaxFL[idx] + ZetazFU[jdx]) + 2.0f * Uc[i] - Uf[i];
    }

    // Quina Superior Direita
    if (ix >= nx_abc - N_abc && ix < nx_abc - 4 && iz >= 4 && iz < N_abc)
    {
        int idx = iz * N_abc + (ix - (nx_abc - N_abc));
        int jdx = iz * nx_abc + ix;
    
        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);

        float psix = (a1 * (PsixFR[idx + 1] - PsixFR[idx - 1]) +
                      a2 * (PsixFR[idx + 2] - PsixFR[idx - 2]) +
                      a3 * (PsixFR[idx + 3] - PsixFR[idx - 3]) +
                      a4 * (PsixFR[idx + 4] - PsixFR[idx - 4])) / dx;

        float psiz = (a1 * (PsizFU[jdx + nx_abc]     - PsizFU[jdx - nx_abc]) +
                      a2 * (PsizFU[jdx + 2 * nx_abc] - PsizFU[jdx - 2 * nx_abc]) +
                      a3 * (PsizFU[jdx + 3 * nx_abc] - PsizFU[jdx - 3 * nx_abc]) +
                      a4 * (PsizFU[jdx + 4 * nx_abc] - PsizFU[jdx - 4 * nx_abc])) / dz;

        Uf[i] = (vp[i] * vp[i]) * (dt * dt) * (pxx + pzz + psix + psiz + ZetaxFR[idx] + ZetazFU[jdx]) + 2.0f * Uc[i] - Uf[i];
    }

    // Quina Inferior Esquerda
    if (ix >= 4 && ix < N_abc && iz >= nz_abc - N_abc && iz < nz_abc - 4)
    {
        int idx = iz * N_abc + ix;
        int jdx = (iz - (nz_abc - N_abc)) * nx_abc + ix;

        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);

        float psix = (a1 * (PsixFL[idx + 1] - PsixFL[idx - 1]) +
                      a2 * (PsixFL[idx + 2] - PsixFL[idx - 2]) +
                      a3 * (PsixFL[idx + 3] - PsixFL[idx - 3]) +
                      a4 * (PsixFL[idx + 4] - PsixFL[idx - 4])) / dx;

        float psiz = (a1 * (PsizFD[jdx + nx_abc]     - PsizFD[jdx - nx_abc]) +
                      a2 * (PsizFD[jdx + 2 * nx_abc] - PsizFD[jdx - 2 * nx_abc]) +
                      a3 * (PsizFD[jdx + 3 * nx_abc] - PsizFD[jdx - 3 * nx_abc]) +
                      a4 * (PsizFD[jdx + 4 * nx_abc] - PsizFD[jdx - 4 * nx_abc])) / dz;

        Uf[i] = (vp[i] * vp[i]) * (dt * dt) * (pxx + pzz + psix + psiz + ZetaxFL[idx] + ZetazFD[jdx]) + 2.0f * Uc[i] - Uf[i];
    }

    // Quina Inferior Direita
    if (ix >= nx_abc - N_abc && ix < nx_abc - 4 && iz >= nz_abc - N_abc && iz < nz_abc - 4)
    {
        int idx = iz * N_abc + (ix - (nx_abc - N_abc));
        int jdx = (iz - (nz_abc - N_abc)) * nx_abc + ix;

        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);

        float psix = (a1 * (PsixFR[idx + 1] - PsixFR[idx - 1]) +
                      a2 * (PsixFR[idx + 2] - PsixFR[idx - 2]) +
                      a3 * (PsixFR[idx + 3] - PsixFR[idx - 3]) +
                      a4 * (PsixFR[idx + 4] - PsixFR[idx - 4])) / dx;

        float psiz = (a1 * (PsizFD[jdx + nx_abc]     - PsizFD[jdx - nx_abc]) +
                      a2 * (PsizFD[jdx + 2 * nx_abc] - PsizFD[jdx - 2 * nx_abc]) +
                      a3 * (PsizFD[jdx + 3 * nx_abc] - PsizFD[jdx - 3 * nx_abc]) +
                      a4 * (PsizFD[jdx + 4 * nx_abc] - PsizFD[jdx - 4 * nx_abc])) / dz;

        Uf[i] = (vp[i] * vp[i]) * (dt * dt) * (pxx + pzz + psix + psiz + ZetaxFR[idx] + ZetazFD[jdx]) + 2.0f * Uc[i] - Uf[i];
    }
}
'''

updateWaveEquationCPMLKernel = cp.RawKernel(updateWaveEquationCPMLCuda, 'updateWaveEquationCPMLCuda')


updateWaveEquationVTICPMLCuda = r'''
extern "C" __global__
void updateWaveEquationVTICPMLCuda(float* Uf,const float* Uc,const float* vp,const float* epsilon,const float* delta,const int nx_abc,const int nz_abc,const float dz,const float dx,const float dt,float* PsixFR,float* PsixFL,float* PsizFU,float* PsizFD,float* ZetaxFR,float* ZetaxFL,float* ZetazFU,float* ZetazFD,const int N_abc)
{
    const float c0 = -1435.0f / 504.0f;
    const float c1 =  8.0f / 5.0f;
    const float c2 = -1.0f / 5.0f;
    const float c3 =  8.0f / 315.0f;
    const float c4 = -1.0f / 560.0f;
    const float a1 =  4.0f / 5.0f;
    const float a2 = -1.0f / 5.0f;
    const float a3 =  4.0f / 105.0f;
    const float a4 = -1.0f / 280.0f;
    float Sd;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = nz_abc * nx_abc;

    if (i >= total_size) return;

    int iz = i/nx_abc;
    int ix = i%nx_abc;

    // Regiao Interior
    if (iz >= N_abc && iz < nz_abc - N_abc && ix >= N_abc && ix < nx_abc - N_abc)
    {
        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);

        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);

        float px = (a1*(Uc[i+1] - Uc[i-1]) +
                    a2*(Uc[i+2] - Uc[i-2]) +
                    a3*(Uc[i+3] - Uc[i-3]) +
                    a4*(Uc[i+4] - Uc[i-4])) / dx;

        float pz = (a1 * (Uc[i + nx_abc] - Uc[i - nx_abc]) +
                    a2 * (Uc[i + 2*nx_abc] - Uc[i - 2*nx_abc]) +
                    a3 * (Uc[i + 3*nx_abc] - Uc[i - 3*nx_abc]) +
                    a4 * (Uc[i + 4*nx_abc] - Uc[i - 4*nx_abc])) / dz;
        
        float num = -2.0f*(epsilon[i]-delta[i])*(px*px)*(pz*pz);
        float den = (1.0f + 2.0f*epsilon[i])*(px*px*px*px) + (pz*pz*pz*pz) + 2.0f*(1.0f + delta[i])*(px*px)*(pz*pz);

        if (fabsf(den)<1e-12f){
            Sd = 0.0f;
        }
        else{
            Sd = num / den;
        }

        Uf[i] = 2.0f * Uc[i] - Uf[i] + (vp[i] * vp[i]) * (dt * dt) * ((1.0f+ 2.0f*epsilon[i]) + Sd) * pxx + (vp[i] * vp[i]) * (dt * dt) *(1.0f + Sd) * pzz;
    }

    // Regiao Esquerda
    if (iz >= N_abc && iz < nz_abc - N_abc && ix >= 4 && ix < N_abc)
    {
        int idx = iz * N_abc + ix;

        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);
        
        float px = (a1*(Uc[i+1] - Uc[i-1]) +
                    a2*(Uc[i+2] - Uc[i-2]) +
                    a3*(Uc[i+3] - Uc[i-3]) +
                    a4*(Uc[i+4] - Uc[i-4])) / dx;

        float pz = (a1 * (Uc[i + nx_abc] - Uc[i - nx_abc]) +
                    a2 * (Uc[i + 2*nx_abc] - Uc[i - 2*nx_abc]) +
                    a3 * (Uc[i + 3*nx_abc] - Uc[i - 3*nx_abc]) +
                    a4 * (Uc[i + 4*nx_abc] - Uc[i - 4*nx_abc])) / dz;

        float psix = (a1 * (PsixFL[idx + 1] - PsixFL[idx - 1]) +
                      a2 * (PsixFL[idx + 2] - PsixFL[idx - 2]) +
                      a3 * (PsixFL[idx + 3] - PsixFL[idx - 3]) +
                      a4 * (PsixFL[idx + 4] - PsixFL[idx - 4])) / dx;

        float num = -2.0f*(epsilon[i]-delta[i])*((px + psix)*(px + psix))*(pz*pz);
        float den = (1.0f + 2.0f*epsilon[i])*((px + psix)*(px + psix)*(px + psix)*(px + psix)) + (pz*pz*pz*pz) + 2.0f*(1.0f + delta[i])*((px + psix)*(px + psix))*(pz*pz);

        if (fabsf(den)<1e-12f){
            Sd = 0.0f;
        }
        else{
            Sd = num / den;
        }

        Uf[i] = 2.0f * Uc[i] - Uf[i] + (vp[i] * vp[i]) * (dt * dt) * ((1.0f+ 2.0f*epsilon[i]) + Sd) * (pxx + psix + ZetaxFL[idx]) + (vp[i] * vp[i]) * (dt * dt) *(1.0f + Sd) * pzz;
    }

    // Regiao Direita
    if (iz >= N_abc && iz < nz_abc - N_abc && ix >= nx_abc - N_abc && ix < nx_abc - 4)
    {
        int idx = iz * N_abc + (ix - (nx_abc - N_abc));

        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);
        
        float px = (a1*(Uc[i+1] - Uc[i-1]) +
                    a2*(Uc[i+2] - Uc[i-2]) +
                    a3*(Uc[i+3] - Uc[i-3]) +
                    a4*(Uc[i+4] - Uc[i-4])) / dx;

        float pz = (a1 * (Uc[i + nx_abc] - Uc[i - nx_abc]) +
                    a2 * (Uc[i + 2*nx_abc] - Uc[i - 2*nx_abc]) +
                    a3 * (Uc[i + 3*nx_abc] - Uc[i - 3*nx_abc]) +
                    a4 * (Uc[i + 4*nx_abc] - Uc[i - 4*nx_abc])) / dz;

        float psix = (a1 * (PsixFR[idx + 1] - PsixFR[idx - 1]) +
                      a2 * (PsixFR[idx + 2] - PsixFR[idx - 2]) +
                      a3 * (PsixFR[idx + 3] - PsixFR[idx - 3]) +
                      a4 * (PsixFR[idx + 4] - PsixFR[idx - 4])) / dx;

        float num = -2.0f*(epsilon[i]-delta[i])*((px + psix)*(px + psix))*(pz*pz);
        float den = (1.0f + 2.0f*epsilon[i])*((px + psix)*(px + psix)*(px + psix)*(px + psix)) + (pz*pz*pz*pz) + 2.0f*(1.0f + delta[i])*((px + psix)*(px + psix))*(pz*pz);

        if (fabsf(den)<1e-12f){
            Sd = 0.0f;
        }
        else{
            Sd = num / den;
        }

        Uf[i] = 2.0f * Uc[i] - Uf[i] + (vp[i] * vp[i]) * (dt * dt) * ((1.0f+ 2.0f*epsilon[i]) + Sd) * (pxx + psix + ZetaxFR[idx]) + (vp[i] * vp[i]) * (dt * dt) *(1.0f + Sd) * pzz;
    }

    // Regiao Superior
    if (iz >= 4 && iz < N_abc && ix >= N_abc && ix < nx_abc - N_abc)
    {
        int jdx = iz * nx_abc + ix;

        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);
        
        float px = (a1*(Uc[i+1] - Uc[i-1]) +
                    a2*(Uc[i+2] - Uc[i-2]) +
                    a3*(Uc[i+3] - Uc[i-3]) +
                    a4*(Uc[i+4] - Uc[i-4])) / dx;

        float pz = (a1 * (Uc[i + nx_abc] - Uc[i - nx_abc]) +
                    a2 * (Uc[i + 2*nx_abc] - Uc[i - 2*nx_abc]) +
                    a3 * (Uc[i + 3*nx_abc] - Uc[i - 3*nx_abc]) +
                    a4 * (Uc[i + 4*nx_abc] - Uc[i - 4*nx_abc])) / dz;

        float psiz = (a1 * (PsizFU[jdx + nx_abc]   - PsizFU[jdx - nx_abc]) +
                    a2 * (PsizFU[jdx + 2 * nx_abc] - PsizFU[jdx - 2 * nx_abc]) +
                    a3 * (PsizFU[jdx + 3 * nx_abc] - PsizFU[jdx - 3 * nx_abc]) +
                    a4 * (PsizFU[jdx + 4 * nx_abc] - PsizFU[jdx - 4 * nx_abc])) / dz;

        float num = -2.0f*(epsilon[i]-delta[i])*(px*px)*((pz + psiz)*(pz + psiz));
        float den = (1.0f + 2.0f*epsilon[i])*(px*px*px*px) + ((pz + psiz)*(pz + psiz)*(pz + psiz)*(pz + psiz)) + 2.0f*(1.0f + delta[i])*(px*px)*((pz + psiz)*(pz + psiz));

        if (fabsf(den)<1e-12f){
            Sd = 0.0f;
        }
        else{
            Sd = num / den;
        }

        Uf[i] = 2.0f * Uc[i] - Uf[i] + (vp[i] * vp[i]) * (dt * dt) * ((1.0f+ 2.0f*epsilon[i]) + Sd) * pxx + (vp[i] * vp[i]) * (dt * dt) *(1.0f + Sd) * (pzz + psiz + ZetazFU[jdx]);
    }

    // Regiao Inferior
    if (iz >= nz_abc - N_abc && iz < nz_abc - 4 && ix >= N_abc && ix < nx_abc - N_abc)
    {
        int jdx = (iz - (nz_abc - N_abc)) * nx_abc + ix;

        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);
        
        float px = (a1*(Uc[i+1] - Uc[i-1]) +
                    a2*(Uc[i+2] - Uc[i-2]) +
                    a3*(Uc[i+3] - Uc[i-3]) +
                    a4*(Uc[i+4] - Uc[i-4])) / dx;

        float pz = (a1 * (Uc[i + nx_abc] - Uc[i - nx_abc]) +
                    a2 * (Uc[i + 2*nx_abc] - Uc[i - 2*nx_abc]) +
                    a3 * (Uc[i + 3*nx_abc] - Uc[i - 3*nx_abc]) +
                    a4 * (Uc[i + 4*nx_abc] - Uc[i - 4*nx_abc])) / dz;

        float psiz = (a1 * (PsizFD[jdx + nx_abc]     - PsizFD[jdx - nx_abc]) +
                    a2 * (PsizFD[jdx + 2 * nx_abc] - PsizFD[jdx - 2 * nx_abc]) +
                    a3 * (PsizFD[jdx + 3 * nx_abc] - PsizFD[jdx - 3 * nx_abc]) +
                    a4 * (PsizFD[jdx + 4 * nx_abc] - PsizFD[jdx - 4 * nx_abc])) / dz;

        float num = -2.0f*(epsilon[i]-delta[i])*(px*px)*((pz + psiz)*(pz + psiz));
        float den = (1.0f + 2.0f*epsilon[i])*(px*px*px*px) + ((pz + psiz)*(pz + psiz)*(pz + psiz)*(pz + psiz)) + 2.0f*(1.0f + delta[i])*(px*px)*((pz + psiz)*(pz + psiz));

        if (fabsf(den)<1e-12f){
            Sd = 0.0f;
        }
        else{
            Sd = num / den;
        }

        Uf[i] = 2.0f * Uc[i] - Uf[i] + (vp[i] * vp[i]) * (dt * dt) * ((1.0f+ 2.0f*epsilon[i]) + Sd) * pxx + (vp[i] * vp[i]) * (dt * dt) *(1.0f + Sd) * (pzz + psiz + ZetazFD[jdx]);
    }


    // Quina Superior Esquerda
    if (ix >= 4 && ix < N_abc && iz >= 4 && iz < N_abc)
    {
        int idx = iz * N_abc + ix;
        int jdx = iz * nx_abc + ix;
    
        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);
        
        float px = (a1*(Uc[i+1] - Uc[i-1]) +
                    a2*(Uc[i+2] - Uc[i-2]) +
                    a3*(Uc[i+3] - Uc[i-3]) +
                    a4*(Uc[i+4] - Uc[i-4])) / dx;

        float pz = (a1 * (Uc[i + nx_abc] - Uc[i - nx_abc]) +
                    a2 * (Uc[i + 2*nx_abc] - Uc[i - 2*nx_abc]) +
                    a3 * (Uc[i + 3*nx_abc] - Uc[i - 3*nx_abc]) +
                    a4 * (Uc[i + 4*nx_abc] - Uc[i - 4*nx_abc])) / dz;

        float psix = (a1 * (PsixFL[idx + 1] - PsixFL[idx - 1]) +
                      a2 * (PsixFL[idx + 2] - PsixFL[idx - 2]) +
                      a3 * (PsixFL[idx + 3] - PsixFL[idx - 3]) +
                      a4 * (PsixFL[idx + 4] - PsixFL[idx - 4])) / dx;

        float psiz = (a1 * (PsizFU[jdx + nx_abc]     - PsizFU[jdx - nx_abc]) +
                      a2 * (PsizFU[jdx + 2 * nx_abc] - PsizFU[jdx - 2 * nx_abc]) +
                      a3 * (PsizFU[jdx + 3 * nx_abc] - PsizFU[jdx - 3 * nx_abc]) +
                      a4 * (PsizFU[jdx + 4 * nx_abc] - PsizFU[jdx - 4 * nx_abc])) / dz;

        float num = -2.0f*(epsilon[i]-delta[i])*((px + psix)*(px + psix))*((pz + psiz)*(pz + psiz));
        float den = (1.0f + 2.0f*epsilon[i])*((px + psix)*(px + psix)*(px + psix)*(px + psix)) + ((pz + psiz)*(pz + psiz)*(pz + psiz)*(pz + psiz)) + 2.0f*(1.0f + delta[i])*((px + psix)*(px + psix))*((pz + psiz)*(pz + psiz));

        if (fabsf(den)<1e-12f){
            Sd = 0.0f;
        }
        else{
            Sd = num / den;
        }

        Uf[i] = 2.0f * Uc[i] - Uf[i] + (vp[i] * vp[i]) * (dt * dt) * ((1.0f+ 2.0f*epsilon[i]) + Sd) * (pxx + psix + ZetaxFL[idx]) + (vp[i] * vp[i]) * (dt * dt) *(1.0f + Sd) * (pzz + psiz + ZetazFU[jdx]);
    }

    // Quina Superior Direita
    if (ix >= nx_abc - N_abc && ix < nx_abc - 4 && iz >= 4 && iz < N_abc)
    {
        int idx = iz * N_abc + (ix - (nx_abc - N_abc));
        int jdx = iz * nx_abc + ix;

        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);
        
        float px = (a1*(Uc[i+1] - Uc[i-1]) +
                    a2*(Uc[i+2] - Uc[i-2]) +
                    a3*(Uc[i+3] - Uc[i-3]) +
                    a4*(Uc[i+4] - Uc[i-4])) / dx;

        float pz = (a1 * (Uc[i + nx_abc] - Uc[i - nx_abc]) +
                    a2 * (Uc[i + 2*nx_abc] - Uc[i - 2*nx_abc]) +
                    a3 * (Uc[i + 3*nx_abc] - Uc[i - 3*nx_abc]) +
                    a4 * (Uc[i + 4*nx_abc] - Uc[i - 4*nx_abc])) / dz;

        float psix = (a1 * (PsixFR[idx + 1] - PsixFR[idx - 1]) +
                      a2 * (PsixFR[idx + 2] - PsixFR[idx - 2]) +
                      a3 * (PsixFR[idx + 3] - PsixFR[idx - 3]) +
                      a4 * (PsixFR[idx + 4] - PsixFR[idx - 4])) / dx;

        float psiz = (a1 * (PsizFU[jdx + nx_abc]     - PsizFU[jdx - nx_abc]) +
                      a2 * (PsizFU[jdx + 2 * nx_abc] - PsizFU[jdx - 2 * nx_abc]) +
                      a3 * (PsizFU[jdx + 3 * nx_abc] - PsizFU[jdx - 3 * nx_abc]) +
                      a4 * (PsizFU[jdx + 4 * nx_abc] - PsizFU[jdx - 4 * nx_abc])) / dz;

        float num = -2.0f*(epsilon[i]-delta[i])*((px + psix)*(px + psix))*((pz + psiz)*(pz + psiz));
        float den = (1.0f + 2.0f*epsilon[i])*((px + psix)*(px + psix)*(px + psix)*(px + psix)) + ((pz + psiz)*(pz + psiz)*(pz + psiz)*(pz + psiz)) + 2.0f*(1.0f + delta[i])*((px + psix)*(px + psix))*((pz + psiz)*(pz + psiz));

        if (fabsf(den)<1e-12f){
            Sd = 0.0f;
        }
        else{
            Sd = num / den;
        }

        Uf[i] = 2.0f * Uc[i] - Uf[i] + (vp[i] * vp[i]) * (dt * dt) * ((1.0f+ 2.0f*epsilon[i]) + Sd) * (pxx + psix + ZetaxFR[idx]) + (vp[i] * vp[i]) * (dt * dt) *(1.0f + Sd) * (pzz + psiz + ZetazFU[jdx]);
    }

    // Quina Inferior Esquerda
    if (ix >= 4 && ix < N_abc && iz >= nz_abc - N_abc && iz < nz_abc - 4)
    {
        int idx = iz * N_abc + ix;
        int jdx = (iz - (nz_abc - N_abc)) * nx_abc + ix;
    
        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);
        
        float px = (a1*(Uc[i+1] - Uc[i-1]) +
                    a2*(Uc[i+2] - Uc[i-2]) +
                    a3*(Uc[i+3] - Uc[i-3]) +
                    a4*(Uc[i+4] - Uc[i-4])) / dx;

        float pz = (a1 * (Uc[i + nx_abc] - Uc[i - nx_abc]) +
                    a2 * (Uc[i + 2*nx_abc] - Uc[i - 2*nx_abc]) +
                    a3 * (Uc[i + 3*nx_abc] - Uc[i - 3*nx_abc]) +
                    a4 * (Uc[i + 4*nx_abc] - Uc[i - 4*nx_abc])) / dz;

        float psix = (a1 * (PsixFL[idx + 1] - PsixFL[idx - 1]) +
                      a2 * (PsixFL[idx + 2] - PsixFL[idx - 2]) +
                      a3 * (PsixFL[idx + 3] - PsixFL[idx - 3]) +
                      a4 * (PsixFL[idx + 4] - PsixFL[idx - 4])) / dx;

        float psiz = (a1 * (PsizFD[jdx + nx_abc]     - PsizFD[jdx - nx_abc]) +
                      a2 * (PsizFD[jdx + 2 * nx_abc] - PsizFD[jdx - 2 * nx_abc]) +
                      a3 * (PsizFD[jdx + 3 * nx_abc] - PsizFD[jdx - 3 * nx_abc]) +
                      a4 * (PsizFD[jdx + 4 * nx_abc] - PsizFD[jdx - 4 * nx_abc])) / dz;

        float num = -2.0f*(epsilon[i]-delta[i])*((px + psix)*(px + psix))*((pz + psiz)*(pz + psiz));
        float den = (1.0f + 2.0f*epsilon[i])*((px + psix)*(px + psix)*(px + psix)*(px + psix)) + ((pz + psiz)*(pz + psiz)*(pz + psiz)*(pz + psiz)) + 2.0f*(1.0f + delta[i])*((px + psix)*(px + psix))*((pz + psiz)*(pz + psiz));

        if (fabsf(den)<1e-12f){
            Sd = 0.0f;
        }
        else{
            Sd = num / den;
        }

        Uf[i] = 2.0f * Uc[i] - Uf[i] + (vp[i] * vp[i]) * (dt * dt) * ((1.0f+ 2.0f*epsilon[i]) + Sd) * (pxx + psix + ZetaxFL[idx]) + (vp[i] * vp[i]) * (dt * dt) *(1.0f + Sd) * (pzz + psiz + ZetazFD[jdx]);
    }

    // Quina Inferior Direita
    if (ix >= nx_abc - N_abc && ix < nx_abc - 4 && iz >= nz_abc - N_abc && iz < nz_abc - 4)
    {
        int idx = iz * N_abc + (ix - (nx_abc - N_abc));
        int jdx = (iz - (nz_abc - N_abc)) * nx_abc + ix;

        float pxx = (c0 * Uc[i]
            + c1 * (Uc[i + 1] + Uc[i - 1])
            + c2 * (Uc[i + 2] + Uc[i - 2])
            + c3 * (Uc[i + 3] + Uc[i - 3])
            + c4 * (Uc[i + 4] + Uc[i - 4])) / (dx * dx);
        
        float pzz = (c0 * Uc[i]
            + c1 * (Uc[i + nx_abc] + Uc[i - nx_abc])
            + c2 * (Uc[i + 2*nx_abc] + Uc[i - 2*nx_abc])
            + c3 * (Uc[i + 3*nx_abc] + Uc[i - 3*nx_abc])
            + c4 * (Uc[i + 4*nx_abc] + Uc[i - 4*nx_abc])) / (dz * dz);
        
        float px = (a1*(Uc[i+1] - Uc[i-1]) +
                    a2*(Uc[i+2] - Uc[i-2]) +
                    a3*(Uc[i+3] - Uc[i-3]) +
                    a4*(Uc[i+4] - Uc[i-4])) / dx;

        float pz = (a1 * (Uc[i + nx_abc] - Uc[i - nx_abc]) +
                    a2 * (Uc[i + 2*nx_abc] - Uc[i - 2*nx_abc]) +
                    a3 * (Uc[i + 3*nx_abc] - Uc[i - 3*nx_abc]) +
                    a4 * (Uc[i + 4*nx_abc] - Uc[i - 4*nx_abc])) / dz;

        float psix = (a1 * (PsixFR[idx + 1] - PsixFR[idx - 1]) +
                      a2 * (PsixFR[idx + 2] - PsixFR[idx - 2]) +
                      a3 * (PsixFR[idx + 3] - PsixFR[idx - 3]) +
                      a4 * (PsixFR[idx + 4] - PsixFR[idx - 4])) / dx;

        float psiz = (a1 * (PsizFD[jdx + nx_abc]     - PsizFD[jdx - nx_abc]) +
                      a2 * (PsizFD[jdx + 2 * nx_abc] - PsizFD[jdx - 2 * nx_abc]) +
                      a3 * (PsizFD[jdx + 3 * nx_abc] - PsizFD[jdx - 3 * nx_abc]) +
                      a4 * (PsizFD[jdx + 4 * nx_abc] - PsizFD[jdx - 4 * nx_abc])) / dz;

        float num = -2.0f*(epsilon[i]-delta[i])*((px + psix)*(px + psix))*((pz + psiz)*(pz + psiz));
        float den = (1.0f + 2.0f*epsilon[i])*((px + psix)*(px + psix)*(px + psix)*(px + psix)) + ((pz + psiz)*(pz + psiz)*(pz + psiz)*(pz + psiz)) + 2.0f*(1.0f + delta[i])*((px + psix)*(px + psix))*((pz + psiz)*(pz + psiz));

        if (fabsf(den)<1e-12f){
            Sd = 0.0f;
        }
        else{
            Sd = num / den;
        }

        Uf[i] = 2.0f * Uc[i] - Uf[i] + (vp[i] * vp[i]) * (dt * dt) * ((1.0f+ 2.0f*epsilon[i]) + Sd) * (pxx + psix + ZetaxFR[idx]) + (vp[i] * vp[i]) * (dt * dt) *(1.0f + Sd) * (pzz + psiz + ZetazFD[jdx]);
    }
}
'''

updateWaveEquationVTICPMLKernel = cp.RawKernel(updateWaveEquationVTICPMLCuda, 'updateWaveEquationVTICPMLCuda')

calculateGradientVTICuda = r'''
extern "C" __global__
void calculateGradientVTICuda(const float* current,const float* adj,float* epsilon_partial,float* delta_partial,const float dx,const float dz,const int nx,const int nz,const float* epsilon,const float* delta)
{
    const float c0 = -1435.0f / 504.0f;
    const float c1 =  8.0f / 5.0f;
    const float c2 = -1.0f / 5.0f;
    const float c3 =  8.0f / 315.0f;
    const float c4 = -1.0f / 560.0f;
    const float a1 =  4.0f / 5.0f;
    const float a2 = -1.0f / 5.0f;
    const float a3 =  4.0f / 105.0f;
    const float a4 = -1.0f / 280.0f;

    float dSd_deps;
    float dSd_ddelta;
    float dP_deps;
    float dP_ddelta;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = nz * nx;

    if (i >= total_size) return;

    int iz = i/nx;
    int ix = i%nx;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4) 
    {
        double pxx = (c0 * current[i]
                    + c1 * (current[i + 1] + current[i - 1])
                    + c2 * (current[i + 2] + current[i - 2])
                    + c3 * (current[i + 3] + current[i - 3])
                    + c4 * (current[i + 4] + current[i - 4])) / (dx * dx);

        double pzz = (c0 * current[i]
                    + c1 * (current[i + nx] + current[i - nx])
                    + c2 * (current[i + 2*nx] + current[i - 2*nx])
                    + c3 * (current[i + 3*nx] + current[i - 3*nx])
                    + c4 * (current[i + 4*nx] + current[i - 4*nx])) / (dz * dz);

        double px = (a1*(current[i+1] - current[i-1]) +
                    a2*(current[i+2] - current[i-2]) +
                    a3*(current[i+3] - current[i-3]) +
                    a4*(current[i+4] - current[i-4])) / dx;

        double pz = (a1 * (current[i + nx] - current[i - nx]) +
                    a2 * (current[i + 2*nx] - current[i - 2*nx]) +
                    a3 * (current[i + 3*nx] - current[i - 3*nx]) +
                    a4 * (current[i + 4*nx] - current[i - 4*nx])) / dz;
        
        double num = -2.0f*(epsilon[i]-delta[i])*(px*px)*(pz*pz);
        double den = (1.0f + 2.0f*epsilon[i])*(px*px*px*px) + (pz*pz*pz*pz) + 2.0f*(1.0f + delta[i])*(px*px)*(pz*pz);

        double dnum_deps = -2.0f*px*px*pz*pz;
        double dnum_ddelta = 2.0f*px*px*pz*pz;
        double dden_deps = 2.0f*px*px*px*px;
        double dden_ddelta = 2.0f*px*px*pz*pz;

        if (fabs(den)<1e-150){
            dSd_deps = 0.0f;
            dSd_ddelta = 0.0f;
        }
        else{
            dSd_deps = (dnum_deps*den - num*dden_deps)/(den*den);
            dSd_ddelta = (dnum_ddelta*den - num*dden_ddelta)/(den*den);
        }

        dP_deps = ((2.0f + dSd_deps)*pxx + dSd_deps*pzz);
        dP_ddelta = (dSd_ddelta*(pxx + pzz));

        epsilon_partial[i] += adj[i]*dP_deps;
        delta_partial[i] += adj[i]*dP_ddelta;
    
    }
}
'''    

calculateGradientVTIKernel = cp.RawKernel(calculateGradientVTICuda,"calculateGradientVTICuda")

calculateGradientTTICuda = r'''
extern "C" __global__
void calculateGradientTTICuda(const float* current,const float* adj,float* epsilon_partial,float* delta_partial,float* theta_partial,const float dx,const float dz,const int nx,const int nz,const float* epsilon,const float* delta,const float* theta)
{
    const float c0 = -1435.0f / 504.0f;
    const float c1 =  8.0f / 5.0f;
    const float c2 = -1.0f / 5.0f;
    const float c3 =  8.0f / 315.0f;
    const float c4 = -1.0f / 560.0f;
    const float a1 =  4.0f / 5.0f;
    const float a2 = -1.0f / 5.0f;
    const float a3 =  4.0f / 105.0f;
    const float a4 = -1.0f / 280.0f;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = nz * nx;

    if (i >= total_size) return;

    int iz = i / nx;
    int ix = i % nx;

    if (ix >= 4 && ix < nx - 4 && iz >= 4 && iz < nz - 4)
    {
        double pxx = (c0 * current[i]
            + c1 * (current[i + 1] + current[i - 1])
            + c2 * (current[i + 2] + current[i - 2])
            + c3 * (current[i + 3] + current[i - 3])
            + c4 * (current[i + 4] + current[i - 4])) / (dx * dx);

        double pzz = (c0 * current[i]
            + c1 * (current[i + nx] + current[i - nx])
            + c2 * (current[i + 2 * nx] + current[i - 2 * nx])
            + c3 * (current[i + 3 * nx] + current[i - 3 * nx])
            + c4 * (current[i + 4 * nx] + current[i - 4 * nx])) / (dz * dz);

        double pxz = (a1*a1*(current[i + nx + 1] - current[i - nx + 1] + current[i - nx - 1] - current[i + nx - 1])
        + a1*a2*(current[i + 2*nx + 1] - current[i - 2*nx + 1] + current[i - 2*nx - 1] - current[i + 2*nx - 1])
        + a1*a3*(current[i + 3*nx + 1] - current[i - 3*nx + 1] + current[i - 3*nx - 1] - current[i + 3*nx - 1])
        + a1*a4*(current[i + 4*nx + 1] - current[i - 4*nx + 1] + current[i - 4*nx - 1] - current[i + 4*nx - 1])

        + a2*a1*(current[i + nx + 2] - current[i - nx + 2] + current[i - nx - 2] - current[i + nx - 2])
        + a2*a2*(current[i + 2*nx + 2] - current[i - 2*nx + 2] + current[i - 2*nx - 2] - current[i + 2*nx - 2])
        + a2*a3*(current[i + 3*nx + 2] - current[i - 3*nx + 2] + current[i - 3*nx - 2] - current[i + 3*nx - 2])
        + a2*a4*(current[i + 4*nx + 2] - current[i - 4*nx + 2] + current[i - 4*nx - 2] - current[i + 4*nx - 2])

        + a3*a1*(current[i + nx + 3] - current[i - nx + 3] + current[i - nx - 3] - current[i + nx - 3])
        + a3*a2*(current[i + 2*nx + 3] - current[i - 2*nx + 3] + current[i - 2*nx - 3] - current[i + 2*nx - 3])
        + a3*a3*(current[i + 3*nx + 3] - current[i - 3*nx + 3] + current[i - 3*nx - 3] - current[i + 3*nx - 3])
        + a3*a4*(current[i + 4*nx + 3] - current[i - 4*nx + 3] + current[i - 4*nx - 3] - current[i + 4*nx - 3])

        + a4*a1*(current[i + nx + 4] - current[i - nx + 4] + current[i - nx - 4] - current[i + nx - 4])
        + a4*a2*(current[i + 2*nx + 4] - current[i - 2*nx + 4] + current[i - 2*nx - 4] - current[i + 2*nx - 4])
        + a4*a3*(current[i + 3*nx + 4] - current[i - 3*nx + 4] + current[i - 3*nx - 4] - current[i + 3*nx - 4])
        + a4*a4*(current[i + 4*nx + 4] - current[i - 4*nx + 4] + current[i - 4*nx - 4] - current[i + 4*nx - 4])) / (dz * dx);

        double px = (a1 * (current[i + 1] - current[i - 1])
            + a2 * (current[i + 2] - current[i - 2])
            + a3 * (current[i + 3] - current[i - 3])
            + a4 * (current[i + 4] - current[i - 4])) / dx;

        double pz = (a1 * (current[i + nx] - current[i - nx])
            + a2 * (current[i + 2 * nx] - current[i - 2 * nx])
            + a3 * (current[i + 3 * nx] - current[i - 3 * nx])
            + a4 * (current[i + 4 * nx] - current[i - 4 * nx])) / dz;

        double norm = sqrt(px * px + pz * pz);

        double mx = 0.0f;
        double mz = 0.0f;

        if (norm > 1.0e-150)
        {
            mx = px / norm;
            mz = pz / norm;
        }

        double eps = epsilon[i];
        double del = delta[i];
        double th  = theta[i];

        double costh = cosf(th);
        double sinth = sinf(th);

        double h = mx * costh - mz * sinth;
        double q = mx * sinth + mz * costh;

        double h2 = h * h;
        double q2 = q * q;

        double num = -2.0f * (eps - del) * h2 * q2;

        double den =(1.0f + 2.0f * eps) * h2 * h2 + q2 * q2 + 2.0f * (1.0f + del) * h2 * q2;

        double dSd_deps;
        double dSd_ddelta;
        double dSd_dtheta;

        if (fabs(den) >= 1.0e-150)
        {
            double dnum_deps = -2.0f * h2 * q2;
            double dnum_ddelta = 2.0f * h2 * q2;

            double dden_deps = 2.0f * h2 * h2;
            double dden_ddelta = 2.0f * h2 * q2;

            double dnum_dtheta = -2.0f * (eps - del) * (-2.0f * h * q * q * q+ 2.0f * h * h * h * q);

            double dden_dtheta = -4.0f * (1.0f + 2.0f * eps) * h * h * h * q + 4.0f * h * q * q * q + 2.0f * (1.0f + del) * (-2.0f * h * q * q * q+ 2.0f * h * h * h * q);

            dSd_deps = (dnum_deps * den - num * dden_deps) / (den * den);

            dSd_ddelta = (dnum_ddelta * den - num * dden_ddelta) / (den * den);

            dSd_dtheta = (dnum_dtheta * den - num * dden_dtheta) / (den * den);
        }

        double sin2th = sinf(2.0f * th);
        double cos2th = cosf(2.0f * th);

        double dA_dtheta = -2.0f * eps * sin2th + dSd_dtheta;

        double dB_dtheta = 2.0f * eps * sin2th + dSd_dtheta;

        double dC_dtheta = 4.0f * eps * cos2th;

        double dP_deps = (2.0f * costh * costh + dSd_deps) * pxx + (2.0f * sinth * sinth + dSd_deps) * pzz- 2.0f * sin2th * pxz;

        double dP_ddelta = dSd_ddelta * (pxx + pzz);

        double dP_dtheta = dA_dtheta * pxx + dB_dtheta * pzz - dC_dtheta * pxz;

        epsilon_partial[i] += adj[i] * dP_deps;
        delta_partial[i]   += adj[i] * dP_ddelta;
        theta_partial[i]   += adj[i] * dP_dtheta;    
    }
}
'''
calculateGradientTTIKernel = cp.RawKernel(calculateGradientTTICuda,"calculateGradientTTICuda")
