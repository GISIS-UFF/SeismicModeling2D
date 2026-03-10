import cupy as cp

updateWaveEquationCuda = r'''
extern "C" __global__
void updateWaveEquationCuda(float* Uf,const float* Uc,const float* vp,const int nz,const int nx,const float dz,const float dx,const float dt)
{
    const float c0 = -205.0f / 72.0f;
    const float c1 =  8.0f   / 5.0f;
    const float c2 = -1.0f   / 5.0f;
    const float c3 =  8.0f   / 315.0f;
    const float c4 = -1.0f   / 560.0f;

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