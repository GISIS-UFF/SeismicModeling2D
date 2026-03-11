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

updateWaveEquationVTICuda = r'''
extern "C" __global__
void updateWaveEquationVTICuda(float* Uf,const float* Uc,const int nx,const int nz,const float dt,const float dx,const float dz,const float* vp,const float* epsilon,const float* delta )
{
    const float c0 = -205.0f / 72.0f;
    const float c1 =  8.0f   / 5.0f;
    const float c2 = -1.0f   / 5.0f;
    const float c3 =  8.0f   / 315.0f;
    const float c4 = -1.0f   / 560.0f;
    const float a1 = 672.0f  / 840.0f;
    const float a2 = -168.0f / 840.0f;
    const float a3 = 32.0f   / 840.0f;
    const float a4 = -3.0f   / 840.0f;
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
    const float c0 = -205.0f / 72.0f;
    const float c1 =  8.0f   / 5.0f;
    const float c2 = -1.0f   / 5.0f;
    const float c3 =  8.0f   / 315.0f;
    const float c4 = -1.0f   / 560.0f;

    const float a1 =  672.0f / 840.0f;
    const float a2 = -168.0f / 840.0f;
    const float a3 =   32.0f / 840.0f;
    const float a4 =   -3.0f / 840.0f;

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