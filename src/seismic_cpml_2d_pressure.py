import matplotlib.pyplot as plt
import numpy as np
import time


def get_dampening_profiles(velocity, NPOINTS_PML, Rcoef, K_MAX_PML, ALPHA_MAX_PML, NPOWER, DELTAT, DELTAS, dtype=np.float64, qc=False):
    "Get dampening profiles for seismic CPML. Can cope with any number of dimensions defined by velocity array"
    
    profiles = []
    if qc: qc_profiles = []
    
    assert len(DELTAS) == len(velocity.shape)

    maxvel = np.max(velocity)
    
    # for each dimension
    for iN, N in enumerate(velocity.shape):
        
        DELTA = DELTAS[iN]
        
        thickness_PML = NPOINTS_PML * DELTA
        d0 = - (NPOWER + 1) * maxvel * np.log(Rcoef) / (2 * thickness_PML)
        
        d, d_half, K, K_half, alpha, alpha_half, b, b_half, a, a_half = np.zeros(N, dtype=dtype), np.zeros(N, dtype=dtype), np.zeros(N, dtype=dtype), np.zeros(N, dtype=dtype), np.zeros(N, dtype=dtype), np.zeros(N, dtype=dtype), np.zeros(N, dtype=dtype), np.zeros(N, dtype=dtype), np.zeros(N, dtype=dtype), np.zeros(N, dtype=dtype)
        K[:], K_half[:] = 1,1
        
        originleft = thickness_PML
        originright = (N-1)*DELTA - thickness_PML
        
        # calculate profile for each element
        for i in range(N):
        
            # abscissa of current grid point along the damping profile
            val = DELTA * i
            
            # define damping profile at the grid points, left and right edges
            for abscissa_in_PML in [originleft - val, val - originright]:
                if (abscissa_in_PML >= 0.):# if in PML
                    abscissa_normalized = abscissa_in_PML / thickness_PML
                    d[i] = d0 * (abscissa_normalized**NPOWER)
                    # from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
                    K[i] = 1 + (K_MAX_PML - 1) * (abscissa_normalized**NPOWER)
                    alpha[i] = ALPHA_MAX_PML * (1 - abscissa_normalized)
            
            # define damping profile at half the grid points, left and right edges
            for abscissa_in_PML in [originleft - (val + DELTA/2.), (val + DELTA/2.) - originright]:
                if (abscissa_in_PML >= 0.):
                    abscissa_normalized = abscissa_in_PML / thickness_PML
                    d_half[i] = d0 * (abscissa_normalized**NPOWER)
                    # from Stephen Gedney's unpublished class notes for class EE699, lecture 8, slide 8-2
                    K_half[i] = 1 + (K_MAX_PML - 1) * (abscissa_normalized**NPOWER)
                    alpha_half[i] = ALPHA_MAX_PML * (1 - abscissa_normalized)
            
            b[i] = np.exp(-(d[i] / K[i] + alpha[i]) * DELTAT)
            b_half[i] = np.exp(-(d_half[i] / K_half[i] + alpha_half[i]) * DELTAT)
            
            # this to avoid division by zero outside the PML
            if (np.abs(d[i]) > 1e-6):
                a[i] = d[i] * (b[i] - 1) / (K[i] * (d[i] + K[i] * alpha[i]))
            if (np.abs(d_half[i]) > 1e-6):
                a_half[i] = d_half[i] * (b_half[i] - 1) / (K_half[i] * (d_half[i] + K_half[i] * alpha_half[i]))
        
        
        # save all profiles if in qc mode
        if qc:
            qc_profiles.append([d.copy(), d_half.copy(), K.copy(), K_half.copy(), alpha.copy(), alpha_half.copy(), b.copy(), b_half.copy(), a.copy(), a_half.copy()])
            
            
        # reshape profile so that it can be broadcasted onto the velocity array
        shape = np.ones(len(velocity.shape), dtype=int)
        shape[iN] = N

        a, a_half = a.reshape(shape), a_half.reshape(shape)
        b, b_half = b.reshape(shape), b_half.reshape(shape)
        K, K_half = K.reshape(shape), K_half.reshape(shape)
        
        profiles.append([a.copy(), a_half.copy(), b.copy(), b_half.copy(), K.copy(), K_half.copy()])
    
    if qc:
        return profiles, qc_profiles
    return profiles
    
 