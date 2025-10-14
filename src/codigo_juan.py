
#%%
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import sys

def wavelet_ricker(nt,freq,dt,time_init,time_lag,amp):
    fc=freq/(3.*np.sqrt(np.pi));t = np.arange(nt)*dt - time_init;wavelet = -amp*(2.*np.pi*(np.pi*fc*t)**2 - 1.)*np.exp(-np.pi*(np.pi*fc*t)**2)
    return wavelet


def UpdateFD_8thorder(p_old, p_curr, Vp, dx, dz, dt):
    # 8th-order FD coefficients for second derivative
    c0 = -205.0 / 72.0
    c1 = 8.0 / 5.0
    c2 = -1.0 / 5.0
    c3 = 8.0 / 315.0
    c4 = -1.0 / 560.0

    # Second derivative in x (axis=1)
    d2u_dx2 = (
        c4 * cp.roll(p_curr, -4, axis=1) +
        c3 * cp.roll(p_curr, -3, axis=1) +
        c2 * cp.roll(p_curr, -2, axis=1) +
        c1 * cp.roll(p_curr, -1, axis=1) +
        c0 * p_curr +
        c1 * cp.roll(p_curr, 1, axis=1) +
        c2 * cp.roll(p_curr, 2, axis=1) +
        c3 * cp.roll(p_curr, 3, axis=1) +
        c4 * cp.roll(p_curr, 4, axis=1)
    ) / (dx ** 2)

    # Second derivative in z (axis=0)
    d2u_dz2 = (
        c4 * cp.roll(p_curr, -4, axis=0) +
        c3 * cp.roll(p_curr, -3, axis=0) +
        c2 * cp.roll(p_curr, -2, axis=0) +
        c1 * cp.roll(p_curr, -1, axis=0) +
        c0 * p_curr +
        c1 * cp.roll(p_curr, 1, axis=0) +
        c2 * cp.roll(p_curr, 2, axis=0) +
        c3 * cp.roll(p_curr, 3, axis=0) +
        c4 * cp.roll(p_curr, 4, axis=0)
    ) / (dz ** 2)

    # Time stepping (acoustic wave equation)
    p_new = 2.0 * p_curr - p_old + (Vp ** 2) * (dt ** 2) * (d2u_dx2 + d2u_dz2)

    return p_new


def add_extended_borders(velocity_model_cpu, n_border):
    """
    Adds extended borders to a velocity model on the CPU.
    The sides are extended laterally, and the top/bottom are extended,
    with the corners being filled correctly.

    Args:
        velocity_model_cpu (np.ndarray): The input 2D velocity model on the CPU.
        n_border (int): The number of cells to add for the border on all sides.

    Returns:
        np.ndarray: The new, padded velocity model on the CPU.
    """
    if n_border == 0:
        return velocity_model_cpu
        
    print(f"Adding {n_border}-cell extended borders to the model...")
    Nz, Nx = velocity_model_cpu.shape
    Nz_new, Nx_new = Nz + 2 * n_border, Nx + 2 * n_border
    
    # Create the new padded array
    padded_model = np.zeros((Nz_new, Nx_new), dtype=np.float32)
    
    # --- Step 1: Copy the original model into the center ---
    padded_model[n_border : n_border + Nz, n_border : n_border + Nx] = velocity_model_cpu
    
    # --- Step 2: Extend the sides laterally ---
    # Left border: copy the first column of the original model
    left_border = velocity_model_cpu[:, 0:1] # Shape (Nz, 1)
    padded_model[n_border : n_border + Nz, 0:n_border] = left_border
    
    # Right border: copy the last column of the original model
    right_border = velocity_model_cpu[:, -1:] # Shape (Nz, 1)
    padded_model[n_border : n_border + Nz, n_border + Nx :] = right_border
    
    # --- Step 3: Extend the top and bottom, including the new corners ---
    # Top border: copy the first row of the now side-padded model
    top_border = padded_model[n_border:n_border+1, :] # Shape (1, Nx_new)
    padded_model[0:n_border, :] = top_border
    
    # Bottom border: copy the last row of the now side-padded model
    bottom_border = padded_model[n_border+Nz-1:n_border+Nz, :] # Shape (1, Nx_new)
    padded_model[n_border + Nz :, :] = bottom_border
    
    print("Border padding finished.")
    return padded_model



print("Python:", sys.version)
cp.cuda.Device().synchronize()



dx = 20.0  # meters
dz = 20.0  # meters

Nz = 261
Nx = 676
n_border=300
Vp_cpu = np.ones((Nz,Nx)) * 2000.
Vp_cpu = add_extended_borders(Vp_cpu, n_border=n_border)

Nzz, Nxx = Vp_cpu.shape


# physical velocity (m/s)
C_min = Vp_cpu.min()
C_max = Vp_cpu.max()

# ---------------- parameters ----------------
dt = 0.4 * dx / C_max   # approximate 2D CFL criterion
tmax = 6

nt = int(tmax / dt) + 1
time_lag = 0.2
maxAmplitude = 1.0


# Vp_cpu = np.zeros((Nz, Nx), dtype=np.float32)

# Vp_cpu[:100,:] = C_min
# Vp_cpu[100:,:] = C_max


Vp = cp.asarray(Vp_cpu, dtype=cp.float32)

# CFL check (rough)
cfl = C_max * dt / dx  # approximate 2D CFL criterion
print("approx CFL number:", cfl)
if cfl > 1.0:
    print("WARNING: CFL > 1, reduce dt or increase dx/dz for stability.")


# 1. Calculate the stability limit from the time step (dt)
# This is the highest frequency the time-stepping scheme can handle before
# becoming unstable.
f_max_time = 1.0 / (np.pi * dt)

# 2. Calculate the spatial aliasing limit from the grid spacing (dx)
# This is the highest frequency that can be represented on the grid without
# aliasing, determined by the Nyquist frequency at the lowest velocity.
f_max_spatial = C_min / (2.0 * dx)

# 3. The true limit is the stricter (lower) of the two constraints
f_max_usable = min(f_max_time, f_max_spatial)

# 4. It's best practice to stay safely below the theoretical limit.
# 80% is a common and safe margin.
f_practical = f_max_usable

fcut=f_practical


# #%%
# ---------------- grids / fields (shape: [Nz, Nx]) ----------------
# Use (Nz, Nx) where axis 0 is z (rows), axis 1 is x (cols)
p_new = cp.zeros((Nzz, Nxx), dtype=cp.float32)
p_curr = cp.zeros((Nzz, Nxx), dtype=cp.float32)
p_old = cp.zeros((Nzz, Nxx), dtype=cp.float32)

seismogram = cp.zeros((nt, Nx), dtype=cp.float32)

# source indices (z,x)
src_z = 10
src_x = Nxx//2

# Wavelet: do NOT divide by dx/dz here unless you intentionally want
# to normalize by cell area. Keep a plain amplitude wavelet and scale later if needed.
ricker_cpu = dt*dt*np.asarray(wavelet_ricker(nt, fcut, dt, time_lag, 0.0, maxAmplitude), dtype=np.float32)/dx/dz

# If you prefer to inject scaled by dt^2 (as in finite-diff), you can multiply,
# but do that deliberately. I'll keep the raw amplitude:
source = cp.asarray(ricker_cpu, dtype=cp.float32)  # shape (nt,)



src_z += n_border
# src_x += pad

# ---------------- visualization setup ----------------
plt.ion()
fig, ax = plt.subplots(figsize=(7,7))
im = ax.imshow(cp.asnumpy(p_curr), cmap='gray_r', origin='upper', vmin=None, vmax=None)
ax.set_xlabel("x (m)")
ax.set_ylabel("z (m)")
ax.set_title("t = 0.0 s")
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label("pressure amplitude")

def apply_absorbing_boundary(p, npml=20, alpha=3.0):
    Nx, Nz = p.shape
    damp_x = cp.ones(Nx)
    damp_z = cp.ones(Nz)

    # exponential taper
    for i in range(npml):
        val = cp.exp(-alpha * ((npml - i) / npml)**2)
        damp_x[i] *= val
        damp_x[-i-1] *= val
        damp_z[i] *= val
        damp_z[-i-1] *= val

    # apply separable damping
    D = damp_x[:, None] * damp_z[None, :]
    p *= D

    return p
Vp_const = cp.ones((Nzz,Nxx))*1500.
# ---------------- main time loop ----------------
for it in range(nt):

    # inject source (add to p_new at source location)
    # optional: multiply source amplitude by a factor (e.g. dt**2) depending on convention
    p_curr[src_z, src_x] += source[it]

    p_new = UpdateFD_8thorder(p_old, p_curr, Vp, dx, dz, dt)

    # apply damping sponge to reduce wrap-around reflections
    # p_new = apply_absorbing_boundary(p_new, npml=40, alpha=0.5)
    # p_curr = apply_absorbing_boundary(p_curr, npml=40, alpha=0.5)
    # p_old = apply_absorbing_boundary(p_old, npml=40, alpha=0.5)
    
    # register seismogram
    seismogram[it,:] = p_new[src_z,n_border:-n_border]

    # rotate time levels
    p_old, p_curr = p_curr, p_new

    if it == nt-1:
        plt.figure()
        plt.plot(arr[:,Nxx//2])

    # plotting/visual update every N steps
    if it % 50 == 0:
        arr = cp.asnumpy(p_new)
        im.set_data(arr)

        mx = 1e-10
        im.set_clim(-mx, mx)
        ax.set_title(f"Non-reflecting wave equation t = {it*dt:.3f} s")
        plt.pause(0.001)
        

plt.ioff()
plt.show()

# --- Cleanup GPU memory and FFT plans ---

# Delete CuPy arrays explicitly (optional if script exits, but good practice for long runs)
del p_curr, p_old, p_new
del Vp
del source

# Free any unused memory blocks in CuPy's memory pool
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

# Clear FFT plan cache (CuPy keeps cuFFT plans around for reuse)
cp.fft.config.clear_plan_cache()

print("GPU memory and FFT plans cleared.")

print("finished")


# %%
