import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import time
import cupy as cp

from utils import ricker
from utils import updateWaveEquationGPU
from utils import AbsorbingBoundaryGPU

with open("../inputs/Teste.json") as f:
    parameters = json.load(f)

    # Approximation type
    approximation = parameters["approximation"]
    
    # Discretization parameters
    dx   = parameters["dx"]
    dz   = parameters["dz"]
    dt   = parameters["dt"]
    
    # Model size
    L    = parameters["L"]
    D    = parameters["D"]
    T    = parameters["T"]

    # Number of point for absorbing boundary condition
    N_abc = parameters["N_abc"]

    # Number of points in each direction
    nx = int(L/dx)+1
    nz = int(D/dz)+1
    nt = int(T/dt)+1

    nx_abc = nx + 2*N_abc
    nz_abc = nz + 2*N_abc

    # Define arrays for space and time
    x = np.linspace(0, L, nx)
    z = np.linspace(0, D, nz)
    t = np.linspace(0, T, nt)

    # Max frequency
    fcut = parameters["fcut"]

    # Output folders
    seismogramFolder = parameters["seismogramFolder"]
    migratedimageFolder = parameters["migratedimageFolder"]
    snapshotFolder = parameters["snapshotFolder"]
    modelFolder = parameters["modelFolder"]

    # Source and receiver files
    rec_file = parameters["rec_file"]
    src_file = parameters["src_file"]

    # Velocity model file
    vpFile = parameters["vpFile"]
    vsFile = parameters["vsFile"]

    # Snapshot flag
    frame      = parameters["frame"] # time steps to save snapshots
    shot_frame = parameters["shot_frame"] # shots to save snapshots

    #Anisotropy parameters for Layered model
    vpLayer1 = parameters["vpLayer1"]
    vpLayer2 = parameters["vpLayer2"]

# Read receiver and source coordinates from CSV files
receiverTable = pd.read_csv(rec_file)
print(f"info: Imported: {rec_file}")     
sourceTable = pd.read_csv(src_file)
print(f"info: Imported: {src_file}")

# Read receiver and source coordinates
rec_x = receiverTable['coordx'].to_numpy()
rec_z = receiverTable['coordz'].to_numpy()
shot_x = sourceTable['coordx'].to_numpy()
shot_z = sourceTable['coordz'].to_numpy()

Nrec = len(rec_x)
Nshot = len(shot_x) 

# Create Ricker wavelet
source = ricker(fcut, t)
print(f"info: Ricker Source wavelet created: {nt} samples")

def ExpandModel(N_abc, nz_abc, nx_abc, model_data):
    N = N_abc
    
    model_exp = np.zeros((nz_abc, nx_abc),dtype=np.float32)
    model_exp[N:nz_abc-N, N:nx_abc-N] = model_data
    model_exp[0:N, N:nx_abc-N] = model_data[0, :]  
    model_exp[nz_abc-N:nz_abc, N:nx_abc-N] = model_data[-1, :]  
    model_exp[N:nz_abc-N, 0:N] = model_data[:, 0:1] 
    model_exp[N:nz_abc-N, nx_abc-N:nx_abc] = model_data[:, -1:]  
    model_exp[0:N, 0:N] = model_data[0, 0]  
    model_exp[0:N, nx_abc-N:nx_abc] = model_data[0, -1] 
    model_exp[nz_abc-N:nz_abc, 0:N] = model_data[-1, 0]  
    model_exp[nz_abc-N:nz_abc, nx_abc-N:nx_abc] = model_data[-1, -1] 
    print(f"info: Model expanded to {nz_abc}x{nx_abc}")
    # plt.figure()
    # plt.imshow(model_exp, cmap='jet', aspect='auto')
    
    return model_exp

# Initialize velocity model and wavefields
vp         = np.zeros([nz,nx],dtype=np.float32)
current    = np.zeros([nz_abc,nx_abc],dtype=np.float32)
future     = np.zeros([nz_abc,nx_abc],dtype=np.float32)
seismogram = np.zeros([nt,Nrec],dtype=np.float32)
snapshot    = np.zeros([nt,nz_abc,nx_abc],dtype=np.float32)


vp[0:nz//2, :] = vpLayer1
vp[nz//2:nz, :] = vpLayer2

def createCerjanVector(N_abc):
    sb = 3 * N_abc
    A = np.ones(N_abc,dtype = np.float32)
    for i in range(N_abc):
            fb = (N_abc - i) / (np.sqrt(2) * sb)
            A[i] = np.exp(-fb * fb)
            
    return A 

start_time = time.time()
print(f"info: Solving acoustic wave equation")
# Expand velocity model and Create absorbing layers
vp_exp = ExpandModel(N_abc, nz_abc, nx_abc, vp)
vp_exp = cp.asarray(vp_exp)
A = createCerjanVector(N_abc)
A = cp.asarray(A)
future = cp.asarray(future)
current = cp.asarray(current)
seismogram = cp.asarray(seismogram)
snapshot = cp.asarray(snapshot)
source = cp.asarray(source)

threadsperblock = (16, 16)
blockspergrid_x = int(np.ceil(current.shape[1] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(current.shape[0] / threadsperblock[1]))  
blockspergrid   = (blockspergrid_x, blockspergrid_y)

rx = cp.int32(rec_x/dx) + N_abc
rz = cp.int32(rec_z/dz) + N_abc

for shot in range(Nshot):
    print(f"info: Shot {shot+1} of {Nshot}")
    current.fill(0)
    future.fill(0)
    seismogram.fill(0)
    snapshot.fill(0)

    # convert acquisition geometry coordinates to grid points
    sx = cp.int32(shot_x[shot]/dx) + N_abc
    sz = cp.int32(shot_z[shot]/dz) + N_abc           

    for k in range(nt):        
        current[sz,sx] += source[k]
        updateWaveEquationGPU[blockspergrid,threadsperblock](future, current, vp_exp, nz_abc, nx_abc, dz, dx, dt)

        # Apply absorbing boundary condition
        AbsorbingBoundaryGPU[blockspergrid,threadsperblock](N_abc, nz_abc, nx_abc, future, A)
        AbsorbingBoundaryGPU[blockspergrid,threadsperblock](N_abc, nz_abc, nx_abc, current, A)

        # Register seismogram
        seismogram[k, :] = current[rz, rx]

        snapshot[k, :, :] = current

        if (shot + 1) in shot_frame and k in frame:
            snapshotFile = f"{snapshotFolder}Acoustic_shot_{shot+1}_Nx{nx}_Nz{nz}_Nt{nt}_frame_{k}.bin"
            snapshot_cpu = cp.asnumpy(snapshot[k, :, :])  
            snapshot_cpu.tofile(snapshotFile)
            print(f"info: Snapshot saved to {snapshotFile}")
        
        #swap
        current, future = future, current
    
    seismogramFile = f"{seismogramFolder}Acousticseismogram_shot_{shot+1}_Nt{nt}_Nrec{Nrec}.bin"
    seismogram_cpu = cp.asnumpy(seismogram)
    seismogram_cpu.tofile(seismogramFile)
    print(f"info: Seismogram saved to {seismogramFile}")
    print(f"info: Shot {shot+1} completed in {time.time() - start_time:.2f} seconds")