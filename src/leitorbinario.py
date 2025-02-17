import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def ler_sismograma(caminho_arquivo, shape):
    sism = np.fromfile(caminho_arquivo, dtype=np.float64)  
    sism = sism.reshape(shape)
    print(f"Sismograma carregado de: {caminho_arquivo}")
    return sism

def plot_sismograma(sism):
    plt.figure()
    perc = np.percentile(sism,99)
    plt.imshow(sism,aspect='auto',cmap='gray',vmin=-perc,vmax=perc)
    plt.colorbar(label='Amplitude')
    plt.title("Sismograma")
    plt.show(block = False)

def ler_modelo(caminho_arquivo, shape):
    vp = np.fromfile(caminho_arquivo, dtype=np.float32)
    vp = vp.reshape(shape).T
    print(f"Modelo de velocidade carregado de: {caminho_arquivo}")
    return vp

def plot_modelo(vp):
    plt.imshow(vp, aspect='auto', cmap='jet')
    plt.colorbar()
    plt.show()

def ler_snapshot(caminho_arquivo, shape):
    u_snapshot= np.fromfile(caminho_arquivo, dtype=np.float32)
    u_snapshot = u_snapshot.reshape(shape)
    return u_snapshot

def plot_snapshot(u_snapshot, shot, nt):
    fig, ax = plt.subplots(figsize=(10, 10))
    for k in range(nt):
        if (k%100 == 0):
            ax.cla()
            ax.imshow(u_snapshot[shot, k])
            plt.pause(0.1)
    
    
# snapshot = ler_snapshot('D:/GitHub/Acousticwave/outputs/snapshots/snapshot_0x2001x141x383.bin', (3, 2001, 141, 383))

# plot_snapshot(snapshot, 0, 2001)

sism_analitico = ler_sismograma('D:/GitHub/ModelagemSismica/outputs/seismograms/sismograma_analitico_shot_1_2001x5001.bin', (2001, 5001))
sism = ler_sismograma('D:/GitHub/ModelagemSismica/outputs/seismograms/sismograma_shot_1_2001x5001.bin', (2001, 5001))
plot_sismograma(sism)
plot_sismograma(sism_analitico)
sism_residual = sism_analitico - sism
plot_sismograma(sism_residual)

# vp = ler_modelo('D:/GitHub/Acousticwave/inputs/marmousi_vp_383x141.bin', (383, 141))
# plot_modelo(vp)