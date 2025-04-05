import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit
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

sism_analitico = ler_sismograma('D:/GitHub/ModelagemSismica/outputs/seismograms/sismograma_analitico_shot_0_4001x501.bin', (4001, 501))
sism = ler_sismograma('D:/GitHub/ModelagemSismica/outputs/seismograms/sismograma_shot_0_4001x501.bin', (4001, 501))
plot_sismograma(sism)
plot_sismograma(sism_analitico)
sism_residual = sism_analitico - sism
plot_sismograma(sism_residual)

plt.figure()
plt.plot(sism_analitico[:,100], label='analitico')
plt.plot(sism[:,100], label='numerico')
plt.legend()
plt.show()