import numpy as np
import matplotlib.pyplot as plt

def ler_sismograma(caminho_arquivo, shape):
    sism = np.fromfile(caminho_arquivo, dtype=np.float32)  
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

snapshot = ler_sismograma('D:/GitHub/ModelagemSismica/outputs/snapshots/snapshot_frame_1000_shot_0_nx_501_nz_501vti.bin', (501, 501))
# plot_sismograma(snapshot)

delta = 0.1
epsilon = 0.24
vpz = 3000
dx = dz = 10      
dt = 0.0005         
tempo_snapshot = 0.5
theta = np.linspace(0, 2*np.pi, 1000)
fonte_x, fonte_z = 250, 250  

vp_elip = vpz * np.sqrt(1 + 2 * epsilon * (np.sin(theta)**2)) 
vp_frac = vpz * (1 + delta * (np.sin(theta)**2) * (np.cos(theta)**2) + epsilon * (np.sin(theta)**4)) 

scale_pixels = 1 / dx  
raio_elip = vp_elip * tempo_snapshot * scale_pixels
raio_frac = vp_frac * tempo_snapshot * scale_pixels

x_elip = fonte_x + raio_elip * np.sin(theta)
z_elip = fonte_z - raio_elip * np.cos(theta)  
x_frac = fonte_x + raio_frac * np.sin(theta)
z_frac = fonte_z - raio_frac * np.cos(theta)

fig,ax = plt.subplots()
perc = np.percentile(snapshot,99)
ax.imshow(snapshot, cmap='gray', vmin=-perc, vmax=perc)
plt.grid(False)
plt.plot(x_elip, z_elip,'b-', label='Vp_elip')
plt.plot(x_frac, z_frac,'r--', label='Vp_frac')

plt.show()