import numpy as np
import matplotlib.pyplot as plt
import segyio as sgy
import os

#.sgy para .bin
# path = "/home/juanmarques/workspace/SeismicModeling2D/inputs/vp_marmousi-ii.segy"
# with sgy.open(path, ignore_geometry=True) as segyfile:
#     data = np.array(segyfile.trace.raw[:]).T
#     # data = data[:,:3149]
#     perc=95
#     perc_max = np.percentile(data, perc)
#     perc_min = np.percentile(data, 100-perc)
#     plt.figure()
#     plt.imshow(data, aspect='auto', cmap='gray', vmin=perc_min, vmax=perc_max)
#     plt.show()
#     basename = os.path.splitext(path)[0]
#     dataFile = f"{basename}_shape_{data.shape}.bin"
#     data.tofile(dataFile)
#     print(f"info: Model saved to {dataFile}")

#--------------------------------------------------------------------------------------------------

bin_path = "/home/processamento/SeismicModeling2D/inputs/models/vp_marmousi-ii_shape_(2801, 13601).bin"

nz_old = 2801
nx_old = 13601

dh_old = 1.25
dh_new = 10.0

fator = int(dh_new / dh_old)
vp = np.fromfile(bin_path, dtype=np.float32).reshape(nz_old, nx_old)
vp_new = vp[::fator, ::fator]

# Converte de km/s para m/s
vp_new = vp_new * 1000.0

# Tamanho desejado
nz_target = 351
nx_target = 851

# Recorte vertical: usa toda a profundidade reamostrada
z0 = 0
z1 = z0 + nz_target

# Recorte horizontal: região central do Marmousi reamostrado
nx_resampled = vp_new.shape[1]
x0 = (nx_resampled - nx_target) // 2
x1 = x0 + nx_target

vp_new = vp_new[z0:z1, x0:x1]

nz_new, nx_new = vp_new.shape

basename = os.path.splitext(bin_path)[0]
out_path = f"{basename}_dh{dh_new:g}m_Nz{nz_new}_Nx{nx_new}.bin"

vp_new.astype(np.float32).tofile(out_path)

print(f"info: Model saved to {out_path}")

plt.figure(figsize=(12, 4))
plt.imshow(vp_new,cmap="gray",aspect="equal",extent=[0, (nx_new - 1) * dh_new, (nz_new - 1) * dh_new, 0])
plt.colorbar(label="Velocity (m/s)")
plt.xlabel("Distance (m)")
plt.ylabel("Depth (m)")
plt.tight_layout()
plt.show()