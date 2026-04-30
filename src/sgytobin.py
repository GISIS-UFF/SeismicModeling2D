import numpy as np
import matplotlib.pyplot as plt
import segyio as sgy
import os

#.sgy para .bin
path = "/home/juanmarques/workspace/SeismicModeling2D/inputs/vp_marmousi-ii.segy"
with sgy.open(path, ignore_geometry=True) as segyfile:
    data = np.array(segyfile.trace.raw[:]).T
    # data = data[:,:3149]
    perc=95
    perc_max = np.percentile(data, perc)
    perc_min = np.percentile(data, 100-perc)
    plt.figure()
    plt.imshow(data, aspect='auto', cmap='gray', vmin=perc_min, vmax=perc_max)
    plt.show()
    basename = os.path.splitext(path)[0]
    dataFile = f"{basename}_shape_{data.shape}.bin"
    data.tofile(dataFile)
    print(f"info: Model saved to {dataFile}")

#--------------------------------------------------------------------------------------------------

#Reamostrando o modelo

bin_path = "/home/juanmarques/workspace/SeismicModeling2D/inputs/vp_marmousi-ii_shape_(2801, 13601)_dh10m_Nz351_Nx1701.bin"

nz_old = 351
nx_old = 1701

dh_old = 10.0
dh_new = 10.0

fator = int(dh_new / dh_old)
vp = np.fromfile(bin_path, dtype=np.float32).reshape(nz_old, nx_old)
vp_new = vp[::fator, ::fator]

# Converte de km/s para m/s
vp_new = vp_new * 1000.0

nz_new, nx_new = vp_new.shape

basename = os.path.splitext(bin_path)[0]
out_path = f"{basename}_dh{dh_new:g}m_ms_Nz{nz_new}_Nx{nx_new}.bin"

vp_new.astype(np.float32).tofile(out_path)

plt.figure(figsize=(12, 4))
plt.imshow(vp_new, cmap="gray", aspect="auto")
plt.colorbar(label="Velocity (m/s)")
plt.xlabel("Distance index")
plt.ylabel("Depth index")
plt.show()