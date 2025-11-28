import numpy as np
import matplotlib.pyplot as plt
import segyio as sgy
import os

keyword = "bp2007"

for filename in sorted(os.listdir("../inputs/")):
    if keyword in filename and filename.endswith(".sgy"):
        path = os.path.join("../inputs/", filename)
        with sgy.open(path, ignore_geometry=True) as segyfile:
            data = np.array(segyfile.trace.raw[:]).T
            data = data[:,:3149]
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

