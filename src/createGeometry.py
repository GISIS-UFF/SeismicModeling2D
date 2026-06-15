import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# sources
sx_init = 1000
sx_end  = 17000
Nsource = 32
sx = np.linspace(sx_init,sx_end,Nsource, endpoint = "True")
sz = 20*np.ones(len(sx))
sIdx = np.arange(Nsource)

# receivers
rx_init = 50
rx_end = 17000
Nrec = 170
rx = np.linspace(rx_init,rx_end,Nrec, endpoint = "True")
rz = 450*np.ones(len(rx))
rIdx = np.arange(Nrec)

plt.figure()
plt.plot(sx,sz,"r*", markersize=10)
plt.plot(rx,rz,'bv')
plt.xlim(0,rx_end)
plt.ylim(rx_end,0)
plt.show()

receiver_df = pd.DataFrame({'index': rIdx,'coordx': rx,'coordz': rz})

source_df = pd.DataFrame({'index': sIdx,'coordx': sx,'coordz': sz})

receiver_df.to_csv("../inputs/receivers.csv", index=False)
source_df.to_csv("../inputs/sources.csv", index=False)
