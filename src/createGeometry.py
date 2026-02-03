import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# sources
sx_init = 500
sx_end  = 2500
Nsource = 3
sx = np.linspace(sx_init,sx_end,Nsource)
sz = 5*np.ones(len(sx))
sIdx = np.arange(Nsource)

# receivers
rx_init = 0
rx_end = 3000
Nrec = 501
rx = np.linspace(rx_init,rx_end,Nrec,endpoint=False)
rz = 10*np.ones(len(rx))
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
