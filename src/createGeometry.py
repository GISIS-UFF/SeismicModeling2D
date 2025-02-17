import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# sources
sx_init = 1000
sx_end  = 4000
Nsource = 3
sx = np.linspace(sx_init,sx_end,Nsource)
sz = 1000*np.ones(len(sx))
sIdx = np.arange(Nsource)

# sx_init = 1000
# sx_end  = 5000
# Nsource = 3
# sx = np.linspace(sx_init,sx_end,Nsource)
# sz = 50 * np.ones(len(sx))
# sIdx = np.arange(Nsource)

# receivers
rx_init = 0
rx_end = 5001
Nrec = 5001
rx = np.linspace(rx_init,rx_end,Nrec,endpoint=False)
rz = 1000*np.ones(len(rx))
rIdx = np.arange(Nrec)

# rx_init = 0
# rx_end = 5731
# Nrec = 5730
# rx = np.linspace(rx_init,rx_end,Nrec,endpoint=False)
# rz = np.ones(len(rx))
# rIdx = np.arange(Nrec)


plt.figure()
plt.plot(sx,sz,"r*", markersize=10)
plt.plot(rx,rz,'bv')
plt.xlim(0,rx_end)
plt.ylim(rx_end,0)
plt.show()

receiver_df = pd.DataFrame({'index': rIdx,'coordx': rx,'coordz': rz})

source_df = pd.DataFrame({'index': sIdx,'coordx': sx,'coordz': sz})

receiver_df.to_csv("D:/GitHub/ModelagemSismica/inputs/receivers.csv", index=False)
source_df.to_csv("D:/GitHub/ModelagemSismica/inputs/sources.csv", index=False)
