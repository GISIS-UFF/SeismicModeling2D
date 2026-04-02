from utils import Mute
from survey import parameters
import matplotlib.pyplot as plt
import numpy 

pmt = parameters("../inputs/Parameters.json")
shot = 0
seismogramFile = f"{pmt.seismogramFolder}seismogram_shot_{shot+1}_Nt{pmt.nt}_Nrec{pmt.Nrec}.bin"
seismogram = numpy.fromfile(seismogramFile, dtype=numpy.float32).reshape(pmt.nt,pmt.Nrec) 
muted_seismogram = Mute(seismogram, shot, pmt.rec_x, pmt.rec_z, pmt.shot_x, pmt.shot_z, pmt.dt, shift = 0.22,window = 0.05,v0=1500)

plt.figure()
plt.plot(seismogram[:,200], label = "seismogram")
plt.plot(muted_seismogram[:,200], label = "muted")
plt.legend()
plt.figure()
plt.imshow(muted_seismogram, aspect="auto")
plt.colorbar()
plt.show()