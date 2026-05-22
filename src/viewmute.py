from utils import Mute
from survey import parameters
import matplotlib.pyplot as plt
import numpy 

pmt = parameters("../inputs/Parameters.json")
shot_file = 30
shot_idx = shot_file - 1
shift = 0.15
window = 0.05
seismogramFile = (f"{pmt.seismogramFolder}"f"seismogram_shot_{shot_file}_Nt{pmt.nt}_Nrec{pmt.Nrec}.bin")
seismogram = numpy.fromfile(seismogramFile,dtype=numpy.float32).reshape(pmt.nt, pmt.Nrec)
muted_seismogram = Mute(seismogram,shot_idx,pmt.rec_x,pmt.rec_z,pmt.shot_x,pmt.shot_z,pmt.dt,pmt.tlag,shift,window,v0=1500)

dist = numpy.sqrt((pmt.rec_z - pmt.shot_z[shot_idx])**2 + (pmt.rec_x - pmt.shot_x[shot_idx])**2)
traveltimes = dist / 1500  + pmt.tlag + shift
travel_idx = traveltimes/pmt.dt   

plt.figure()
plt.plot(seismogram[:, 85], label="seismogram")
plt.plot(muted_seismogram[:, 85], label="muted")
plt.legend()

plt.figure()
plt.imshow(seismogram, aspect="auto", cmap="gray")
plt.plot(numpy.arange(pmt.Nrec), travel_idx, 'r', linewidth=2, alpha = 0.7)
plt.colorbar()
plt.show()