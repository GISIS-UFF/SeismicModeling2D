from utils import Mute
from survey import parameters
import matplotlib.pyplot as plt
import numpy 

pmt = parameters("../inputs/Parameters.json")
shot = 20
shift = 0.15
window = 0.05
seismogramFile = f"{pmt.seismogramFolder}seismogram_shot_{shot}_Nt{pmt.nt}_Nrec{pmt.Nrec}.bin"
seismogram = numpy.fromfile(seismogramFile, dtype=numpy.float32).reshape(pmt.nt,pmt.Nrec) 
muted_seismogram = Mute(seismogram, shot, pmt.rec_x, pmt.rec_z, pmt.shot_x, pmt.shot_z, pmt.dt,pmt.tlag,shift,window,v0=1500)

dist = numpy.sqrt((pmt.rec_z - pmt.shot_z[shot])**2 + (pmt.rec_x - pmt.shot_x[shot])**2)
traveltimes = dist / 1500 + pmt.tlag 
travel_idx = traveltimes / pmt.dt   

plt.figure()
plt.plot(seismogram[:, 160], label="seismogram")
plt.plot(muted_seismogram[:, 160], label="muted")
plt.legend()

plt.figure()
plt.imshow(muted_seismogram, aspect="auto", cmap="gray")
# plt.plot(numpy.arange(pmt.Nrec), travel_idx, 'r', linewidth=2, alpha = 0.7)
plt.colorbar()
plt.show()