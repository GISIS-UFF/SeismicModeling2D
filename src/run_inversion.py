from survey import parameters
from Modeling2D import wavefield
from Migration import migration
from Inversion import fwi

pmt = parameters("../inputs/Parameters.json")

wf = wavefield(pmt)
wf.initializeWavefields()
wf.loadModels()

mig = migration(wf,pmt)
mig.initializeMigrationfields()

inv = fwi(pmt, wf, mig)

inv.solveFullWaveformInversion()