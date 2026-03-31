from survey import parameters
from Modeling2D import wavefield
from Migration import migration
from Inversion import fwi

pmt = parameters("../inputs/Parameters.json")

wf = wavefield(pmt)
wf.createSourceWavelet()
wf.initializeWavefields()
wf.loadModels()

mig = migration(wf,pmt)

fwi = fwi(pmt, wf, mig)

fwi.solveFullWaveformInversion()
