from survey import parameters
from Modeling2D import wavefield
from Migration import migration

pmt = parameters("../inputs/Parameters.json")

wf = wavefield(pmt)
wf.createSourceWavelet()
wf.initializeWavefields()
wf.loadModels()

mig = migration(wf,pmt)
mig.SolveBackwardWaveEquation()       
