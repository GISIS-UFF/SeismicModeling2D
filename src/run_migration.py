from Modeling2D import wavefield
from Migration import migration

wf = wavefield("../inputs/Parameters.json")
wf.createSourceWavelet()
wf.initializeWavefields()

mig = migration("../inputs/Parameters.json",wf)
mig.SolveBackwardWaveEquation()       
