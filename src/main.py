from survey import parameters
from Modeling2D import wavefield

pmt = parameters("../inputs/Parameters.json")

wavefield = wavefield(pmt)

wavefield.createSourceWavelet()
wavefield.initializeWavefields()
wavefield.loadModels()
wavefield.checkDispersionAndStability()
wavefield.solveWaveEquation()



