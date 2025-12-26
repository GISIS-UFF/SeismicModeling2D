from Modeling2D import wavefield

wavefield = wavefield("../inputs/Parameters.json")

wavefield.createSourceWavelet()
wavefield.initializeWavefields()
wavefield.checkDispersionAndStability()
wavefield.SolveWaveEquation()



