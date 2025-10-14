from AcousticSeismicModeling2D import wavefield

wavefield = wavefield("../inputs/Teste.json")

wavefield.createSourceWavelet()
wavefield.initializeWavefields()
# wavefield.createModelFromVp()
wavefield.checkDispersionAndStability()
wavefield.SolveWaveEquation()



