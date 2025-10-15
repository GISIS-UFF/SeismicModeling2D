from AcousticSeismicModeling2D import wavefield

wavefield = wavefield("../inputs/Teste.json")

wavefield.createSourceWavelet()
wavefield.initializeWavefields()
wavefield.checkDispersionAndStability()
wavefield.SolveWaveEquation()



