from AcousticSeismicModeling2D import wavefield

wavefield = wavefield("../inputs/Teste.json")

wavefield.createSourceWavelet()
wavefield.viewSourceWavelet()

wavefield.initializeWavefields()
# wavefield.createModelFromVp()
wavefield.viewAllModels()

wavefield.checkDispersionAndStability()

wavefield.SolveWaveEquation()
# wavefield.plotEnergyComparison()
# wavefield.viewMigratedImage()


