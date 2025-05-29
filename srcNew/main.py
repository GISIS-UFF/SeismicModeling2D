from AcousticSeismicModeling2D import wavefield

wavefield = wavefield()

wavefield.createSourceWavelet()
wavefield.viewSourceWavelet()

wavefield.initializeWavefields()
wavefield.viewAllModels()

wavefield.checkDispersionAndStability()

wavefield.SolveWaveEquation()

wavefield.viewSeismogram()
wavefield.viewSnapshot()


