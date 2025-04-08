from AcousticSeismicModeling2D import wavefield


wavefield = wavefield()

wavefield.createSourceWavelet()
wavefield.viewSourceWavelet()

wavefield.initializeWavefields()
wavefield.viewVelocityModel()

wavefield.checkDispersionAndStability()

wavefield.solveWaveEquation()

wavefield.viewSeismogram()
wavefield.viewSnapshot(0)
wavefield.viewSnapshot(1)






