from AcousticSeismicModeling2D import wavefield


wavefield = wavefield()

wavefield.createSourceWavelet()
wavefield.viewSourceWavelet()

wavefield.initializeWavefields()
wavefield.viewAllModels()

wavefield.checkDispersionAndStability()

if wavefield.approximation == 'acoustic':
   wavefield.solveAcousticWaveEquation()

elif wavefield.approximation == 'acousticVTI':
   wavefield.solveAcousticVTIWaveEquation()

wavefield.viewSeismogram()
wavefield.viewSnapshot(0)
wavefield.viewSnapshot(1)






