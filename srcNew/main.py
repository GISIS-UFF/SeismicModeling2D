from AcousticSeismicModeling2D import wavefield
import copy


wavefield = wavefield()

wavefield.createSourceWavelet()
wavefield.viewSourceWavelet()

wavefield.initializeWavefields()
wavefield.viewAllModels()
wavefield.checkDispersionAndStability()

wavefield2 = copy.deepcopy(wavefield)

wavefield.solveAcousticWaveEquation()
wavefield2.solveAcousticVTIWaveEquation()

# if wavefield.approximation == 'acoustic':
#    wavefield.solveAcousticWaveEquation()

# elif wavefield.approximation == 'acousticVTI':
#    wavefield.solveAcousticVTIWaveEquation()

wavefield.viewSeismogram()
wavefield.viewSnapshot()

wavefield2.viewSeismogram()
wavefield2.viewSnapshot()







