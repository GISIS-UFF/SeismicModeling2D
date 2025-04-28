from AcousticSeismicModeling2D import wavefield
import copy

wavefield = wavefield()
wavefield.approximation = 'acoustic'

wavefield.createSourceWavelet()

wavefield2 = copy.deepcopy(wavefield)
wavefield2.approximation = 'acousticVTI'

wavefield.initializeWavefields()
wavefield2.initializeWavefields()

wavefield.viewAllModels()
wavefield2.viewAllModels()

wavefield.checkDispersionAndStability()
wavefield2.checkDispersionAndStability()

wavefield.SolveWaveEquation()
wavefield.viewSnapshotAnalyticalComparison()

wavefield2.SolveWaveEquation()
wavefield2.viewSnapshotAnalyticalComparison()

wavefield.viewSeismogramComparison()
