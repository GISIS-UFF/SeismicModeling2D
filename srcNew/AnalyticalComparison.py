from AcousticSeismicModeling2D import wavefield
import copy

wavefield = wavefield()

wavefield.createSourceWavelet()

wavefield.initializeWavefields()
wavefield.checkDispersionAndStability()

if wavefield.approximation == 'acoustic':
   wavefield.solveAcousticWaveEquation()
   wavefield.viewSnapshotAnalyticalComparison()


elif wavefield.approximation == 'acousticVTI':
   wavefield.solveAcousticVTIWaveEquation()
   wavefield.viewSnapshotAnalyticalComparison()

wavefield.viewSeismogramComparison()
