from AcousticSeismicModeling2D import wavefield
import copy

# Compare the analytical solution of the acoustic wave equation with the numerical solution
wf1 = wavefield("../inputs/parametersAnalytical.json")
wf1.approximation = 'acoustic'
wf1.createSourceWavelet()

wf2 = copy.deepcopy(wf1)
wf2.approximation = 'acousticVTI'
wf1.initializeWavefields()
wf2.initializeWavefields()

wf1.viewAllModels()
wf2.viewAllModels()

wf1.checkDispersionAndStability()
wf2.checkDispersionAndStability()

wf1.SolveWaveEquation()
wf2.SolveWaveEquation()

wf1.viewSnapshotAnalyticalComparison()
wf2.viewSnapshotAnalyticalComparison()

# Compare the acoustic and acousticVTI sismograms
wf3 = wavefield()
wf3.approximation = 'acoustic'
wf3.createSourceWavelet()

wf4 = copy.deepcopy(wf3)
wf4.approximation = 'acousticVTI'

wf3.initializeWavefields()
wf4.initializeWavefields()

wf3.viewAllModels()
wf4.viewAllModels()

wf3.checkDispersionAndStability()
wf4.checkDispersionAndStability()

wf3.SolveWaveEquation()
wf4.SolveWaveEquation()

wf3.viewSeismogramComparison()
