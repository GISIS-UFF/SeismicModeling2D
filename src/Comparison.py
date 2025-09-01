from AcousticSeismicModeling2D import wavefield
import copy

# Compare the analytical solution of the acoustic wave equation with the numerical solution
wf1 = wavefield("../inputs/parametersAnalytical.json")
wf1.approximation = 'acoustic'
wf1.createSourceWavelet()

wf2 = copy.deepcopy(wf1)
wf2.approximation = 'acousticVTI'
wf0 = copy.deepcopy(wf1)
wf0.approximation = 'acousticTTI'

wf1.initializeWavefields()
wf2.initializeWavefields()
wf0.initializeWavefields()

wf1.viewAllModels()
wf2.viewAllModels()
wf0.viewAllModels()

wf1.checkDispersionAndStability()
wf2.checkDispersionAndStability()
wf0.checkDispersionAndStability()

wf1.SolveWaveEquation()
wf2.SolveWaveEquation()
wf0.SolveWaveEquation()

wf1.viewSnapshotAnalyticalComparison()
wf2.viewSnapshotAnalyticalComparison()
wf0.viewSnapshotAnalyticalComparison()

# Compare the acoustic and acousticVTI sismograms
wf3 = wavefield("../inputs/parametersMarmousi.json")
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

wf3.viewSeismogram()
wf4.viewSeismogram()

wf3.viewSeismogramComparison("../outputs/seismograms/Acousticseismogram_shot_1_Nt4001_Nrec383.bin", 
                             "../outputs/seismograms/VTIseismogram_shot_1_Nt4001_Nrec383.bin",title="Acoustic - VTI")

# Compare the acousticVTI and acousticTTI sismograms

wf5 = wavefield("../inputs/parametersTTIComparison.json")
wf5.createSourceWavelet()
wf6 = copy.deepcopy(wf5)
wf6.approximation = 'acousticVTI'

wf5.initializeWavefields()
wf6.initializeWavefields()

wf5.viewAllModels()
wf6.viewAllModels()

wf5.checkDispersionAndStability()
wf6.checkDispersionAndStability()

wf5.SolveWaveEquation()
wf6.SolveWaveEquation()

wf5.viewSeismogram()
wf6.viewSeismogram()

wf5.viewSeismogramComparison("../outputs/seismograms/TTIseismogram_shot_1_Nt4001_Nrec383.bin",
                             "../outputs/seismograms/VTIseismogram_shot_1_Nt4001_Nrec383.bin", title="TTI - VTI")    
