from AcousticSeismicModeling2D import wavefield

wf1 = wavefield("../inputs/parametersQSV.json")
wf2 = wavefield("../inputs/parametersQSVsolution.json")

wf1.approximation = 'acousticVTI'
wf2.approximation = 'acousticVTI'

wf1.createSourceWavelet()
wf2.createSourceWavelet()

wf1.initializeWavefields()
wf2.initializeWavefields()

wf1.viewAllModels()
wf2.viewAllModels()

wf1.checkDispersionAndStability()
wf2.checkDispersionAndStability()

wf1.SolveWaveEquation()
wf2.SolveWaveEquation()

wf1.viewSeismogram()
wf2.viewSeismogram()
wf1.viewSnapshot()
wf2.viewSnapshot()