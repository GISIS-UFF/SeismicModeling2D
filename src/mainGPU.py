from GPUAcousticSeismicModeling2D import wavefield_GPU

wavefieldGPU = wavefield_GPU("../inputs/Parameters.json")

wavefieldGPU.createSourceWavelet()
wavefieldGPU.initializeWavefields()
wavefieldGPU.checkDispersionAndStability()
wavefieldGPU.SolveWaveEquation()


