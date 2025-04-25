from AcousticSeismicModeling2D import wavefield


wavefield = wavefield()

wavefield.initializeWavefields()
wavefield.viewAllModels()

wavefield.createVTIModelFromVp()