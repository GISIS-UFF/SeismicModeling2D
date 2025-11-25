from AcousticSeismicModeling2D import wavefield


wavefield = wavefield("../inputs/Teste.json")

wavefield.initializeWavefields()
wavefield.createModelFromVp()