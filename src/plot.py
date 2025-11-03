from Viewdata import Plotting 
from AcousticSeismicModeling2D import wavefield

plt = Plotting("../inputs/Teste.json")

# plt.viewModel("marmousi_vp_")
plt.viewSnapshot("TTI_CPML_shot_1_Nx141_Nz141_Nt6000_")
plt.viewSeismogram("../outputs/seismograms/TTICPMLseismogram_shot_1_Nt6000_Nrec235.bin", perc=95)
# plt.viewSeismogramComparison("../outputs/seismograms/AcousticCPMLseismogram_shot_1_Nt6001_Nrec235.bin", "../outputs/seismograms/Acousticseismogram_shot_1_Nt6001_Nrec235.bin")
# plt.viewMigratedImage("../outputs/migrated_image/,perc=99)