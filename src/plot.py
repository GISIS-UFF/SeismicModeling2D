from Viewdata import Plotting 
from AcousticSeismicModeling2D import wavefield

plt = Plotting("../inputs/Teste.json")

# plt.viewModel("marmousi_vp_")
plt.viewSnapshot("Acoustic_shot_1_Nx141_Nz141_Nt6001_")
plt.viewSeismogram("../outputs/seismograms/Acousticseismogram_shot_1_Nt6001_Nrec235.bin", perc=99)
# plt.viewSeismogramComparison(filename1, filename2)
# plt.viewMigratedImage("../outputs/migrated_image/,perc=99)