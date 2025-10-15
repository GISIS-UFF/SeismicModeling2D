from Viewdata import Plotting 
from AcousticSeismicModeling2D import wavefield

plt = Plotting("../inputs/Teste.json")

# plt.viewModel("marmousi_vp_")
plt.viewSnapshot("TTI_CPML_shot_1_Nx141_Nz141")
plt.viewSeismogram("../outputs/seismograms/TTICPMLseismogram_shot_1_Nt4001_Nrec184.bin", perc=95)
# plt.viewSeismogramComparison(filename1, filename2)
# plt.viewMigratedImage("../outputs/migrated_image/,perc=99)