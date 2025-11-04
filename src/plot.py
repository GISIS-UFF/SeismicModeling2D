from Viewdata import Plotting 
from AcousticSeismicModeling2D import wavefield

plt = Plotting("../inputs/Teste.json")

# plt.viewModel("layered")
# plt.viewSnapshot("Acoustic_CPML_shot_1_Nx501_Nz201_Nt6000_")
plt.viewSeismogram("../outputs/seismograms/VTICPMLseismogram_shot_1_Nt20001_Nrec383.bin", perc=99)
# plt.viewSeismogramComparison(99,5000,"../outputs/seismograms/AcousticCPMLseismogram_shot_1_Nt6000_Nrec501.bin", "../outputs/seismograms/VTICPMLseismogram_shot_1_Nt6000_Nrec501.bin","../outputs/seismograms/TTICPMLseismogram_shot_1_Nt6000_Nrec501.bin")
# plt.viewMigratedImage("../outputs/migrated_image/,perc=99)
# plt.viewSnapshotAnalyticalComparison(1,"../outputs/snapshots/TTI_CPML_shot_1_Nx501_Nz201_Nt6000_frame_1000.bin")