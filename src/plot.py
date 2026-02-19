from Viewdata import plotting 
from survey import parameters

pmt = parameters("../inputs/Parameters.json")
plt = plotting(pmt)

# plt.viewModel("layer2vp_Nz301_Nx301.bin")
# plt.viewSnapshot("acousticcerjan_shot_1_Nx301_Nz101_Nt4001_frame_1200.bin", "../inputs/layeredvp_Nz101_Nx301.bin")
# plt.movieSnapshot("acoustic_shot_1_Nx301_Nz301_Nt6001_frame_", "../inputs/layer2vp_Nz301_Nx301.bin", savegif = False)
# plt.viewSeismogram("../outputs/seismograms/seismogram_shot_1_Nt6001_Nrec501.bin", perc=99)
# plt.viewSeismogramComparison(95,0,"../outputs/seismograms/VTIseismogram_shot_1_Nt20001_Nrec501.bin", "../outputs/seismograms/VTINewseismogram_shot_1_Nt20001_Nrec501.bin")
plt.viewMigratedImage("../outputs/migrated_image/migrated_image_acoustic_Nx301_Nz101.bin",perc=99)

