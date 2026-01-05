from Viewdata import plotting 

plt = plotting("../inputs/Parameters.json")

# plt.viewModel("marmousi_vp_383x141.bin")
# plt.viewSnapshot("acoustic_shot_1_Nx383_Nz141_Nt4001_frame_", "../inputs/marmousi_vp_383x141.bin")
# plt.movieSnapshot("acoustic_shot_1_Nx383_Nz141_Nt4001_frame_", "../inputs/marmousi_vp_383x141.bin")
# plt.viewSeismogram("../outputs/seismograms/acoustic_seismogram_shot_1_Nt4001_Nrec501.bin", perc=95)
# plt.viewSeismogramComparison(95,0,"../outputs/seismograms/VTIseismogram_shot_1_Nt20001_Nrec501.bin", "../outputs/seismograms/VTINewseismogram_shot_1_Nt20001_Nrec501.bin")
# plt.viewMigratedImage("../outputs/migrated_image/migrated_image_acoustic_Nx383_Nz141.bin",perc=99)
# plt.viewSnapshotAnalyticalComparison(1,"../outputs/snapshots/TTI_CPML_shot_1_Nx501_Nz201_Nt6000_frame_1000.bin")