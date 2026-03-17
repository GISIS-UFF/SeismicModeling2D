from Viewdata import plotting 
from survey import parameters

pmt = parameters("../inputs/Parameters.json")
plt = plotting(pmt)

# plt.viewModel(f"layer2")
# plt.viewSnapshot(f"../inputs/layer2vp_Nz301_Nx301.bin",f"{pmt.approximation}backward_shot_1_Nx{pmt.nx}_Nz{pmt.nz}_Nt{pmt.nt}_frame_")
# plt.viewSnapshot(f"../inputs/layer2vp_Nz301_Nx301.bin",F"{pmt.approximation}backward_shot_1_Nx{pmt.nx}_Nz{pmt.nz}_Nt{pmt.nt}_frame_2500.bin")
# plt.movieSnapshot(f"{pmt.approximation}backward_shot_1_Nx{pmt.nx}_Nz{pmt.nz}_Nt{pmt.nt}_frame_", f"../inputs/layer2vp_Nz301_Nx301.bin", savegif = False)
# plt.viewSeismogram(f"{pmt.seismogramFolder}seismogram_shot_1_Nt{pmt.nt}_Nrec{pmt.Nrec}.bin", perc=96)
# plt.viewSeismogramComparison(95,0,"../outputs/seismograms/VTIseismogram_shot_1_Nt20001_Nrec501.bin", "../outputs/seismograms/VTINewseismogram_shot_1_Nt20001_Nrec501.bin")
plt.viewImage(f"{pmt.migratedimageFolder}migrated_image_{pmt.approximation}_Nx{pmt.nx}_Nz{pmt.nz}.bin",laplacian=True,perc=99)
# plt.plotImageTrace(f"{pmt.migratedimageFolder}migrated_image_{pmt.approximation}_Nx{pmt.nx}_Nz{pmt.nz}.bin", f"../inputs/layer2vp_Nz{pmt.nz}_Nx{pmt.nx}.bin", laplacian = True, ix=None, perc=99)
# plt.movieImage(f"{pmt.approximation}_shot_1_Nx{pmt.nx}_Nz{pmt.nz}_frame_", f"../inputs/layer2vp_Nz301_Nx301.bin",laplacian = True, interval=200, savegif = False)


