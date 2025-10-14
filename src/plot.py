from Viewdata import Plotting 
from AcousticSeismicModeling2D import wavefield


wf = wavefield("../inputs/Teste.json")
plt = Plotting(wf)

# # plt.viewModel()
# plt.viewSnapshot()
plt.viewSeismogram(perc=95)
# plt.viewMigratedImage(perc=99)