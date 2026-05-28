from survey import parameters
from Modeling2D import wavefield
from Migration import migration
from Inversion import fwi

pmt = parameters("../inputs/Parameters.json")

for fmax in pmt.freqs:
    print(f"\033[31minfo: FWI frequency {fmax} of {pmt.freqs}\033[0m")

    pmt.fcut = fmax

    wf = wavefield(pmt)
    wf.createSourceWavelet()
    wf.initializeWavefields()
    wf.loadModels()
    wf.checkDispersionAndStability()
    wf.SolveWaveEquation()

    mig = migration(wf,pmt)
    mig.initializeMigrationfields()

    inv = fwi(pmt, wf, mig)

    inv.solveFullWaveformInversion(fmax)