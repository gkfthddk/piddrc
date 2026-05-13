import h5py
import numpy as np
import uproot
from pathlib import Path

h5_path = "h5s/gamma_1-120.h5py"
digi_dir = "/users/yulee/dream/tools/hdfs/gamma_1-120GeV"

with h5py.File(h5_path, "r") as f:
    best_idx = 0
    start = f["DRcalo3dHits.time"][best_idx]
    seed = f["seed"][best_idx]
    energy = f["E_gen"][best_idx]
    
    print(f"HDF5: best_idx={best_idx}, seed={seed}, energy={energy}")
    print(f"HDF5 'time' starts at: {np.min(start):.2f}, mean: {np.mean(start):.2f}")

digi_path = Path(digi_dir) / f"digi_gamma_{int(seed)}.root"
print(f"Digi path: {digi_path}")
with uproot.open(digi_path) as root_file:
    tree = root_file["events"]
    times = tree["RawCalorimeterHits/RawCalorimeterHits.timeStamp"].array(library="np")
    photons = tree["RawCalorimeterHits/RawCalorimeterHits.amplitude"].array(library="np")
    px = tree["GenParticles/GenParticles.momentum.x"].array(library="np")
    py = tree["GenParticles/GenParticles.momentum.y"].array(library="np")
    pz = tree["GenParticles/GenParticles.momentum.z"].array(library="np")
    
    for i in range(len(px)):
        en = np.sqrt(px[i][0]**2 + py[i][0]**2 + pz[i][0]**2)
        if abs(en - energy) < 1e-3:
            digi_idx = i
            print(f"Found digi event at index {digi_idx} with energy {en:.2f}")
            event_times = times[i]
            print(f"Digi 'timeStamp' starts at: {np.min(event_times):.2f}, mean: {np.mean(event_times):.2f}")
            break
