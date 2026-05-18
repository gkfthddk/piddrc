#!/usr/bin/env python
"""Create event displays for Sim3dCalorimeterHits.

This script visualizes the truth-level shower structure for gamma and pi0 events.
Since Sim3d hits typically lack timing in this project's HDF5 schema, 
points are colored by energy deposition.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def set_style():
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "figure.titlesize": 18,
        "savefig.dpi": 200,
    })

def _resolve_paths(patterns: Iterable[str]) -> list[Path]:
    roots = [Path("."), Path("h5s"), Path("/store/ml/dual-readout/h5s")]
    paths: list[Path] = []
    for pattern in patterns:
        expanded = str(Path(pattern).expanduser())
        found: list[Path] = []
        if Path(expanded).is_absolute():
            found.extend(sorted(Path(p) for p in glob.glob(expanded)))
        else:
            for root in roots:
                found.extend(sorted(root.glob(pattern)))
        if found:
            paths.extend(found)
    return paths

def main():
    parser = argparse.ArgumentParser(description="Sim3d Event Display")
    parser.add_argument("--files", nargs="+", default=["h5s/gamma_1-120GeV.h5py", "h5s/pi0_1-120GeV.h5py"])
    parser.add_argument("--event-count", type=int, default=20)
    parser.add_argument("--out-dir", default="plots/event_display")
    parser.add_argument("--min-theta", type=float, default=1.5)
    parser.add_argument("--min-energy", type=float, default=30.0)
    parser.add_argument("--max-energy", type=float, default=70.0)
    parser.add_argument("--score-file", default="save/split_mamba_40k_rw005_lowlr_dw005/output.json")
    parser.add_argument("--config-file", default="save/split_mamba_40k_rw005_lowlr_dw005/config.json")
    args = parser.parse_args()

    set_style()
    files = _resolve_paths(args.files)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import json
    # Load config to get file indices
    file_id_map = {}
    if Path(args.config_file).exists():
        try:
            with open(args.config_file, "r") as f_cfg:
                cfg = json.load(f_cfg)
                train_files = cfg.get("train_files", [])
                for i, f_name in enumerate(train_files):
                    # Use basename for matching
                    file_id_map[Path(f_name).name] = i
            print(f"Loaded file ID map from config: {file_id_map}")
        except Exception as e:
            print(f"Error loading config file: {e}")

    # Load scores and find valid events from JSON
    # Structure: valid_events_per_file[file_id] = [internal_idx1, internal_idx2, ...]
    valid_events_per_file = {}
    event_probs = {}
    
    if Path(args.score_file).exists():
        try:
            with open(args.score_file, "r") as f_score:
                data = json.load(f_score)
                for entry in data:
                    ev_id = entry.get("event_id") # [file_id, internal_idx]
                    if ev_id is None or len(ev_id) != 2: continue
                    
                    fid, idx_in_file = ev_id[0], ev_id[1]
                    theta = entry.get("theta")
                    energy = entry.get("E_gen")
                    logits = entry.get("logits")
                    label_idx = entry.get("label")
                    
                    # Apply physics filters from JSON data
                    theta_ok = (theta is not None and theta >= args.min_theta)
                    energy_ok = (energy is not None and args.min_energy <= energy <= args.max_energy)
                    
                    if theta_ok and energy_ok:
                        if fid not in valid_events_per_file:
                            valid_events_per_file[fid] = []
                        valid_events_per_file[fid].append(idx_in_file)
                        
                        # Store ALL probabilities
                        if logits is not None:
                            exps = np.exp(np.array(logits) - np.max(logits))
                            probs = exps / np.sum(exps)
                            event_probs[tuple(ev_id)] = probs
                            
            print(f"Found total {sum(len(v) for v in valid_events_per_file.values())} valid events in JSON.")
        except Exception as e:
            print(f"Error loading score file: {e}")
    else:
        print(f"Score file {args.score_file} not found. CANNOT proceed with JSON-first filtering.")
        return

    # Process each requested file
    for file_path in files:
        # Determine the file ID from config
        curr_file_id = file_id_map.get(file_path.name)
        if curr_file_id is None:
            print(f"Warning: {file_path.name} not in config. Skipping.")
            continue
            
        # Get valid indices for THIS specific file
        target_indices = valid_events_per_file.get(curr_file_id, [])
        if not target_indices:
            print(f"No valid events found in JSON for {file_path.name}.")
            continue
            
        # Limit to requested event count
        target_indices = target_indices[:args.event_count]
        print(f"Generating {len(target_indices)} plots for {file_path.name} (File ID: {curr_file_id})...")
        
        label = "gamma" if "gamma" in file_path.name else "pi0"
        with h5py.File(file_path, "r") as f:
            # Metadata helper
            def get_scalar_val(key, e_idx):
                if key not in f: return None
                val = np.asarray(f[key][e_idx]).reshape(-1)
                return val[0] if val.size > 0 else None

            for idx in target_indices:
                # --- Load Sim3d (Truth) ---
                # --- Load Sim3d (Truth) ---
                sx = f["Sim3dCalorimeterHits.position.x"][idx]
                sy = f["Sim3dCalorimeterHits.position.y"][idx]
                sz = f["Sim3dCalorimeterHits.position.z"][idx]
                se = f["Sim3dCalorimeterHits.energy"][idx]
                
                s_mask = np.isfinite(sx) & (se > 0)
                sx, sy, sz, se = sx[s_mask], sy[s_mask], sz[s_mask], se[s_mask]
                
                # Filter Sim3d (top 99%)
                if se.size > 0:
                    s_threshold = sum(se) * 0.005
                    s_e_mask = se > s_threshold
                    sx, sy, sz, se = sx[s_e_mask], sy[s_e_mask], sz[s_e_mask], se[s_e_mask]

                # --- Load DRcalo (Reco) with Timing and Type ---
                dx = f["DRcalo3dHits.position.x"][idx]
                dy = f["DRcalo3dHits.position.y"][idx]
                dz = f["DRcalo3dHits.position.z"][idx]
                da = f["DRcalo3dHits.amplitude_sum"][idx]
                dt = f["DRcalo3dHits.time"][idx]
                dte = f["DRcalo3dHits.time_end"][idx] if "DRcalo3dHits.time_end" in f else dt
                dtype = f["DRcalo3dHits.type"][idx] if "DRcalo3dHits.type" in f else np.zeros_like(da)
                
                # Filter by finite values, positive amplitude, and type == False (typically Scintillation)
                d_mask = np.isfinite(dx) & (da > 0) & (dtype.astype(bool) == False)
                dx, dy, dz, da, dt, dte = dx[d_mask], dy[d_mask], dz[d_mask], da[d_mask], dt[d_mask], dte[d_mask]
                d_time_mid = 0.5 * (dt + dte)

                # Filter DRcalo (top 99.9% / 0.1% threshold)
                if da.size > 0:
                    d_threshold = sum(da) * 0.00025
                    d_a_mask = da > d_threshold
                    dx, dy, dz, da, d_time_mid = dx[d_a_mask], dy[d_a_mask], dz[d_a_mask], da[d_a_mask], d_time_mid[d_a_mask]

                # Sort Sim3d by energy to draw large hits on top
                if se.size > 0:
                    sort_idx_s = np.argsort(se)
                    sx, sy, sz, se = sx[sort_idx_s], sy[sort_idx_s], sz[sort_idx_s], se[sort_idx_s]
                
                # Sort DRcalo by amplitude to draw large hits on top
                if da.size > 0:
                    sort_idx_d = np.argsort(da)
                    dx, dy, dz, da, d_time_mid = dx[sort_idx_d], dy[sort_idx_d], dz[sort_idx_d], da[sort_idx_d], d_time_mid[sort_idx_d]

                # Calculate Energy Weighted Centers (Centroids)
                def get_centroid(h, v, w):
                    if w.size == 0 or np.sum(w) == 0: return 0, 0
                    return np.sum(h * w) / np.sum(w), np.sum(v * w) / np.sum(w)

                def get_centroid_3d(z_arr, y_arr, x_arr, w_arr):
                    if w_arr.size == 0 or np.sum(w_arr) == 0: return 0, 0, 0
                    return (np.sum(z_arr * w_arr) / np.sum(w_arr), 
                            np.sum(y_arr * w_arr) / np.sum(w_arr), 
                            np.sum(x_arr * w_arr) / np.sum(w_arr))

                sc_sz, sc_sy, sc_sx = get_centroid_3d(sz, sy, sx, se)
                dc_dz, dc_dy = get_centroid(dz, dy, da)

                # --- Enhanced Multi-panel Plotting (2x2 Grid) ---
                fig = plt.figure(figsize=(18, 16))
                gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)
                
                # Normalization
                s_norm = matplotlib.colors.LogNorm(vmin=max(se.min(), 1e-3), vmax=se.max()) if se.size > 0 else None
                if d_time_mid.size > 0:
                    t_min, t_max = np.percentile(d_time_mid, [5, 95])
                    if t_max <= t_min: t_max = t_min + 1.0
                    d_time_norm = matplotlib.colors.Normalize(vmin=t_min, vmax=t_max)
                else:
                    d_time_norm = None

                # Calculate Rank-based sizes (Robust to outliers)
                def get_rank_sizes(arr, min_s, max_s):
                    if arr.size == 0: return np.array([])
                    if arr.size == 1: return np.array([max_s])
                    ranks = np.argsort(np.argsort(arr))
                    return min_s + (max_s - min_s) * (ranks / (arr.size - 1))

                s_sizes_3d = get_rank_sizes(se, 1, 20)
                s_sizes_2d = get_rank_sizes(se, 1, 30)
                d_sizes_2d = get_rank_sizes(da, 1, 50)

                # --- (0,0) Panel 1: 3D Plot View 1 (Isometric) ---
                ax3d_1 = fig.add_subplot(gs[0, 0], projection='3d')
                if se.size > 0:
                    sc3d_1 = ax3d_1.scatter(sz, sy, sx, c=se, s=s_sizes_3d, cmap="magma", norm=s_norm, alpha=0.7, edgecolors='none', marker='o')
                ax3d_1.set_xlabel("z [mm]", labelpad=5)
                ax3d_1.set_ylabel("y [mm]", labelpad=5)
                ax3d_1.set_zlabel("x [mm]", labelpad=5)
                ax3d_1.set_title("3D View (Isometric)", pad=10)
                ax3d_1.view_init(elev=20, azim=-60)

                # --- (0,1) Panel 2: 3D Plot View 2 (Side View) ---
                ax3d_2 = fig.add_subplot(gs[0, 1], projection='3d')
                if se.size > 0:
                    sc3d_2 = ax3d_2.scatter(sz, sy, sx, c=se, s=s_sizes_3d, cmap="magma", norm=s_norm, alpha=0.7, edgecolors='none', marker='o')
                ax3d_2.set_xlabel("z [mm]", labelpad=5)
                ax3d_2.set_ylabel("y [mm]", labelpad=5)
                ax3d_2.set_zlabel("x [mm]", labelpad=5)
                ax3d_2.set_title("3D View (Side/Longitudinal)", pad=10)
                ax3d_2.view_init(elev=0, azim=-90)

                # --- (1,0) Panel 3: Z-Y Projection (Sim3d Truth) ---
                ax_zy_s = fig.add_subplot(gs[1, 0])
                if se.size > 0:
                    ax_zy_s.scatter(sz, sy, c=se, s=s_sizes_2d, cmap="magma", norm=s_norm, alpha=0.6, edgecolors='none', marker='o')
                ax_zy_s.set_xlabel("z [mm] (Beamline)")
                ax_zy_s.set_ylabel("y [mm] (Depth)")
                ax_zy_s.set_title("ZY Projection (Sim3d Truth - Energy)")
                ax_zy_s.grid(True, linestyle="--", alpha=0.3)

                # --- (1,1) Panel 4: Z-Y Projection (DRcalo Reco) ---
                ax_zy_d = fig.add_subplot(gs[1, 1])
                if da.size > 0:
                    sc_d = ax_zy_d.scatter(dz, dy, c=d_time_mid, s=d_sizes_2d, cmap="coolwarm", norm=d_time_norm, alpha=0.7, edgecolors='none', marker='o')
                    
                    # Highlight peak
                    max_idx = np.argmax(da)
                    ax_zy_d.scatter(dz[max_idx], dy[max_idx], c=[d_time_mid[max_idx]], s=d_sizes_2d[max_idx], 
                                    cmap="coolwarm", norm=d_time_norm, edgecolors='white', linewidths=1.5, marker='o', zorder=10)
                    
                    cbar_d = fig.colorbar(sc_d, ax=ax_zy_d, label="Median Time [ns]", fraction=0.046, pad=0.04)
                    
                    # Add Quartile Size Legend
                    for perc, _label in [(1.0, "Top 1% (Max)"), (0.75, "Top 25% (Q3)"), (0.5, "Top 50% (Med)"), (0.25, "Top 75% (Q1)")]:
                        size = 5 + (100 - 5) * perc
                        ax_zy_d.scatter([], [], c='gray', s=size, alpha=0.5, label=_label)
                    ax_zy_d.legend(title="Amplitude Quartiles", loc='lower right', fontsize='small', frameon=True, framealpha=0.8)
                    
                ax_zy_d.set_xlabel("z [mm] (Beamline)")
                ax_zy_d.set_ylabel("y [mm] (Depth)")
                ax_zy_d.set_title(f"DRcalo Reco (Max Amp: {da.max():.1f})")
                ax_zy_d.grid(True, linestyle="--", alpha=0.3)

                # Shared Colorbar for Sim3d Energy
                cbar_s_ax = fig.add_axes([0.48, 0.58, 0.015, 0.35])
                fig.colorbar(sc3d_1, cax=cbar_s_ax, label="Sim3d Energy [MeV]")

                # Metadata
                pdg = get_scalar_val("GenParticles.PDG", idx)
                energy_gen = get_scalar_val("E_gen", idx)
                score_key = (curr_file_id, idx)
                probs = event_probs.get(score_key) # Array of 4 probs
                if probs is not None:
                    p_e, p_g, p_p0, p_pp = probs
                    score_str = f"\nPred: [e:{p_e:.2f}, g:{p_g:.2f}, p0:{p_p0:.2f}, p+:{p_pp:.2f}]"
                else:
                    score_str = ""
                
                fig.suptitle(f"Event {idx} | {label} | PDG: {pdg} | E_gen: {energy_gen:.2f} GeV{score_str}", fontsize=22, y=0.96)
                
                # Centering and Equal aspect ratio
                def set_equal_3d_weighted(ax, mz, my, mx, fixed_range_zy=None, fixed_range_x=250):
                    if fixed_range_zy is None:
                        rz = ry = 250
                    else:
                        rz = ry = fixed_range_zy
                    
                    ax.set_xlim(mz - rz, mz + rz)
                    ax.set_ylim(my - ry, my + ry)
                    ax.set_zlim(mx - fixed_range_x, mx + fixed_range_x)

                def set_equal_2d_weighted(ax, mh, mv, fixed_range=250):
                    ax.set_xlim(mh - fixed_range, mh + fixed_range)
                    ax.set_ylim(mv - fixed_range, mv + fixed_range)
                    ax.set_aspect('equal')

                # Set 3D plots Z, Y range to 50mm (±25mm)
                set_equal_3d_weighted(ax3d_1, sc_sz, sc_sy, sc_sx, fixed_range_zy=25)
                set_equal_3d_weighted(ax3d_2, sc_sz, sc_sy, sc_sx, fixed_range_zy=25)
                # Extreme Close-up of Sim3d Truth (Bottom Left) to 50mm total (±25mm)
                set_equal_2d_weighted(ax_zy_s, sc_sz, sc_sy, fixed_range=25)
                # Keep DRcalo Reco (Bottom Right) at 500mm (±250mm)
                set_equal_2d_weighted(ax_zy_d, dc_dz, dc_dy, fixed_range=250)
                
                out_path = out_dir / f"event_display_{label}_idx{idx}.png"
                plt.savefig(out_path, bbox_inches='tight')
                plt.close()
                print(f"  - Wrote {out_path}")

if __name__ == "__main__":
    main()
