#!/usr/bin/env python
"""Extract unique hit coordinates from HDF5 datasets and plot the exact full calorimeter cross-section."""

import argparse
import glob
import os
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np

def _resolve_paths(patterns):
    roots = [Path("."), Path("h5s"), Path("/store/ml/dual-readout/h5s")]
    paths = []
    for pattern in patterns:
        expanded = str(Path(pattern).expanduser())
        found = []
        if Path(expanded).is_absolute():
            found.extend(sorted(Path(p) for p in glob.glob(expanded)))
        else:
            for root in roots:
                found.extend(sorted(root.glob(pattern)))
        if found:
            paths.extend(found)
    return paths

def main():
    parser = argparse.ArgumentParser(description="Plot Full Calorimeter Cross-Section")
    parser.add_argument(
        "--files", 
        nargs="+", 
        default=["h5s/gamma_1-120GeV.h5py", "h5s/pi0_1-120GeV.h5py"],
        help="HDF5 file paths or globs"
    )
    parser.add_argument(
        "--scan-events", 
        type=int, 
        default=500, 
        help="Number of events to scan to capture all unique tower/channel positions"
    )
    parser.add_argument(
        "--hit-key", 
        choices=["3d", "2d"], 
        default="3d", 
        help="Whether to extract coordinates from 3D or 2D calorimeter hits"
    )
    parser.add_argument(
        "--out-dir", 
        default="plots/figures", 
        help="Directory to save output plots"
    )
    parser.add_argument(
        "--out-name", 
        default="calorimeter_cross_section", 
        help="Output file base name"
    )
    args = parser.parse_args()

    # Set publication aesthetic styles
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'serif'

    # Resolve HDF5 input paths
    files = _resolve_paths(args.files)
    if not files:
        print(f"Error: No HDF5 files found matching {args.files}!")
        print("Please check your symlink 'h5s' or provide an explicit file path.")
        return

    print(f"Resolving tower positions from {len(files)} files: {files}")
    
    unique_coords = set()
    
    # Choose position keys based on 3d/2d hits
    prefix = "DRcalo3dHits" if args.hit_key == "3d" else "DRcalo2dHits"
    x_key = f"{prefix}.position.x"
    y_key = f"{prefix}.position.y"
    z_key = f"{prefix}.position.z"
    
    # 1. Scan HDF5 files to gather unique (x, y) coordinates
    for file_path in files:
        print(f"Scanning {file_path.name}...")
        try:
            with h5py.File(file_path, "r") as f:
                if x_key not in f or y_key not in f:
                    print(f"  Warning: dataset keys {x_key} or {y_key} not found in this file. Skipping.")
                    continue
                
                x_ds = f[x_key]
                y_ds = f[y_key]
                
                n_events = x_ds.shape[0]
                scan_limit = min(n_events, args.scan_events)
                
                for idx in range(scan_limit):
                    x_arr = np.asarray(x_ds[idx])
                    y_arr = np.asarray(y_ds[idx])
                    
                    # Clean NaNs and infs
                    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
                    x_valid = x_arr[mask]
                    y_valid = y_arr[mask]
                    
                    # Store as unique rounded tuple (to eliminate precision duplicates)
                    for x, y in zip(x_valid, y_valid):
                        unique_coords.add((round(float(x), 2), round(float(y), 2)))
                        
            print(f"  Collected {len(unique_coords)} unique tower positions so far.")
        except Exception as e:
            print(f"  Error reading {file_path.name}: {e}")

    if not unique_coords:
        print("Error: Could not extract any valid coordinates!")
        return

    # Convert coordinates to array for mapping
    coords = np.array(list(unique_coords))
    xs = coords[:, 0]
    ys = coords[:, 1]
    
    # Calculate geometric statistics
    radii = np.sqrt(xs**2 + ys**2)
    min_r = np.min(radii)
    max_r = np.max(radii)
    mean_r = np.mean(radii)

    print("\nGeometry Statistics:")
    print(f"  - Total unique towers/channels found: {len(unique_coords)}")
    print(f"  - X limits: [{np.min(xs):.1f}, {np.max(xs):.1f}] mm")
    print(f"  - Y limits: [{np.min(ys):.1f}, {np.max(ys):.1f}] mm")
    print(f"  - Radial limits: Inner={min_r:.1f} mm, Outer={max_r:.1f} mm")

    # 2. Draw full calorimeter cross-section
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    
    # Plot all collected unique tower positions
    sc = ax.scatter(
        xs, ys, 
        color='#00a8ff', s=18, alpha=0.85, 
        edgecolors='#0097e6', linewidths=0.5, 
        label=f'Tower Channels ({len(unique_coords)} total)',
        zorder=3
    )
    
    # 3. Draw outer and inner radial boundary circles representing barrel geometry
    inner_circle = plt.Circle(
        (0, 0), min_r, color='#2f3640', fill=False, 
        linestyle=':', linewidth=1.2, alpha=0.6, 
        label=f'Inner Radius Boundary ({min_r:.1f} mm)'
    )
    outer_circle = plt.Circle(
        (0, 0), max_r, color='#2f3640', fill=False, 
        linestyle='--', linewidth=1.2, alpha=0.6, 
        label=f'Outer Radius Boundary ({max_r:.1f} mm)'
    )
    ax.add_patch(inner_circle)
    ax.add_patch(outer_circle)
    
    # Draw radial spokes/grids if appropriate
    ax.grid(True, linestyle="--", alpha=0.25, zorder=1)
    
    # Draw crosshairs for center alignment
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.4, zorder=2)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.4, zorder=2)

    # Plot styling
    ax.set_aspect('equal')
    ax.set_title(f"Full Calorimeter Transverse Cross-Section ({prefix} Layout)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("X position [mm]", fontsize=10)
    ax.set_ylabel("Y position [mm]", fontsize=10)
    
    ax.legend(loc='upper right', frameon=True, shadow=False, fontsize=9)
    
    # Save options
    os.makedirs(args.out_dir, exist_ok=True)
    png_path = os.path.join(args.out_dir, f"{args.out_name}.png")
    pdf_path = os.path.join(args.out_dir, f"{args.out_name}.pdf")
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"\nSuccessfully generated calorimeter cross-section plots:")
    print(f"  - PNG: {png_path}")
    print(f"  - PDF (Vector): {pdf_path}")

if __name__ == "__main__":
    main()
