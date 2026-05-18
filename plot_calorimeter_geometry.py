#!/usr/bin/env python
"""Parse the official calorimeter geometry XML using dr_modules and plot full cross-sections."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Ensure current directory is in the path to load dr_modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from dr_modules.utils import DRgeo

def main():
    # Set premium publication-quality plotting styles
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.2
    
    xml_path = "dr_modules/DRcalo.xml"
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} not found!")
        print("Please make sure the 'dr_modules' folder is in the same directory.")
        return
        
    print("Loading official calorimeter geometry from DRcalo.xml...")
    geo = DRgeo(xml_path)
    
    # Extract dimensions
    n_phi = geo.sipmlayer.shape[0]    # 283 phi modules
    n_theta = geo.sipmlayer.shape[1]  # 92 theta modules (positive Z side)
    
    print(f"Geometry loaded successfully:")
    print(f"  - Phi towers: {n_phi}")
    print(f"  - Theta rings (positive Z): {n_theta}")
    print(f"  - Barrel: 52 slices | Endcap: 40 slices")
    
    # Create figures output directory
    os.makedirs("plots/figures", exist_ok=True)
    
    # -----------------------------------------------------------------
    # 1. Generate Transverse Cross-Section (X-Y Plane at Z = 0)
    # -----------------------------------------------------------------
    print("\nGenerating Transverse Cross-Section (X-Y Plane at Z=0)...")
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Use the central barrel ring (theta slice 0)
    theta_idx = 0
    inner_radii = []
    outer_radii = []
    
    # Plot all 283 projective towers around the circle
    for phi_idx in range(n_phi):
        # Inner face center (head)
        ix, iy, iz = geo.head[phi_idx][theta_idx]
        # Outer face center (tail/sipm layer)
        ox, oy, oz = geo.sipmlayer[phi_idx][theta_idx]
        
        # Draw the projective radial line representing the tower volume
        ax.plot([ix, ox], [iy, oy], color='#00a8ff', linewidth=1.0, alpha=0.75, zorder=3)
        
        inner_radii.append(np.sqrt(ix**2 + iy**2))
        outer_radii.append(np.sqrt(ox**2 + oy**2))
        
    mean_inner_r = np.mean(inner_radii)
    mean_outer_r = np.mean(outer_radii)
    
    # Draw perfect concentric circle boundaries representing inner and outer calorimeter envelopes
    inner_c = plt.Circle(
        (0, 0), mean_inner_r, color='#2f3640', fill=False, 
        linestyle=':', linewidth=1.5, alpha=0.8, 
        label=f'Inner Envelope (R = {mean_inner_r:.1f} mm)'
    )
    outer_c = plt.Circle(
        (0, 0), mean_outer_r, color='#2f3640', fill=False, 
        linestyle='--', linewidth=1.5, alpha=0.8, 
        label=f'Outer Envelope (R = {mean_outer_r:.1f} mm)'
    )
    ax.add_patch(inner_c)
    ax.add_patch(outer_c)
    
    # Grid and crosshairs
    ax.grid(True, linestyle="--", alpha=0.25, zorder=1)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.4, zorder=2)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.4, zorder=2)
    
    ax.set_aspect('equal')
    ax.set_title("Calorimeter Barrel Transverse Cross-Section (X-Y Plane at Z=0)\n283 Projective Towers Map", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("X coordinate [mm]", fontsize=10)
    ax.set_ylabel("Y coordinate [mm]", fontsize=10)
    
    # Custom legend
    legend_elements = [
        Line2D([0], [0], color='#00a8ff', linewidth=1.5, label='Projective Tower Axials'),
        Line2D([0], [0], color='#2f3640', linestyle=':', linewidth=1.5, label=f'Inner Radius ({mean_inner_r:.1f} mm)'),
        Line2D([0], [0], color='#2f3640', linestyle='--', linewidth=1.5, label=f'Outer Radius ({mean_outer_r:.1f} mm)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=9)
    
    # Save transverse plots
    trans_png = "plots/figures/calorimeter_transverse_section.png"
    trans_pdf = "plots/figures/calorimeter_transverse_section.pdf"
    plt.tight_layout()
    plt.savefig(trans_png, dpi=300, bbox_inches='tight')
    plt.savefig(trans_pdf, bbox_inches='tight')
    plt.close()
    
    # -----------------------------------------------------------------
    # 2. Generate Longitudinal Cross-Section (Z-X Plane)
    # -----------------------------------------------------------------
    print("Generating Longitudinal Cross-Section (Z-X Plane)...")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    
    # We choose phi slice 0 to show the full profile along Z
    phi_idx = 0
    
    # Full theta rings: -92 to 91 (reflecting negative Z side)
    theta_indices = list(range(-n_theta, 0)) + list(range(0, n_theta))
    
    for theta_idx in theta_indices:
        is_neg = theta_idx < 0
        actual_theta = abs(theta_idx) - 1 if is_neg else theta_idx
        
        # Inner face center (head)
        ix, iy, iz = geo.head[phi_idx][actual_theta]
        if is_neg: iz = -iz
        
        # Outer face center (tail)
        ox, oy, oz = geo.sipmlayer[phi_idx][actual_theta]
        if is_neg: oz = -oz
        
        # Color-code: barrel is first 52 slices, endcap is last 40 slices
        is_barrel = actual_theta < 52
        color = '#2ecc71' if is_barrel else '#e74c3c'
        alpha = 0.85 if is_barrel else 0.75
        
        # Draw the line representing the longitudinal profile of each projective tower (Z-X plane)
        ax.plot([iz, oz], [ix, ox], color=color, linewidth=1.4, alpha=alpha)
        # Mirror along the beamline to show the full circular/cylindrical profile
        ax.plot([iz, oz], [-ix, -ox], color=color, linewidth=1.4, alpha=alpha)
        
    # Grid and crosshairs
    ax.grid(True, linestyle="--", alpha=0.25, zorder=1)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4, zorder=2)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4, zorder=2)
    
    ax.set_aspect('equal')
    ax.set_title("Calorimeter Longitudinal Cross-Section (Z-X Plane Profile)\nProjective Barrel & Endcap Tower Modules", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Z coordinate (Beamline) [mm]", fontsize=10)
    ax.set_ylabel("X coordinate [mm]", fontsize=10)
    
    # Custom legend
    legend_elements_long = [
        Line2D([0], [0], color='#2ecc71', linewidth=2.0, label='Barrel Towers (52 x 2 slices)'),
        Line2D([0], [0], color='#e74c3c', linewidth=2.0, label='Endcap Towers (40 x 2 slices)')
    ]
    ax.legend(handles=legend_elements_long, loc='upper right', frameon=True, fontsize=9)
    
    # Save longitudinal plots
    long_png = "plots/figures/calorimeter_longitudinal_section.png"
    long_pdf = "plots/figures/calorimeter_longitudinal_section.pdf"
    plt.tight_layout()
    plt.savefig(long_png, dpi=300, bbox_inches='tight')
    plt.savefig(long_pdf, bbox_inches='tight')
    plt.close()
    
    print("\nSuccessfully generated all geometry plots:")
    print(f"  - Transverse Section (X-Y):")
    print(f"    * PNG: {trans_png}")
    print(f"    * PDF: {trans_pdf}")
    print(f"  - Longitudinal Section (Z-X):")
    print(f"    * PNG: {long_png}")
    print(f"    * PDF: {long_pdf}")

if __name__ == "__main__":
    main()
