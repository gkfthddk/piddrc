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
    
    import matplotlib.patches as patches
    
    # Use the central barrel ring (theta slice 0)
    theta_idx = 0
    inner_radii = []
    outer_radii = []
    
    # Calculate constant inner and outer radius first
    for phi_idx in range(n_phi):
        ix, iy, iz = geo.head[phi_idx][theta_idx]
        ox, oy, oz = geo.sipmlayer[phi_idx][theta_idx]
        inner_radii.append(np.sqrt(ix**2 + iy**2))
        outer_radii.append(np.sqrt(ox**2 + oy**2))
        
    mean_inner_r = np.mean(inner_radii)
    mean_outer_r = np.mean(outer_radii)
    
    # Step size for 283 slices in radians
    dphi = 2 * np.pi / 283
    
    # Draw all 283 projective towers as solid quadrilateral slices
    for phi_idx in range(n_phi):
        phi_center = phi_idx * dphi
        phi_start = phi_center - dphi / 2
        phi_end = phi_center + dphi / 2
        
        # Corner 1: Inner start
        x1 = mean_inner_r * np.cos(phi_start)
        y1 = mean_inner_r * np.sin(phi_start)
        
        # Corner 2: Inner end
        x2 = mean_inner_r * np.cos(phi_end)
        y2 = mean_inner_r * np.sin(phi_end)
        
        # Corner 3: Outer end
        x3 = mean_outer_r * np.cos(phi_end)
        y3 = mean_outer_r * np.sin(phi_end)
        
        # Corner 4: Outer start
        x4 = mean_outer_r * np.cos(phi_start)
        y4 = mean_outer_r * np.sin(phi_start)
        
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        poly = patches.Polygon(pts, closed=True, edgecolor='#2c3e50', facecolor='#3498db', linewidth=0.4, alpha=0.6, zorder=3)
        ax.add_patch(poly)
        
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
    
    # Explicitly set limits to frame the full circular calorimeter ring (radii 1800 to 3800 mm)
    ax.set_xlim(-4200, 4200)
    ax.set_ylim(-4200, 4200)
    ax.set_aspect('equal')
    
    ax.set_title("Calorimeter Barrel Transverse Cross-Section (X-Y Plane at Z=0)\n283 Projective Towers Map", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("X coordinate [mm]", fontsize=10)
    ax.set_ylabel("Y coordinate [mm]", fontsize=10)
    
    # Custom legend
    legend_elements = [
        Line2D([0], [0], color='#3498db', linewidth=6.0, alpha=0.6, label='Projective Barrel Towers (283 wedges)'),
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
    # 2. Generate Longitudinal Cross-Section (Z-X Plane, First Quadrant)
    # -----------------------------------------------------------------
    print("Generating Longitudinal Cross-Section (Z-X Plane, First Quadrant)...")
    fig, ax = plt.subplots(figsize=(9, 8))
    
    import xml.etree.ElementTree as ET
    import matplotlib.patches as patches
    
    # Parse XML directly to compute exact tower boundaries
    root = ET.parse(xml_path).getroot()
    
    for component in ["barrel", "endcap"]:
        detector = f'detectors/detector/{component}'
        height = float(root.find(detector).get("height").split("*")[0]) * 1000
        rmin = float(root.find(detector).get("rmin").split("*")[0]) * 1000
        fInnerX = rmin
        currentTheta = float(root.find(detector).get("theta"))
        
        # Color-code based on component
        color = '#2ecc71' if component == "barrel" else '#e74c3c'
        alpha = 0.65 if component == "barrel" else 0.55  # slightly transparent fill to show overlap/grid
        
        for line in root.find(detector):
            deltatheta = float(line.get('deltatheta'))
            theta_start = currentTheta
            theta_end = currentTheta + deltatheta
            currentToC = currentTheta + deltatheta / 2.
            currentTheta += deltatheta
            
            if component == "barrel":
                fCurrentInnerR = fInnerX / np.cos(currentToC)
            else:
                fCurrentInnerR = fInnerX / np.sin(currentToC)
                
            r_inner = fCurrentInnerR
            r_outer = fCurrentInnerR + height
            
            # Corner 1: Inner start
            z1 = r_inner * np.sin(theta_start)
            x1 = r_inner * np.cos(theta_start)
            
            # Corner 2: Inner end
            z2 = r_inner * np.sin(theta_end)
            x2 = r_inner * np.cos(theta_end)
            
            # Corner 3: Outer end
            z3 = r_outer * np.sin(theta_end)
            x3 = r_outer * np.cos(theta_end)
            
            # Corner 4: Outer start
            z4 = r_outer * np.sin(theta_start)
            x4 = r_outer * np.cos(theta_start)
            
            # Create and draw the quadrilateral polygon representing the tower profile
            pts = np.array([[z1, x1], [z2, x2], [z3, x3], [z4, x4]])
            poly = patches.Polygon(pts, closed=True, edgecolor='#2c3e50', facecolor=color, linewidth=0.5, alpha=alpha, zorder=3)
            ax.add_patch(poly)
        
    # Draw Barrel-Endcap and Kinematic performance division boundary lines
    # Theta edges: 0.15, 0.3, barrel_theta_min (0.613626), 0.9, 1.5708
    theta_edges = [0.15, 0.3, 0.613626, 0.9, 1.5708]
    
    for edge in theta_edges:
        # Determine styling: all kinematic and barrel-endcap bounds are unified as dashed orange lines
        is_bt = abs(edge - 0.613626) < 1e-4
        is_pi2 = abs(edge - 1.5708) < 1e-3
        
        # Dynamically set ray length and text placement radius to stay outside the calorimeter and within new Z<5000, X<4000 limits
        if is_bt:
            r_max = 5300.0
            r_label = 5250.0
        elif abs(edge - 0.9) < 1e-4:
            r_max = 5000.0
            r_label = 4300.0  # Pulled down and left along the ray to completely clear the legend box
        elif is_pi2:
            r_max = 3900.0
            r_label = 3850.0
        else:
            r_max = 4900.0
            r_label = 4800.0
            
        zb = r_max * np.cos(edge)
        xb = r_max * np.sin(edge)
        
        color = '#e67e22'
        ls = '--'
        lw = 1.3
        
        # Plot ray in the first quadrant
        ax.plot([0, zb], [0, xb], color=color, linestyle=ls, linewidth=lw, zorder=5)
            
        # Text labels positioning and formatting
        z_lbl = r_label * np.cos(edge)
        x_lbl = r_label * np.sin(edge)
        
        if is_bt:
            lbl_text = r"$\theta_{min}^{barrel} \approx 0.61$ rad"
        elif is_pi2:
            lbl_text = r"$\pi/2 \approx 1.57$ rad"
        else:
            lbl_text = f"{edge:.2f} rad"
            
        # Optimize alignment per ray angle to avoid overlapping the line
        if is_pi2:
            ha_pos = 'center'
            va_pos = 'bottom'
        elif is_bt or abs(edge - 0.9) < 1e-4:
            ha_pos = 'left'
            va_pos = 'bottom'
        else:
            ha_pos = 'left'
            va_pos = 'center'
            
        # Elegant white bbox to prevent overlapping with grid lines or tower blocks
        bbox_style = dict(facecolor='white', edgecolor='none', alpha=0.85, boxstyle='round,pad=0.2')
            
        # Draw text label in the first quadrant (increased font size + bbox, bold fontweight removed)
        ax.text(z_lbl, x_lbl, lbl_text, color='#d35400', fontsize=11.0, 
                ha=ha_pos, va=va_pos, bbox=bbox_style, zorder=6)
            
    # Grid and crosshairs
    ax.grid(True, linestyle="--", alpha=0.25, zorder=1)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5, zorder=2)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5, zorder=2)
    
    # Zoom into the first quadrant, setting Z up to 5000 mm and X up to 4000 mm (with 100mm margin)
    ax.set_xlim(-100, 5100)
    ax.set_ylim(-100, 4100)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=11.5)
    
    ax.set_title("Calorimeter Longitudinal Cross-Section (Z-X First Quadrant Profile)\nProjective Barrel & Endcap Tower Modules", fontsize=15.0, fontweight='bold', pad=15)
    ax.set_xlabel("Z coordinate (Beamline) [mm]", fontsize=12.5)
    ax.set_ylabel("X coordinate (Radius) [mm]", fontsize=12.5)
    
    # Custom legend (increased font size)
    legend_elements_long = [
        Line2D([0], [0], color='#2ecc71', linewidth=2.5, label='Barrel Region (52 slices)'),
        Line2D([0], [0], color='#e74c3c', linewidth=2.5, label='Endcap Region (40 slices)'),
        Line2D([0], [0], color='#e67e22', linestyle='--', linewidth=1.3, label='Kinematic Bin Boundaries')
    ]
    ax.legend(handles=legend_elements_long, loc='upper right', frameon=True, fontsize=11.0)
    
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
