#!/usr/bin/env python
"""Generate a publication-quality 2D cross-section lattice schematic of a Dual-Readout Calorimeter."""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def main():
    # Set publication aesthetic styles
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'serif'
    
    fig, ax = plt.subplots(figsize=(6, 6.2))
    
    # Dimensions of the absorber block (in mm)
    absorber_width = 9.0
    absorber_height = 9.0
    pitch = 1.5          # Fiber pitch (distance between centers)
    fiber_radius = 0.5   # Fiber radius (1.0 mm diameter)
    
    # 1. Draw absorber matrix (Solid copper block background)
    absorber = patches.Rectangle(
        (0, 0), absorber_width, absorber_height, 
        facecolor='#c87d55', edgecolor='#a0522d', linewidth=1.5,
        label='Copper Absorber Matrix', zorder=1
    )
    ax.add_patch(absorber)
    
    # 2. Draw grid boundaries (grooves in the copper absorber)
    for x in np.arange(0, absorber_width + 0.1, pitch):
        ax.axvline(x, color='#8b4513', linestyle='-', linewidth=0.5, alpha=0.3, zorder=2)
    for y in np.arange(0, absorber_height + 0.1, pitch):
        ax.axhline(y, color='#8b4513', linestyle='-', linewidth=0.5, alpha=0.3, zorder=2)
        
    # 3. Generate checkerboard array of fibers
    x_centers = np.arange(pitch/2, absorber_width, pitch)
    y_centers = np.arange(pitch/2, absorber_height, pitch)
    
    s_count = 0
    c_count = 0
    
    for i, x in enumerate(x_centers):
        for j, y in enumerate(y_centers):
            # Alternating S and C fibers (checkerboard)
            is_scintillator = (i + j) % 2 == 0
            
            if is_scintillator:
                # Scintillating fiber (Green)
                face_color = '#4cd137'
                edge_color = '#44bd32'
                s_count += 1
            else:
                # Cherenkov fiber (Blue)
                face_color = '#00a8ff'
                edge_color = '#0097e6'
                c_count += 1
                
            fiber = patches.Circle(
                (x, y), fiber_radius, 
                facecolor=face_color, edgecolor=edge_color, 
                linewidth=1.2, zorder=3
            )
            ax.add_patch(fiber)
            
    # Set boundaries and aspect ratio
    ax.set_xlim(-0.2, absorber_width + 0.2)
    ax.set_ylim(-0.2, absorber_height + 0.2)
    ax.set_aspect('equal')
    
    # Axes styling
    ax.set_title("DRC Tower Cross-Section Lattice", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("X coordinate (mm)", fontsize=10)
    ax.set_ylabel("Y coordinate (mm)", fontsize=10)
    
    # Clean spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    # 4. Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        patches.Patch(facecolor='#c87d55', edgecolor='#a0522d', linewidth=1.5, label='Copper Absorber Matrix'),
        Line2D([0], [0], marker='o', color='w', label='Scintillating Fiber (S) [dE/dx]',
               markerfacecolor='#4cd137', markeredgecolor='#44bd32', markersize=10, markeredgewidth=1.2),
        Line2D([0], [0], marker='o', color='w', label='Cherenkov Fiber (C) [Relativistic / EM]',
               markerfacecolor='#00a8ff', markeredgecolor='#0097e6', markersize=10, markeredgewidth=1.2)
    ]
    ax.legend(
        handles=legend_elements, loc='upper center', 
        bbox_to_anchor=(0.5, -0.15), frameon=True, shadow=False, ncol=1, fontsize=9
    )
    
    # Save options
    os.makedirs("plots", exist_ok=True)
    png_path = "plots/drc_lattice_schematic.png"
    pdf_path = "plots/drc_lattice_schematic.pdf"
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated 2D lattice plots:")
    print(f"  - PNG: {png_path}")
    print(f"  - PDF (Vector): {pdf_path}")

if __name__ == "__main__":
    main()
