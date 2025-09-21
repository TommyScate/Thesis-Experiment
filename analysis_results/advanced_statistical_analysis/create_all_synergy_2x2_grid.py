#!/usr/bin/env python3
"""
Create all 6 synergy 2x2 tables in a single beautiful image
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_single_2x2_table(ax, row, feature1_name, feature2_name, synergy_pct):
    """Create a single 2x2 synergy table in the given axes"""
    
    # Extract the 4 values for the 2x2 table
    top_left = row['feature1_only_median']      # Yes F1, No F2
    top_right = row['both_median']              # Yes F1, Yes F2  
    bottom_left = row['neither_median']         # No F1, No F2
    bottom_right = row['feature2_only_median']  # No F1, Yes F2
    
    # Values for color mapping
    values = [top_left, top_right, bottom_left, bottom_right]
    min_val = min(values)
    max_val = max(values)
    
    # Set axes limits
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    
    # Color mapping function (red to green)
    def get_color(value):
        if max_val == min_val:
            return '#ffeb3b'  # Yellow if all same
        norm = (value - min_val) / (max_val - min_val)
        # Red to yellow to green gradient
        if norm < 0.5:
            red = 1.0
            green = norm * 2
            blue = 0.0
        else:
            red = 2 * (1 - norm)
            green = 1.0
            blue = 0.0
        return (red, green, blue, 0.9)
    
    # Create the 2x2 grid
    cell_positions = [
        (0, 1),  # top_left: Yes F1, No F2
        (1, 1),  # top_right: Yes F1, Yes F2
        (0, 0),  # bottom_left: No F1, No F2
        (1, 0),  # bottom_right: No F1, Yes F2
    ]
    
    cell_values = [top_left, top_right, bottom_left, bottom_right]
    
    # Draw cells
    for (x, y), value in zip(cell_positions, cell_values):
        # Cell rectangle
        rect = patches.Rectangle((x, y), 1, 1, linewidth=2, 
                               edgecolor='#2c3e50', facecolor=get_color(value))
        ax.add_patch(rect)
        
        # Value text (engagement mean)
        ax.text(x + 0.5, y + 0.6, f'{value:.1f}', ha='center', va='center',
               fontsize=11, fontweight='bold', color='#2c3e50')
        
        # Units text
        ax.text(x + 0.5, y + 0.3, 'eng/month', ha='center', va='center',
               fontsize=8, color='#7f8c8d', style='italic')
    
    # Add title with synergy percentage (much higher up)
    ax.text(0.5, 3.2, f'{feature1_name} + {feature2_name}',
            ha='center', va='center', fontsize=12, fontweight='bold', color='#2c3e50',
            transform=ax.transData)
    ax.text(0.5, 2.9, f'+{synergy_pct:.1f}% Synergy',
            ha='center', va='center', fontsize=11, fontweight='bold', color='#e74c3c',
            transform=ax.transData)
    
    # Add axis labels OUTSIDE the grid
    # X-axis (Feature 2)
    ax.text(0.5, -0.3, f'No {feature2_name}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#2c3e50')
    ax.text(1.5, -0.3, f'Yes {feature2_name}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#2c3e50')
    
    # Y-axis (Feature 1) 
    ax.text(-0.3, 1.5, f'Yes {feature1_name}', ha='center', va='center', rotation=90,
            fontsize=10, fontweight='bold', color='#2c3e50')
    ax.text(-0.3, 0.5, f'No {feature1_name}', ha='center', va='center', rotation=90,
            fontsize=10, fontweight='bold', color='#2c3e50')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def main():
    """Create all 6 synergy tables in one beautiful image"""
    print("Creating all synergy 2x2 tables in one image...")
    
    # Read synergy data
    synergy_file = 'synergy_analysis/synergy_effects.csv'
    if not Path(synergy_file).exists():
        print(f"Error: {synergy_file} not found!")
        return
        
    df = pd.read_csv(synergy_file)
    
    # Get top 6 synergies for likes_per_month
    likes_data = df[df['metric'] == 'reactions_per_month'].head(6)
    
    # Create plots directory if it doesn't exist
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    # Create figure with 2x3 grid (smaller for better fit)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Create each synergy table
    for i, (_, row) in enumerate(likes_data.iterrows()):
        feature1_name = row['feature1_name']
        feature2_name = row['feature2_name']
        synergy_pct = row['synergy_percent']
        
        print(f"Creating table {i+1}: {feature1_name} + {feature2_name} (+{synergy_pct:.1f}%)")
        create_single_2x2_table(axes[i], row, feature1_name, feature2_name, synergy_pct)
    
    # Overall title (higher up)
    plt.suptitle('Feature Synergy Analysis: 2×2 Engagement Tables',
                fontsize=16, fontweight='bold', color='#2c3e50', y=0.96)
    
    plt.tight_layout(rect=(0, 0.05, 1, 0.88))
    
    # Save the combined image
    output_path = plots_dir / 'all_synergy_2x2_tables.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n✅ All 6 synergy tables created in one image!")
    print(f"Generated: {output_path}")
    print("\nSynergy Effects:")
    for i, (_, row) in enumerate(likes_data.iterrows()):
        feature1_name = row['feature1_name']
        feature2_name = row['feature2_name']
        synergy_pct = row['synergy_percent']
        print(f"{i+1}. {feature1_name} + {feature2_name}: +{synergy_pct:.1f}% synergy")

if __name__ == "__main__":
    main()