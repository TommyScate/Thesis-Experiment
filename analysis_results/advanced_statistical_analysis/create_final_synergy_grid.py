#!/usr/bin/env python3
"""
Create the most beautiful 2x2 synergy tables grid
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_beautiful_2x2_table(ax, row, feature1_name, feature2_name, synergy_pct):
    """Create a single beautiful 2x2 synergy table"""
    
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
    ax.set_xlim(-0.7, 2.2)
    ax.set_ylim(-0.7, 2.7)
    ax.set_aspect('equal')
    
    # Enhanced color mapping function (beautiful red to green)
    def get_color(value):
        if max_val == min_val:
            return '#3498db'  # Blue if all same
        norm = (value - min_val) / (max_val - min_val)
        
        # Beautiful gradient: Red -> Orange -> Yellow -> Light Green -> Green
        if norm <= 0.25:
            # Red to Orange
            mix = norm * 4
            return (1.0, mix * 0.5, 0.0, 0.85)
        elif norm <= 0.5:
            # Orange to Yellow
            mix = (norm - 0.25) * 4
            return (1.0, 0.5 + mix * 0.5, 0.0, 0.85)
        elif norm <= 0.75:
            # Yellow to Light Green
            mix = (norm - 0.5) * 4
            return (1.0 - mix * 0.5, 1.0, mix * 0.4, 0.85)
        else:
            # Light Green to Green
            mix = (norm - 0.75) * 4
            return (0.5 - mix * 0.5, 1.0, 0.4 + mix * 0.6, 0.85)
    
    # Create the 2x2 grid
    cell_positions = [
        (0, 1),  # top_left: Yes F1, No F2
        (1, 1),  # top_right: Yes F1, Yes F2
        (0, 0),  # bottom_left: No F1, No F2
        (1, 0),  # bottom_right: No F1, Yes F2
    ]
    
    cell_values = [top_left, top_right, bottom_left, bottom_right]
    
    # Draw cells with beautiful styling
    for (x, y), value in zip(cell_positions, cell_values):
        # Cell rectangle with rounded corners effect
        rect = patches.Rectangle((x + 0.05, y + 0.05), 0.9, 0.9, linewidth=3, 
                               edgecolor='#34495e', facecolor=get_color(value),
                               joinstyle='round', capstyle='round')
        ax.add_patch(rect)
        
        # Add subtle shadow effect
        shadow = patches.Rectangle((x + 0.02, y + 0.02), 0.9, 0.9, linewidth=0, 
                                 facecolor='#bdc3c7', alpha=0.3, zorder=0)
        ax.add_patch(shadow)
        
        # Value text (engagement mean) - large and bold
        ax.text(x + 0.5, y + 0.6, f'{value:.1f}', ha='center', va='center',
               fontsize=16, fontweight='bold', color='#2c3e50')
        
        # Units text - smaller and elegant
        ax.text(x + 0.5, y + 0.25, 'eng/month', ha='center', va='center',
               fontsize=9, color='#7f8c8d', style='italic', alpha=0.8)
    
    # Beautiful title with synergy percentage
    ax.text(0.5, 2.35, f'{feature1_name} + {feature2_name}', 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', alpha=0.8))
    ax.text(0.5, 2.05, f'+{synergy_pct:.1f}% Synergy', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='#e74c3c',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', alpha=0.8))
    
    # Beautiful axis labels OUTSIDE the grid
    # X-axis (Feature 2) - bottom with styling
    ax.text(0.5, -0.4, f'No {feature2_name}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ebf3fd', alpha=0.7))
    ax.text(1.5, -0.4, f'Yes {feature2_name}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#e8f5e8', alpha=0.7))
    
    # Y-axis (Feature 1) - left side with styling
    ax.text(-0.5, 1.5, f'Yes\n{feature1_name}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#e8f5e8', alpha=0.7))
    ax.text(-0.5, 0.5, f'No\n{feature1_name}', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ebf3fd', alpha=0.7))
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def main():
    """Create the most beautiful synergy grid"""
    print("Creating the most beautiful synergy 2x2 grid...")
    
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
    
    # Remove old individual synergy files
    print("Removing individual synergy files...")
    individual_files = [
        'synergy_2x2_quality_actions.png',
        'synergy_2x2_actions_subjects.png', 
        'synergy_2x2_camera_quality.png',
        'synergy_2x2_camera_lighting.png',
        'synergy_2x2_style_modifiers_artists.png',
        'synergy_2x2_lighting_quality.png'
    ]
    
    for filename in individual_files:
        file_path = plots_dir / filename
        if file_path.exists():
            file_path.unlink()
            print(f"Removed {filename}")
    
    # Set beautiful style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()
    
    # Create each synergy table
    for i, (_, row) in enumerate(likes_data.iterrows()):
        feature1_name = row['feature1_name']
        feature2_name = row['feature2_name']
        synergy_pct = row['synergy_percent']
        
        print(f"Creating beautiful table {i+1}: {feature1_name} + {feature2_name} (+{synergy_pct:.1f}%)")
        create_beautiful_2x2_table(axes[i], row, feature1_name, feature2_name, synergy_pct)
    
    # Beautiful overall title
    plt.suptitle('Feature Synergy Analysis: When 1+1 > 2\n2×2 Engagement Matrices Showing Feature Interaction Effects', 
                fontsize=20, fontweight='bold', color='#2c3e50', y=0.96)
    
    # Add subtle background
    fig.patch.set_facecolor('#f8f9fa')
    
    plt.tight_layout(rect=(0, 0.03, 1, 0.90))
    
    # Save the beautiful combined image
    output_path = plots_dir / 'all_synergy_2x2_tables.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='#f8f9fa', edgecolor='none')
    plt.close()
    
    print(f"\n✅ Beautiful synergy grid created!")
    print(f"Generated: {output_path}")
    print(f"🗑️  Removed {len([f for f in individual_files if (plots_dir / f).exists()])} individual synergy files")
    print("\nSynergy Effects (% MORE engagement when combined):")
    for i, (_, row) in enumerate(likes_data.iterrows()):
        feature1_name = row['feature1_name']
        feature2_name = row['feature2_name']
        synergy_pct = row['synergy_percent']
        print(f"{i+1}. {feature1_name} + {feature2_name}: +{synergy_pct:.1f}% MORE engagement")

if __name__ == "__main__":
    main()