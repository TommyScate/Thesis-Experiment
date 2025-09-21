#!/usr/bin/env python3
"""
Create beautiful 2x2 synergy tables with color-coding
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_synergy_2x2_table(row, feature1_name, feature2_name, synergy_pct, output_path):
    """Create a single 2x2 synergy table with color-coding"""
    
    # Extract the 4 values for the 2x2 table
    # Top left: Yes Feature 1, No Feature 2
    top_left = row['feature1_only_median']
    # Top right: Yes Feature 1, Yes Feature 2  
    top_right = row['both_median']
    # Bottom left: No Feature 1, No Feature 2
    bottom_left = row['neither_median']
    # Bottom right: No Feature 1, Yes Feature 2
    bottom_right = row['feature2_only_median']
    
    # Values for color mapping
    values = [top_left, top_right, bottom_left, bottom_right]
    min_val = min(values)
    max_val = max(values)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
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
        return (red, green, blue, 0.8)
    
    # Create the 2x2 grid
    cell_positions = [
        (0, 1),  # top_left
        (1, 1),  # top_right
        (0, 0),  # bottom_left
        (1, 0),  # bottom_right
    ]
    
    cell_values = [top_left, top_right, bottom_left, bottom_right]
    cell_labels = [
        f'Yes {feature1_name}\nNo {feature2_name}',
        f'Yes {feature1_name}\nYes {feature2_name}',
        f'No {feature1_name}\nNo {feature2_name}',
        f'No {feature1_name}\nYes {feature2_name}'
    ]
    
    # Draw cells
    for i, ((x, y), value, label) in enumerate(zip(cell_positions, cell_values, cell_labels)):
        # Cell rectangle
        rect = patches.Rectangle((x, y), 1, 1, linewidth=3, 
                               edgecolor='#34495e', facecolor=get_color(value))
        ax.add_patch(rect)
        
        # Label text (feature combination)
        ax.text(x + 0.5, y + 0.75, label, ha='center', va='center',
               fontsize=11, fontweight='bold', color='#2c3e50')
        
        # Value text (mean engagement)
        ax.text(x + 0.5, y + 0.35, f'{value:.1f}', ha='center', va='center',
               fontsize=16, fontweight='bold', color='#2c3e50')
        
        # Units text
        ax.text(x + 0.5, y + 0.15, 'likes/month', ha='center', va='center',
               fontsize=9, color='#7f8c8d', style='italic')
    
    # Add title with synergy percentage
    plt.suptitle(f'{feature1_name} + {feature2_name} Synergy Analysis\nSynergy Effect: +{synergy_pct:.1f}% Extra Engagement', 
                fontsize=18, fontweight='bold', color='#2c3e50', y=0.95)
    
    # Add explanation
    fig.text(0.5, 0.08, 
            f'Synergy: Combining {feature1_name} + {feature2_name} gives {synergy_pct:.1f}% MORE engagement\n'
            f'than expected from their individual effects alone',
            ha='center', fontsize=12, color='#7f8c8d', style='italic')
    
    # Add color legend
    fig.text(0.5, 0.02, 
            '🔴 Red = Lower Engagement    🟡 Yellow = Medium Engagement    🟢 Green = Higher Engagement',
            ha='center', fontsize=10, color='#95a5a6')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

def main():
    """Create all 6 synergy 2x2 tables"""
    print("Creating beautiful 2x2 synergy tables...")
    
    # Read synergy data
    synergy_file = 'synergy_analysis/synergy_effects.csv'
    if not Path(synergy_file).exists():
        print(f"Error: {synergy_file} not found!")
        return
        
    df = pd.read_csv(synergy_file)
    
    # Get top 6 synergies for likes_per_month
    likes_data = df[df['metric'] == 'likes_per_month'].head(6)
    
    # Create plots directory if it doesn't exist
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    # Create each synergy table
    for i, (_, row) in enumerate(likes_data.iterrows()):
        feature1_name = row['feature1_name']
        feature2_name = row['feature2_name']
        synergy_pct = row['synergy_percent']
        
        # Create safe filename
        safe_name1 = feature1_name.lower().replace(' ', '_')
        safe_name2 = feature2_name.lower().replace(' ', '_')
        filename = f'synergy_2x2_{safe_name1}_{safe_name2}.png'
        output_path = plots_dir / filename
        
        print(f"Creating {filename}...")
        create_synergy_2x2_table(row, feature1_name, feature2_name, synergy_pct, output_path)
    
    print("\n✅ All 6 synergy 2x2 tables created!")
    print("Generated files:")
    for i, (_, row) in enumerate(likes_data.iterrows()):
        feature1_name = row['feature1_name']
        feature2_name = row['feature2_name']
        safe_name1 = feature1_name.lower().replace(' ', '_')
        safe_name2 = feature2_name.lower().replace(' ', '_')
        filename = f'synergy_2x2_{safe_name1}_{safe_name2}.png'
        synergy_pct = row['synergy_percent']
        print(f"- {filename} ({feature1_name} + {feature2_name}: +{synergy_pct:.1f}% synergy)")

if __name__ == "__main__":
    main()