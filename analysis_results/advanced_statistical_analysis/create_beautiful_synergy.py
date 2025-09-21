#!/usr/bin/env python3
"""
Create beautiful, understandable synergy visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_beautiful_synergy_plot():
    """Create beautiful, intuitive synergy visualization"""
    print("Creating beautiful synergy plot...")
    
    synergy_file = 'synergy_analysis/synergy_effects.csv'
    if not Path(synergy_file).exists():
        return
        
    df = pd.read_csv(synergy_file)
    
    # Get top 6 synergies for engagement/month
    likes_data = df[df['metric'] == 'reactions_per_month'].head(6)
    
    if likes_data.empty:
        return
    
    # Set up beautiful styling
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # Create beautiful figure
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(likes_data.iterrows()):
        if i >= 6:
            break
            
        ax = axes[i]
        
        # Data for the visualization
        neither = row['neither_median']
        feature1_only = row['feature1_only_median'] 
        feature2_only = row['feature2_only_median']
        both = row['both_median']
        expected = neither + (feature1_only - neither) + (feature2_only - neither)
        synergy_pct = row['synergy_percent']
        
        # Create beautiful bar chart
        categories = ['Neither', f'{row["feature1_name"]}\nOnly', f'{row["feature2_name"]}\nOnly', 'Expected\nCombined', 'Actual\nCombined']
        values = [neither, feature1_only, feature2_only, expected, both]
        bar_colors = ['#bdc3c7', colors[i], colors[(i+3)%6], '#f39c12', '#e74c3c']
        
        # Create bars
        bars = ax.bar(categories, values, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Highlight the synergy effect
        synergy_height = both - expected
        if synergy_height > 0:
            ax.bar(['Actual\nCombined'], [synergy_height], bottom=[expected], 
                  color='#27ae60', alpha=0.9, edgecolor='white', linewidth=2, 
                  label=f'Synergy: +{synergy_pct:.1f}%')
        
        # Beautiful styling
        ax.set_title(f'{row["feature1_name"]} + {row["feature2_name"]}\nSynergy Effect: {synergy_pct:.1f}%', 
                    fontsize=14, fontweight='bold', color='#2c3e50', pad=20)
        ax.set_ylabel('Median Engagement per Month', fontsize=12, color='#2c3e50')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold',
                   fontsize=10, color='#2c3e50')
        
        # Add synergy annotation
        if synergy_height > 0:
            ax.annotate(f'+{synergy_pct:.1f}%\nSynergy', 
                       xy=(4, both), xytext=(4.3, both + 5),
                       fontsize=11, fontweight='bold', color='#27ae60',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f4e6', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
        
        # Grid and styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(values) * 1.2)
        
        # Clean spines
        for spine in ax.spines.values():
            spine.set_color('#bdc3c7')
            spine.set_linewidth(1)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    
    # Overall title
    plt.suptitle('Top 6 Feature Synergies: When 1+1 > 2\n(How Feature Combinations Create Extra Engagement)', 
                fontsize=20, fontweight='bold', color='#2c3e50', y=0.98)
    
    # Add explanation text
    fig.text(0.5, 0.02, 
            'Synergy Effect: The additional engagement gained when features are combined together, beyond their individual effects.\n'
            'Green bars show the "bonus" engagement from feature synergy.',
            ha='center', fontsize=12, color='#7f8c8d', style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig('plots/beautiful_synergy_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created beautiful synergy plot")

def create_synergy_summary_chart():
    """Create a summary chart of all synergies"""
    print("Creating synergy summary chart...")
    
    synergy_file = 'synergy_analysis/synergy_effects.csv'
    if not Path(synergy_file).exists():
        return
        
    df = pd.read_csv(synergy_file)
    
    # Get top 6 synergies for engagement/month
    likes_data = df[df['metric'] == 'reactions_per_month'].head(6)
    
    if likes_data.empty:
        return
    
    # Create summary horizontal bar chart
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Prepare data
    combinations = []
    synergy_values = []
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for _, row in likes_data.iterrows():
        combinations.append(f'{row["feature1_name"]} + {row["feature2_name"]}')
        synergy_values.append(row['synergy_percent'])
    
    # Create horizontal bars
    y_pos = np.arange(len(combinations))
    bars = ax.barh(y_pos, synergy_values, color=colors, alpha=0.8, 
                   edgecolor='white', linewidth=2, height=0.6)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, synergy_values)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
               f'{value:.1f}%', va='center', fontweight='bold', fontsize=12, color='#2c3e50')
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(combinations, fontsize=12, color='#2c3e50')
    ax.set_xlabel('Synergy Effect (%)', fontsize=14, fontweight='bold', color='#2c3e50')
    ax.set_title('Feature Synergy Ranking: Top Combinations for Maximum Engagement\n'
                'Higher percentages = stronger synergy when features are combined', 
                fontsize=16, fontweight='bold', color='#2c3e50', pad=25)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Set limits
    ax.set_xlim(0, max(synergy_values) * 1.15)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    
    plt.tight_layout()
    plt.savefig('plots/synergy_summary_ranking.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created synergy summary chart")

def main():
    """Create beautiful synergy visualizations"""
    print("Creating beautiful synergy visualizations...")
    
    # Delete old synergy file
    old_file = 'plots/correct_synergy_analysis_top_6.png'
    if Path(old_file).exists():
        Path(old_file).unlink()
    
    # Create new beautiful visualizations
    create_beautiful_synergy_plot()
    create_synergy_summary_chart()
    
    print("\nBeautiful synergy visualizations completed!")
    print("Generated files:")
    print("- beautiful_synergy_analysis.png (intuitive bar charts)")
    print("- synergy_summary_ranking.png (ranking overview)")

if __name__ == "__main__":
    main()