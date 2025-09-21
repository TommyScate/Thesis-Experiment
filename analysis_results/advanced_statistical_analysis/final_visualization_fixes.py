#!/usr/bin/env python3
"""
Final fixes for visualizations based on user feedback
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_beautiful_bucket_analysis():
    """Create prettier bucket analysis with larger y-axis"""
    print("Creating beautiful bucket analysis...")
    
    # Load data
    data_file = "../../final_dataset/complete_analysis_py_adjusted_csv_normalized.csv"
    try:
        df = pd.read_csv(data_file, sep=';', encoding='utf-8', decimal=',')
    except:
        df = pd.read_csv(data_file, sep=';', encoding='latin-1', decimal=',')
    
    df.columns = df.columns.str.strip()
    
    # Convert numeric columns
    numeric_cols = ['likes_per_month'] + [
        'prompt_word_count', 'negative_word_count', 'actions_verbs_count', 
        'artists_count', 'camera_composition_count', 'lighting_color_count', 
        'quality_boosters_count', 'style_codes_count', 'style_modifiers_count', 
        'subjects_count'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # All 10 count features
    count_features = [
        'prompt_word_count', 'negative_word_count', 'actions_verbs_count', 
        'artists_count', 'camera_composition_count', 'lighting_color_count', 
        'quality_boosters_count', 'style_codes_count', 'style_modifiers_count', 
        'subjects_count'
    ]
    
    # Set up beautiful styling
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e', '#95a5a6', '#16a085']
    
    # Create beautiful plot
    fig, axes = plt.subplots(2, 5, figsize=(30, 16))
    axes = axes.flatten()
    
    for i, feature in enumerate(count_features):
        if feature not in df.columns:
            continue
            
        # Define bucket boundaries
        if feature == 'prompt_word_count':
            buckets = [0, 20, 40, 60, float('inf')]
            labels = ['1-20', '21-40', '41-60', '61+']
        elif feature == 'negative_word_count':
            buckets = [0, 10, 20, 30, float('inf')]
            labels = ['0-10', '11-20', '21-30', '31+']
        else:
            buckets = [0, 1, 2, 4, float('inf')]
            labels = ['0', '1', '2-3', '4+']
        
        # Create buckets
        bucket_col = f"{feature}_bucket"
        df[bucket_col] = pd.cut(df[feature], bins=buckets, labels=labels, include_lowest=True)
        
        # Calculate engagement stats
        bucket_stats = df.groupby(bucket_col, observed=False)['likes_per_month'].agg([
            'count', 'median'
        ]).reset_index()
        
        # Calculate percentage change from baseline
        baseline_median = bucket_stats.iloc[0]['median']
        bucket_stats['pct_change'] = (
            (bucket_stats['median'] - baseline_median) / baseline_median * 100
        )
        
        # Create beautiful bar plot
        bars = axes[i].bar(bucket_stats[bucket_col], bucket_stats['pct_change'], 
                          color=colors[i % len(colors)], alpha=0.8, edgecolor='white', linewidth=2)
        
        # Styling
        axes[i].set_title(f'{feature.replace("_", " ").title()}', 
                         fontweight='bold', fontsize=14, color='#2c3e50', pad=20)
        axes[i].set_ylabel('% Change from Baseline', fontsize=12, color='#2c3e50')
        axes[i].axhline(y=0, color='#34495e', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Set larger y-axis limits to prevent overflow
        y_min = min(bucket_stats['pct_change']) - 20
        y_max = max(bucket_stats['pct_change']) + 30
        axes[i].set_ylim(y_min, y_max)
        
        # Beautiful value labels
        for bar, value in zip(bars, bucket_stats['pct_change']):
            axes[i].text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (8 if bar.get_height() >= 0 else -15), 
                        f'{value:.1f}%', ha='center', 
                        va='bottom' if bar.get_height() >= 0 else 'top', 
                        fontweight='bold', fontsize=11, color='#2c3e50',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Beautiful grid
        axes[i].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        axes[i].set_axisbelow(True)
        
        # Clean spines
        for spine in axes[i].spines.values():
            spine.set_color('#bdc3c7')
            spine.set_linewidth(1)
    
    plt.suptitle('Comprehensive Bucket Analysis: All Count Features', 
                fontsize=24, fontweight='bold', color='#2c3e50', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('plots/beautiful_bucket_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created beautiful bucket analysis")

def create_correct_synergy_plot():
    """Create synergy plot using the ORIGINAL synergy data"""
    print("Creating correct synergy plot...")
    
    synergy_file = 'synergy_analysis/synergy_effects.csv'
    if not Path(synergy_file).exists():
        return
        
    df = pd.read_csv(synergy_file)
    
    # Use the original format and get top 6 for likes
    likes_data = df[df['metric'] == 'likes_per_month'].head(6)
    
    if likes_data.empty:
        return
    
    # Create 2x3 subplot for 6 synergies
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(likes_data.iterrows()):
        if i >= 6:
            break
            
        # Use the original synergy percentage from the CSV
        synergy_pct = row['synergy_percent']
        
        # Create 2x2 grid showing the correct medians
        # Use the exact values from the original data
        data_matrix = np.array([
            [row['neither_median'], row['feature2_only_median']],
            [row['feature1_only_median'], row['both_median']]
        ])
        
        # Create heatmap
        im = axes[i].imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=70, vmax=160)
        
        # Add text annotations with the exact values
        for j in range(2):
            for k in range(2):
                axes[i].text(k, j, f'{data_matrix[j, k]:.1f}', 
                           ha='center', va='center', fontweight='bold', 
                           fontsize=13, color='black')
        
        # Set labels
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels([f'No {row["feature2_name"]}', row['feature2_name']], fontsize=11)
        axes[i].set_yticklabels([f'No {row["feature1_name"]}', row['feature1_name']], fontsize=11)
        
        # Title with the correct synergy percentage
        axes[i].set_title(f'{row["feature1_name"]} × {row["feature2_name"]}\nSynergy: {synergy_pct:.1f}%', 
                         fontweight='bold', fontsize=13, pad=15)
        
        # Remove ticks
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        
        # Add colorbar for the first plot only
        if i == 0:
            cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
            cbar.set_label('Median Likes/Month', fontsize=11)
    
    plt.suptitle('Top 6 Feature Synergies: 2×2 Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('plots/correct_synergy_analysis_top_6.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created correct synergy plot")

def create_perfectly_aligned_tables():
    """Create combination tables with perfect text alignment"""
    print("Creating perfectly aligned tables...")
    
    for metric in ['likes_per_month', 'reactions_per_month']:
        try:
            combo_file = f'top_terms/{metric}_top_combinations.csv'
            if not Path(combo_file).exists():
                continue
                
            df = pd.read_csv(combo_file)
            
            # Sort by median engagement (highest to lowest)
            df = df.sort_values('median_engagement', ascending=False).head(10)
            
            # Create table
            fig, ax = plt.subplots(figsize=(18, 12))  # Even wider
            ax.axis('tight')
            ax.axis('off')
            
            # Prepare data
            table_data = df[['feature_combination', 'median_engagement']].copy()
            table_data.columns = ['Feature Combination', f'Median {metric.replace("_", " ").title()}']
            
            # Round median values
            table_data.iloc[:, 1] = table_data.iloc[:, 1].round(2)
            
            # Create table with perfect proportions
            table = ax.table(cellText=table_data.values.tolist(),
                           colLabels=table_data.columns.tolist(),
                           cellLoc='left',  # Default to left
                           loc='center',
                           colWidths=[0.8, 0.2])  # Even more space for combinations
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 3)  # Make rows taller
            
            # Style header
            for i in range(len(table_data.columns)):
                table[(0, i)].set_facecolor('#2c3e50')
                table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
                if i == 0:  # Feature combination header - left aligned
                    table[(0, i)].set_text_props(weight='bold', color='white', ha='left')
                else:  # Median header - center aligned
                    table[(0, i)].set_text_props(weight='bold', color='white', ha='center')
            
            # Style data rows with perfect alignment
            for i in range(1, len(table_data) + 1):
                for j in range(len(table_data.columns)):
                    # Alternate row colors
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#ecf0f1')
                    else:
                        table[(i, j)].set_facecolor('white')
                    
                    # Perfect text alignment
                    if j == 0:  # Feature combination - LEFT aligned
                        table[(i, j)].set_text_props(ha='left', va='center', fontsize=10)
                        table[(i, j)].set_text_props(wrap=True)  # Allow text wrapping
                    else:  # Median value - CENTER aligned
                        table[(i, j)].set_text_props(ha='center', va='center', fontsize=11, fontweight='bold')
                    
                    # Add padding
                    table[(i, j)].PAD = 0.1
            
            plt.title(f'Top 10 Feature Combinations by {metric.replace("_", " ").title()}\n(Ordered by Highest Engagement)', 
                     fontsize=18, fontweight='bold', color='#2c3e50', pad=25)
            
            plt.savefig(f'plots/perfectly_aligned_combinations_{metric}.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Created perfectly aligned table for {metric}")
            
        except Exception as e:
            print(f"Error creating {metric} table: {e}")

def main():
    """Run all final fixes"""
    print("Creating final visualization fixes...")
    
    # Delete old versions first
    old_files = [
        'plots/comprehensive_bucket_analysis.png',
        'plots/synergy_analysis_top_6.png', 
        'plots/top_combinations_table_likes_per_month_improved.png',
        'plots/top_combinations_table_reactions_per_month_improved.png'
    ]
    
    for file in old_files:
        if Path(file).exists():
            Path(file).unlink()
    
    # Create all fixed visualizations
    create_beautiful_bucket_analysis()
    create_correct_synergy_plot()
    create_perfectly_aligned_tables()
    
    print("\nAll final fixes completed!")
    print("Generated files:")
    print("- beautiful_bucket_analysis.png (prettier, larger y-axis)")
    print("- correct_synergy_analysis_top_6.png (original synergy data)")
    print("- perfectly_aligned_combinations_*.png (perfect text alignment)")

if __name__ == "__main__":
    main()