#!/usr/bin/env python3
"""
Fix specific issues in the advanced analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from pathlib import Path
import itertools

def calculate_cliffs_delta(group1, group2):
    """Calculate Cliff's delta effect size"""
    if len(group1) == 0 or len(group2) == 0:
        return 0.0
        
    # Remove NaN values
    group1 = group1.dropna()
    group2 = group2.dropna()
    
    if len(group1) == 0 or len(group2) == 0:
        return 0.0
        
    n1, n2 = len(group1), len(group2)
    
    # Calculate how many times group1 > group2
    dominance = 0
    for x in group1:
        for y in group2:
            if x > y:
                dominance += 1
            elif x < y:
                dominance -= 1
                
    return dominance / (n1 * n2)

def fix_median_contrasts():
    """Fix median contrasts to include has_weights"""
    print("Fixing median contrasts to include has_weights...")
    
    # Load data
    data_file = "../../final_dataset/complete_analysis_py_adjusted_csv_normalized.csv"
    try:
        df = pd.read_csv(data_file, sep=';', encoding='utf-8', decimal=',')
    except:
        df = pd.read_csv(data_file, sep=';', encoding='latin-1', decimal=',')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert numeric columns properly
    numeric_cols = ['likes_per_month', 'reactions_per_month', 'like_count', 'total_reactions']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create has_weights feature
    if 'weights' in df.columns:
        df['has_weights'] = df['weights'].astype(str).str.upper() == 'TRUE'
    else:
        df['has_weights'] = False
    
    # Define all features including has_weights
    boolean_features = [
        'has_subjects', 'has_style_modifiers', 'has_quality_boosters',
        'has_camera_composition', 'has_lighting_color', 'has_artists', 
        'has_actions_verbs', 'has_style_codes', 'has_weights'
    ]
    
    # Create boolean features from count columns
    count_features = [
        'subjects_count', 'style_modifiers_count', 'quality_boosters_count', 
        'camera_composition_count', 'lighting_color_count', 'artists_count',
        'actions_verbs_count', 'style_codes_count'
    ]
    
    for feature in count_features:
        if feature in df.columns:
            bool_feature = f"has_{feature.replace('_count', '')}"
            df[bool_feature] = (df[feature] > 0).astype(bool)
    
    engagement_metrics = ['likes_per_month', 'reactions_per_month']
    
    for metric in engagement_metrics:
        if metric not in df.columns:
            continue
            
        results = []
        
        for feature in boolean_features:
            if feature not in df.columns:
                continue
                
            # Split data by feature presence
            has_feature = df[df[feature] == True][metric].dropna()
            no_feature = df[df[feature] == False][metric].dropna()
            
            if len(has_feature) < 10 or len(no_feature) < 10:
                continue
            
            # Calculate medians
            median_with = has_feature.median()
            median_without = no_feature.median()
            median_diff = median_with - median_without
            
            # Mann-Whitney U test
            try:
                u_stat, p_value = mannwhitneyu(has_feature, no_feature, alternative='two-sided')
            except:
                u_stat, p_value = np.nan, np.nan
            
            # Cliff's delta effect size
            cliffs_delta = calculate_cliffs_delta(has_feature, no_feature)
            
            # Effect size interpretation
            if abs(cliffs_delta) < 0.11:
                effect_size = "Negligible"
            elif abs(cliffs_delta) < 0.28:
                effect_size = "Small"
            elif abs(cliffs_delta) < 0.43:
                effect_size = "Medium"
            else:
                effect_size = "Large"
            
            results.append({
                'feature': feature.replace('has_', '').replace('_', ' ').title(),
                'median_with_feature': median_with,
                'median_without_feature': median_without,
                'median_difference': median_diff,
                'cliffs_delta': cliffs_delta,
                'effect_size': effect_size,
                'mann_whitney_u': u_stat,
                'p_value': p_value,
                'n_with_feature': len(has_feature),
                'n_without_feature': len(no_feature)
            })
        
        # Convert to DataFrame and sort by effect size
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('cliffs_delta', key=abs, ascending=False)
        
        # Save results
        output_file = f'median_contrasts/{metric}_median_contrasts.csv'
        results_df.to_csv(output_file, index=False)
        print(f"Updated {output_file}")

def fix_bucket_analysis():
    """Fix bucket analysis to use count features"""
    print("Fixing bucket analysis...")
    
    # Load data
    data_file = "../../final_dataset/complete_analysis_py_adjusted_csv_normalized.csv"
    try:
        df = pd.read_csv(data_file, sep=';', encoding='utf-8', decimal=',')
    except:
        df = pd.read_csv(data_file, sep=';', encoding='latin-1', decimal=',')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert numeric columns properly
    numeric_cols = ['likes_per_month', 'reactions_per_month'] + [
        'prompt_word_count', 'negative_word_count', 'actions_verbs_count',
        'artists_count', 'camera_composition_count', 'lighting_color_count',
        'quality_boosters_count', 'style_codes_count', 'style_modifiers_count',
        'subjects_count'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Define count features for bucket analysis as requested
    count_features = [
        'prompt_word_count', 'negative_word_count', 'actions_verbs_count', 
        'artists_count', 'camera_composition_count', 'lighting_color_count', 
        'quality_boosters_count', 'style_codes_count', 'style_modifiers_count', 
        'subjects_count'
    ]
    
    results = []
    
    for feature in count_features:
        if feature not in df.columns:
            continue
            
        # Create buckets based on distribution
        feature_data = df[feature].dropna()
        if len(feature_data) == 0:
            continue
            
        # Define bucket boundaries
        if feature == 'prompt_word_count':
            buckets = [0, 20, 40, 60, float('inf')]
            labels = ['1-20', '21-40', '41-60', '61+']
        elif feature == 'negative_word_count':
            buckets = [0, 10, 20, 30, float('inf')]
            labels = ['0-10', '11-20', '21-30', '31+']
        else:
            # For other count features, use 0, 1, 2-3, 4+ structure
            buckets = [0, 1, 2, 4, float('inf')]
            labels = ['0', '1', '2-3', '4+']
        
        # Create bucket column
        bucket_col = f"{feature}_bucket"
        df[bucket_col] = pd.cut(df[feature], bins=buckets, labels=labels, include_lowest=True)
        
        # Calculate engagement by bucket
        for metric in ['likes_per_month', 'reactions_per_month']:
            if metric not in df.columns:
                continue
                
            bucket_stats = df.groupby(bucket_col)[metric].agg([
                'count', 'median', 'mean', 'std'
            ]).reset_index()
            
            bucket_stats['feature'] = feature
            bucket_stats['engagement_metric'] = metric
            bucket_stats = bucket_stats.rename(columns={bucket_col: 'bucket'})
            
            # Calculate percentage change from baseline (bucket 0 or first bucket)
            baseline_median = bucket_stats.iloc[0]['median']
            bucket_stats['pct_change_from_baseline'] = (
                (bucket_stats['median'] - baseline_median) / baseline_median * 100
            )
            
            results.append(bucket_stats)
    
    # Combine results
    if results:
        all_results = pd.concat(results, ignore_index=True)
        output_file = 'bucket_analysis/bucket_statistics.csv'
        all_results.to_csv(output_file, index=False)
        print(f"Updated {output_file}")

def create_top_combinations_table():
    """Create top combinations table visualization"""
    print("Creating top combinations table...")
    
    for metric in ['likes_per_month', 'reactions_per_month']:
        try:
            # Load the combinations data
            combo_file = f'top_terms/{metric}_top_combinations.csv'
            if not Path(combo_file).exists():
                continue
                
            df = pd.read_csv(combo_file)
            
            # Create table visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('tight')
            ax.axis('off')
            
            # Prepare data for table - combination and median engagement only
            table_data = df[['combination', 'median_engagement']].head(10)
            table_data.columns = ['Feature Combination', f'Median {metric.replace("_", " ").title()}']
            
            # Create table
            table = ax.table(cellText=table_data.values,
                           colLabels=table_data.columns,
                           cellLoc='left',
                           loc='center',
                           bbox=[0, 0, 1, 1])
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style header
            for i in range(len(table_data.columns)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(table_data) + 1):
                for j in range(len(table_data.columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f1f1f2')
                    else:
                        table[(i, j)].set_facecolor('white')
            
            plt.title(f'Top 10 Feature Combinations by {metric.replace("_", " ").title()}', 
                     fontsize=16, fontweight='bold', pad=20)
            
            plt.savefig(f'plots/top_combinations_table_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating {metric} combinations table: {e}")

def fix_median_contrasts_plot():
    """Fix median contrasts plot to remove negligible text"""
    print("Fixing median contrasts plot...")
    
    try:
        # Load median contrasts data
        likes_file = 'median_contrasts/likes_per_month_median_contrasts.csv'
        if not Path(likes_file).exists():
            return
            
        df = pd.read_csv(likes_file)
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by effect size
        df_sorted = df.sort_values('cliffs_delta')
        
        # Create bars
        bars = ax.barh(df_sorted['feature'], df_sorted['cliffs_delta'], 
                      color=['red' if x < 0 else 'green' for x in df_sorted['cliffs_delta']])
        
        # Customize plot
        ax.set_xlabel("Cliff's Delta Effect Size", fontsize=12)
        ax.set_title("Feature Impact on Engagement\n(Median Contrasts with Effect Sizes)", fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on bars (no negligible text as requested)
        for bar, value in zip(bars, df_sorted['cliffs_delta']):
            ax.text(value + (0.01 if value >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left' if value >= 0 else 'right', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/median_contrasts_effect_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error creating median contrasts plot: {e}")

def main():
    """Run all fixes"""
    print("Running analysis fixes...")
    
    # Create directories if needed
    Path('median_contrasts').mkdir(exist_ok=True)
    Path('bucket_analysis').mkdir(exist_ok=True)
    Path('plots').mkdir(exist_ok=True)
    Path('top_terms').mkdir(exist_ok=True)
    
    fix_median_contrasts()
    fix_bucket_analysis()
    create_top_combinations_table()
    fix_median_contrasts_plot()
    
    print("All fixes completed!")

if __name__ == "__main__":
    main()