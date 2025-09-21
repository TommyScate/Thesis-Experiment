#!/usr/bin/env python3
"""
Create all fixed visualizations based on user feedback
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import itertools

def create_comprehensive_bucket_analysis():
    """Create bucket analysis showing ALL count features (10 total)"""
    print("Creating comprehensive bucket analysis...")
    
    # Load data
    data_file = "../../final_dataset/complete_analysis_py_adjusted_csv_normalized.csv"
    try:
        df = pd.read_csv(data_file, sep=';', encoding='utf-8', decimal=',')
    except:
        df = pd.read_csv(data_file, sep=';', encoding='latin-1', decimal=',')
    
    df.columns = df.columns.str.strip()
    
    # Convert numeric columns
    numeric_cols = ['likes_per_month', 'reactions_per_month'] + [
        'prompt_word_count', 'negative_word_count', 'actions_verbs_count', 
        'artists_count', 'camera_composition_count', 'lighting_color_count', 
        'quality_boosters_count', 'style_codes_count', 'style_modifiers_count', 
        'subjects_count'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # All 10 count features for bucket analysis
    count_features = [
        'prompt_word_count', 'negative_word_count', 'actions_verbs_count', 
        'artists_count', 'camera_composition_count', 'lighting_color_count', 
        'quality_boosters_count', 'style_codes_count', 'style_modifiers_count', 
        'subjects_count'
    ]
    
    # Create comprehensive plot with all 10 features
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
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
        
        # Create bar plot
        bars = axes[i].bar(bucket_stats[bucket_col], bucket_stats['pct_change'])
        axes[i].set_title(f'{feature.replace("_", " ").title()}', fontweight='bold', fontsize=12)
        axes[i].set_ylabel('% Change from Baseline')
        axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, value in zip(bars, bucket_stats['pct_change']):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 
                        (5 if bar.get_height() >= 0 else -10), 
                        f'{value:.1f}%', ha='center', 
                        va='bottom' if bar.get_height() >= 0 else 'top', 
                        fontweight='bold', fontsize=10)
    
    plt.suptitle('Comprehensive Bucket Analysis: All Count Features', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/comprehensive_bucket_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created comprehensive bucket analysis with all 10 features")

def create_cooccurrence_frequency_matrix():
    """Create co-occurrence matrix showing FREQUENCIES, not correlations"""
    print("Creating co-occurrence frequency matrix...")
    
    # Load data
    data_file = "../../final_dataset/complete_analysis_py_adjusted_csv_normalized.csv"
    try:
        df = pd.read_csv(data_file, sep=';', encoding='utf-8', decimal=',')
    except:
        df = pd.read_csv(data_file, sep=';', encoding='latin-1', decimal=',')
    
    df.columns = df.columns.str.strip()
    
    # Create boolean features
    count_features = [
        'subjects_count', 'style_modifiers_count', 'quality_boosters_count', 
        'camera_composition_count', 'lighting_color_count', 'artists_count',
        'actions_verbs_count', 'style_codes_count'
    ]
    
    for feature in count_features:
        if feature in df.columns:
            bool_feature = f"has_{feature.replace('_count', '')}"
            df[bool_feature] = (df[feature] > 0).astype(bool)
    
    # Features for co-occurrence analysis
    features_to_analyze = [
        'has_actions_verbs', 'has_artists', 'has_camera_composition', 
        'has_lighting_color', 'has_quality_boosters', 'has_style_codes', 
        'has_style_modifiers', 'has_subjects'
    ]
    
    available_features = [f for f in features_to_analyze if f in df.columns]
    
    # Create frequency matrix
    n_features = len(available_features)
    frequency_matrix = np.zeros((n_features, n_features))
    
    for i, feature1 in enumerate(available_features):
        for j, feature2 in enumerate(available_features):
            if i != j:
                # Count prompts with both features
                both_present = df[df[feature1] & df[feature2]].shape[0]
                frequency_matrix[i, j] = both_present
            else:
                # Diagonal: total prompts with this feature
                frequency_matrix[i, j] = df[df[feature1]].shape[0]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    clean_names = [f.replace('has_', '').replace('_', ' ').title() for f in available_features]
    
    # Create heatmap with frequency values
    sns.heatmap(frequency_matrix, 
                annot=True, 
                fmt='.0f',  # Show as integers
                xticklabels=clean_names,
                yticklabels=clean_names,
                cmap='YlOrRd',
                cbar_kws={'label': 'Number of Prompts'},
                square=True)
    
    ax.set_title('Feature Co-occurrence Frequencies\n(Number of Prompts with Both Features)', 
                fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('plots/cooccurrence_frequency_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created co-occurrence frequency matrix")

def create_fixed_median_contrasts():
    """Create wider median contrasts plot with dotted lines instead of side numbers"""
    print("Creating fixed median contrasts plot...")
    
    likes_file = 'median_contrasts/likes_per_month_median_contrasts.csv'
    if not Path(likes_file).exists():
        return
        
    df = pd.read_csv(likes_file)
    
    # Create wider horizontal bar plot
    fig, ax = plt.subplots(figsize=(16, 10))  # Much wider
    
    # Sort by effect size
    df_sorted = df.sort_values('cliffs_delta')
    
    # Create bars
    bars = ax.barh(df_sorted['feature'], df_sorted['cliffs_delta'], 
                  color=['red' if x < 0 else 'green' for x in df_sorted['cliffs_delta']])
    
    # Customize plot
    ax.set_xlabel("Cliff's Delta Effect Size", fontsize=14)
    ax.set_title("Feature Impact on Engagement\n(Median Contrasts with Effect Sizes)", 
                fontsize=16, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Add dotted lines from bar ends to x-axis instead of side numbers
    for bar, value in zip(bars, df_sorted['cliffs_delta']):
        # Dotted line from end of bar to x-axis
        y_pos = bar.get_y() + bar.get_height()/2
        ax.plot([value, value], [y_pos, -0.5], 
               linestyle=':', color='gray', alpha=0.7, linewidth=1)
    
    # Ensure good spacing and readability
    ax.set_ylim(-0.8, len(df_sorted) - 0.2)
    
    plt.tight_layout()
    plt.savefig('plots/median_contrasts_effect_sizes_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created fixed median contrasts plot")

def fix_partial_correlations():
    """Fix partial correlations to include weights data"""
    print("Fixing partial correlations...")
    
    # Load data and regenerate partial correlations with weights
    data_file = "../../final_dataset/complete_analysis_py_adjusted_csv_normalized.csv"
    try:
        df = pd.read_csv(data_file, sep=';', encoding='utf-8', decimal=',')
    except:
        df = pd.read_csv(data_file, sep=';', encoding='latin-1', decimal=',')
    
    df.columns = df.columns.str.strip()
    
    # Convert numeric columns
    numeric_cols = ['likes_per_month', 'reactions_per_month', 'prompt_word_count']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create boolean features including weights
    count_features = [
        'subjects_count', 'style_modifiers_count', 'quality_boosters_count', 
        'camera_composition_count', 'lighting_color_count', 'artists_count',
        'actions_verbs_count', 'style_codes_count'
    ]
    
    for feature in count_features:
        if feature in df.columns:
            bool_feature = f"has_{feature.replace('_count', '')}"
            df[bool_feature] = (df[feature] > 0).astype(bool)
    
    # Handle weights
    if 'weights' in df.columns:
        df['has_weights'] = df['weights'].astype(str).str.upper() == 'TRUE'
    else:
        df['has_weights'] = False
    
    # Calculate time since posting
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    current_time = pd.Timestamp.now()
    time_diffs = []
    for created_date in df['created_at']:
        if pd.isna(created_date):
            time_diffs.append(365)
        else:
            days_diff = (current_time - created_date).days
            time_diffs.append(days_diff)
    
    df['months_since_posting'] = np.maximum(np.array(time_diffs) / 30.44, 0.1)
    
    # Features to analyze
    boolean_features = [
        'has_subjects', 'has_style_modifiers', 'has_quality_boosters',
        'has_camera_composition', 'has_lighting_color', 'has_artists', 
        'has_actions_verbs', 'has_style_codes', 'has_weights'
    ]
    
    control_vars = ['prompt_word_count', 'months_since_posting']
    available_features = [f for f in boolean_features if f in df.columns]
    available_controls = [c for c in control_vars if c in df.columns]
    
    likes_results = []
    reactions_results = []
    
    for feature in available_features:
        for metric, results_list in [('likes_per_month', likes_results), 
                                   ('reactions_per_month', reactions_results)]:
            if metric not in df.columns:
                continue
                
            # Raw correlation
            valid_data = df[[metric, feature]].dropna()
            if len(valid_data) < 100:
                continue
                
            from scipy import stats
            raw_corr, raw_p = stats.spearmanr(valid_data[metric], valid_data[feature])
            
            # Simple partial correlation (controlling for length and age)
            if available_controls:
                try:
                    partial_data = df[[metric, feature] + available_controls].dropna()
                    if len(partial_data) >= 100:
                        # Simple partial correlation calculation
                        import pingouin as pg
                        partial_corr_result = pg.partial_corr(
                            data=partial_data, x=feature, y=metric, covar=available_controls
                        )
                        partial_corr = float(partial_corr_result['r'].iloc[0])
                        partial_p = float(partial_corr_result['p-val'].iloc[0])
                        correlation_change = partial_corr - raw_corr
                    else:
                        partial_corr, partial_p = np.nan, np.nan
                        correlation_change = np.nan
                except:
                    partial_corr, partial_p = np.nan, np.nan
                    correlation_change = np.nan
            else:
                partial_corr, partial_p = raw_corr, raw_p
                correlation_change = 0
            
            results_list.append({
                'feature': feature.replace('has_', '').replace('_', ' ').title(),
                'raw_correlation': raw_corr,
                'raw_p_value': raw_p,
                'partial_correlation': partial_corr,
                'partial_p_value': partial_p,
                'correlation_change': correlation_change,
                'controls_used': ', '.join(available_controls)
            })
    
    # Save updated results
    for metric, results_list in [('likes_per_month', likes_results), 
                               ('reactions_per_month', reactions_results)]:
        if results_list:
            results_df = pd.DataFrame(results_list)
            results_df = results_df.sort_values('partial_correlation', key=abs, ascending=False)
            output_file = f'partial_correlations/{metric}_partial_correlations.csv'
            results_df.to_csv(output_file, index=False)
            print(f"Updated {output_file} with weights data")

def create_partial_correlations_plot():
    """Create partial correlations comparison plot"""
    print("Creating partial correlations plot...")
    
    likes_file = 'partial_correlations/likes_per_month_partial_correlations.csv'
    if not Path(likes_file).exists():
        return
        
    df = pd.read_csv(likes_file)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create comparison plot
    x = range(len(df))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], df['raw_correlation'], width, 
                   label='Raw Correlation', color='skyblue')
    bars2 = ax.bar([i + width/2 for i in x], df['partial_correlation'], width, 
                   label='Partial Correlation', color='lightcoral')
    
    ax.set_xlabel('Feature Families')
    ax.set_ylabel('Spearman Correlation')
    ax.set_title('Raw vs Partial Correlations (Controlled for Length & Age)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['feature'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/partial_correlations_comparison_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created partial correlations plot")

def create_top_6_synergy_plot():
    """Create synergy plot showing top 6 instead of top 3"""
    print("Creating top 6 synergy plot...")
    
    synergy_file = 'synergy_analysis/synergy_effects.csv'
    if not Path(synergy_file).exists():
        return
        
    df = pd.read_csv(synergy_file)
    
    # Get top 6 synergies for likes_per_month
    likes_synergies = df[df['metric'] == 'likes_per_month'].head(6)
    
    if likes_synergies.empty:
        return
    
    # Create plot with 2 rows, 3 columns for 6 synergies
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(likes_synergies.iterrows()):
        if i >= 6:
            break
            
        # Create 2x2 heatmap for each synergy
        data = np.array([
            [row['neither_median'], row['feature2_only_median']],
            [row['feature1_only_median'], row['both_median']]
        ])
        
        im = axes[i].imshow(data, cmap='RdYlGn', aspect='auto')
        
        # Add text annotations
        for j in range(2):
            for k in range(2):
                axes[i].text(k, j, f'{data[j, k]:.1f}', ha='center', va='center', 
                           fontweight='bold', fontsize=12)
        
        # Labels
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels([f'No {row["feature2_name"]}', row["feature2_name"]])
        axes[i].set_yticklabels([f'No {row["feature1_name"]}', row["feature1_name"]])
        axes[i].set_title(f'{row["feature1_name"]} × {row["feature2_name"]}\nSynergy: {row["synergy_percent"]:.1f}%', 
                         fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], label='Median Reactions/Month')
    
    plt.suptitle('Top 6 Feature Synergies: 2×2 Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/synergy_analysis_top_6.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created top 6 synergy plot")

def create_improved_combinations_tables():
    """Create improved top combinations tables with better formatting"""
    print("Creating improved combinations tables...")
    
    for metric in ['likes_per_month', 'reactions_per_month']:
        try:
            combo_file = f'top_terms/{metric}_top_combinations.csv'
            if not Path(combo_file).exists():
                continue
                
            df = pd.read_csv(combo_file)
            
            # Sort by median engagement (highest to lowest)
            df = df.sort_values('median_engagement', ascending=False).head(10)
            
            # Create improved table
            fig, ax = plt.subplots(figsize=(16, 10))  # Wider figure
            ax.axis('tight')
            ax.axis('off')
            
            # Prepare data
            table_data = df[['feature_combination', 'median_engagement']].copy()
            table_data.columns = ['Feature Combination', f'Median {metric.replace("_", " ").title()}']
            
            # Round median values
            table_data.iloc[:, 1] = table_data.iloc[:, 1].round(2)
            
            # Create table with improved proportions
            table = ax.table(cellText=table_data.values.tolist(),
                           colLabels=table_data.columns.tolist(),
                           cellLoc='left',
                           loc='center',
                           colWidths=[0.75, 0.25])  # 75% for combination, 25% for median
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5)
            
            # Style header
            for i in range(len(table_data.columns)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
                if i == 1:  # Center the median column
                    table[(0, i)].set_text_props(weight='bold', color='white', ha='center')
            
            # Style data rows
            for i in range(1, len(table_data) + 1):
                for j in range(len(table_data.columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f1f1f2')
                    else:
                        table[(i, j)].set_facecolor('white')
                    
                    # Center align the median values
                    if j == 1:
                        table[(i, j)].set_text_props(ha='center')
            
            plt.title(f'Top 10 Feature Combinations by {metric.replace("_", " ").title()}\n(Ordered by Highest Engagement)', 
                     fontsize=16, fontweight='bold', pad=20)
            
            plt.savefig(f'plots/top_combinations_table_{metric}_improved.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Created improved combinations table for {metric}")
            
        except Exception as e:
            print(f"Error creating {metric} combinations table: {e}")

def main():
    """Run all visualization fixes"""
    print("Creating all fixed visualizations...")
    
    # Create plots directory
    Path('plots').mkdir(exist_ok=True)
    
    # Create all fixed visualizations
    create_comprehensive_bucket_analysis()
    create_cooccurrence_frequency_matrix()
    create_fixed_median_contrasts()
    fix_partial_correlations()
    create_partial_correlations_plot()
    create_top_6_synergy_plot()
    create_improved_combinations_tables()
    
    print("\nAll fixed visualizations completed!")
    print("Generated files:")
    print("- comprehensive_bucket_analysis.png (10 features)")
    print("- cooccurrence_frequency_matrix.png (frequencies, not correlations)")
    print("- median_contrasts_effect_sizes_fixed.png (wider, dotted lines)")
    print("- partial_correlations_comparison_fixed.png (with weights data)")
    print("- synergy_analysis_top_6.png (6 synergies, not 3)")
    print("- top_combinations_table_*_improved.png (better formatting)")

if __name__ == "__main__":
    main()