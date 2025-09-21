#!/usr/bin/env python3
"""
Refined Comprehensive Correlation Analysis - 20 Variables with Advanced Techniques
Focuses on Spearman correlations, log transformations, percentile rankings, and success thresholds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main refined correlation analysis function."""
    print("🚀 Starting REFINED Comprehensive Correlation Analysis")
    print("=" * 60)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Define engagement metrics
    metrics = {
        'like_count': 'Raw Likes',
        'total_reactions': 'Raw Total Engagement', 
        'likes_per_month': 'Normalized Likes/Month',
        'reactions_per_month': 'Normalized Engagement/Month'
    }
    
    # Create main output directory
    output_base = Path('analysis_results/refined_correlation_analysis')
    output_base.mkdir(exist_ok=True, parents=True)
    
    # Analyze each metric with comprehensive techniques
    for metric, label in metrics.items():
        print(f"\n🎯 Analyzing {label}...")
        
        # Create metric-specific directory
        metric_dir = output_base / metric
        metric_dir.mkdir(exist_ok=True)
        
        # Perform comprehensive analysis
        analyze_metric_comprehensive(df, metric, label, metric_dir)
    
    print(f"\n✅ All refined analyses complete!")
    print(f"📁 Results in: {output_base}")

def load_and_prepare_data():
    """Load and prepare the dataset with refined features."""
    print("📊 Loading dataset...")
    
    # Load with proper handling
    df = pd.read_csv('final_dataset/complete_analysis_py_adjusted_csv_normalized.csv', 
                     sep=';', decimal=',')
    
    print(f"✅ Dataset loaded: {len(df):,} rows")
    
    # Create binary features for the 8 main categories
    categories = ['subjects', 'style_modifiers', 'quality_boosters', 
                  'camera_composition', 'lighting_color', 'artists', 
                  'actions_verbs', 'style_codes']
    
    for cat in categories:
        df[f'has_{cat}'] = (df[f'{cat}_count'] > 0).astype(int)
    
    # Handle special boolean features
    df['has_weights'] = df['weights'].astype(int)
    # has_negative is already in the dataset
    
    print(f"✅ Binary features created")
    return df

def analyze_metric_comprehensive(df, metric, label, output_dir):
    """Perform comprehensive correlation analysis for one engagement metric."""
    
    # Define the refined 20-variable feature set in STRICT CONSISTENT ORDER
    # 1. Boolean variables (8) - alphabetical by category
    boolean_features = [
        'has_actions_verbs',
        'has_artists',
        'has_camera_composition',
        'has_lighting_color',
        'has_quality_boosters',
        'has_style_codes',
        'has_style_modifiers',
        'has_subjects'
    ]
    
    # 2. Count variables (8) - same order as boolean
    count_features = [
        'actions_verbs_count',
        'artists_count',
        'camera_composition_count',
        'lighting_color_count',
        'quality_boosters_count',
        'style_codes_count',
        'style_modifiers_count',
        'subjects_count'
    ]
    
    # 3. Special booleans (2) - alphabetical
    special_features = ['has_negative', 'has_weights']
    
    # 4. Word counts (2) - alphabetical
    word_features = ['negative_word_count', 'prompt_word_count']
    
    # FINAL ORDERED FEATURE LIST (20 features total)
    all_features = boolean_features + count_features + special_features + word_features
    
    print(f"  📊 Analyzing {len(all_features)} refined features...")
    
    # 1. Basic Spearman correlations
    correlations = calculate_spearman_correlations(df, metric, all_features)
    correlations.to_csv(output_dir / 'spearman_correlations.csv', index=False)
    
    # 2. Log transformation analysis
    log_correlations = analyze_log_transformations(df, metric, all_features)
    log_correlations.to_csv(output_dir / 'log_transformed_correlations.csv', index=False)
    
    # 3. Percentile ranking analysis
    percentile_correlations = analyze_percentile_rankings(df, metric, all_features)
    percentile_correlations.to_csv(output_dir / 'percentile_correlations.csv', index=False)
    
    # 4. Success threshold analysis
    threshold_analysis = analyze_success_thresholds(df, metric, all_features)
    threshold_analysis.to_csv(output_dir / 'threshold_analysis.csv', index=False)
    
    # 5. Comprehensive t-tests for binary features
    t_test_results = perform_comprehensive_t_tests(df, metric, boolean_features + special_features)
    if len(t_test_results) > 0:
        t_test_df = pd.DataFrame(t_test_results)
        t_test_df.to_csv(output_dir / 'comprehensive_t_tests.csv', index=False)
    
    # 6. Create comprehensive visualizations
    create_comprehensive_visualizations(
        correlations, log_correlations, percentile_correlations, 
        threshold_analysis, t_test_results, metric, label, output_dir
    )
    
    # 7. Generate comprehensive report
    create_comprehensive_report(
        correlations, log_correlations, percentile_correlations,
        threshold_analysis, t_test_results, metric, label, output_dir
    )
    
    print(f"  ✅ Comprehensive {metric} analysis complete")

def calculate_spearman_correlations(df, metric, features):
    """Calculate Spearman correlations (robust to outliers, handles non-linear relationships)."""
    results = []
    
    for feature in features:
        if feature in df.columns:
            try:
                # Get clean data
                x = df[feature].dropna()
                y = df[metric].dropna()
                
                # Align the data
                common_idx = x.index.intersection(y.index)
                x_clean = x.loc[common_idx]
                y_clean = y.loc[common_idx]
                
                if len(x_clean) > 20 and len(y_clean) > 20:
                    # Spearman correlation (primary focus) - returns tuple
                    spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)
                    
                    # Also calculate Pearson for comparison - returns tuple
                    pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
                    
                    results.append({
                        'feature': feature,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'abs_spearman': abs(float(spearman_r)),
                        'n_samples': len(x_clean),
                        'spearman_significant': float(spearman_p) < 0.05,
                        'effect_size': classify_effect_size(abs(float(spearman_r)))
                    })
                    
            except Exception as e:
                print(f"    ⚠️ Error with {feature}: {e}")
    
    # Create DataFrame and maintain consistent feature order
    corr_df = pd.DataFrame(results)
    if len(corr_df) > 0:
        # Reorder according to predefined feature order
        feature_order = {feature: i for i, feature in enumerate(features)}
        corr_df['order'] = corr_df['feature'].map(feature_order)
        corr_df = corr_df.sort_values('order').drop('order', axis=1)
    
    return corr_df

def analyze_log_transformations(df, metric, all_features):
    """Analyze correlations after log transformation - all 20 features with appropriate handling."""
    results = []
    
    # Define which features can be log-transformed (count variables)
    count_vars = ['actions_verbs_count', 'artists_count', 'camera_composition_count',
                  'lighting_color_count', 'quality_boosters_count', 'style_codes_count',
                  'style_modifiers_count', 'subjects_count', 'negative_word_count', 'prompt_word_count']
    
    for feature in all_features:
        if feature in df.columns:
            try:
                if feature in count_vars:
                    # Log transform count variables
                    transformed_feature = np.log1p(df[feature])
                    transformation_type = 'log1p'
                else:
                    # Keep boolean variables as-is (log transformation not meaningful)
                    transformed_feature = df[feature]
                    transformation_type = 'original'
                
                # Get clean data
                x = transformed_feature.dropna()
                y = df[metric].dropna()
                
                # Align the data
                common_idx = x.index.intersection(y.index)
                x_clean = x.loc[common_idx]
                y_clean = y.loc[common_idx]
                
                if len(x_clean) > 20:
                    spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)
                    pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
                    
                    results.append({
                        'feature': feature,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'abs_spearman': abs(float(spearman_r)),
                        'transformation': transformation_type
                    })
                    
            except Exception as e:
                print(f"    ⚠️ Transformation error with {feature}: {e}")
    
    # Maintain consistent order
    corr_df = pd.DataFrame(results)
    if len(corr_df) > 0:
        feature_order = {feature: i for i, feature in enumerate(all_features)}
        corr_df['order'] = corr_df['feature'].map(feature_order)
        corr_df = corr_df.sort_values('order').drop('order', axis=1)
    
    return corr_df

def analyze_percentile_rankings(df, metric, all_features):
    """Analyze correlations using percentile rankings - all 20 features."""
    results = []
    
    for feature in all_features:
        if feature in df.columns:
            try:
                # Convert to percentile ranks (works for both boolean and count variables)
                percentile_feature = df[feature].rank(pct=True)
                percentile_metric = df[metric].rank(pct=True)
                
                # Get clean data
                x = percentile_feature.dropna()
                y = percentile_metric.dropna()
                
                # Align the data
                common_idx = x.index.intersection(y.index)
                x_clean = x.loc[common_idx]
                y_clean = y.loc[common_idx]
                
                if len(x_clean) > 20:
                    spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)
                    pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
                    
                    results.append({
                        'feature': feature,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'abs_spearman': abs(float(spearman_r)),
                        'transformation': 'percentile_rank'
                    })
                    
            except Exception as e:
                print(f"    ⚠️ Percentile ranking error with {feature}: {e}")
    
    # Maintain consistent order
    corr_df = pd.DataFrame(results)
    if len(corr_df) > 0:
        feature_order = {feature: i for i, feature in enumerate(all_features)}
        corr_df['order'] = corr_df['feature'].map(feature_order)
        corr_df = corr_df.sort_values('order').drop('order', axis=1)
    
    return corr_df

def analyze_success_thresholds(df, metric, all_features):
    """Analyze how features relate to success thresholds (e.g., top 10%, 25%, 50%)."""
    results = []
    
    # Define success thresholds
    thresholds = {
        'top_10_percent': df[metric].quantile(0.9),
        'top_25_percent': df[metric].quantile(0.75),
        'top_50_percent': df[metric].quantile(0.5)
    }
    
    for threshold_name, threshold_value in thresholds.items():
        # Create binary success variable
        success = (df[metric] >= threshold_value).astype(int)
        
        for feature in all_features:
            if feature in df.columns:
                try:
                    # Get clean data
                    x = df[feature].dropna()
                    y = success.dropna()
                    
                    # Align the data
                    common_idx = x.index.intersection(y.index)
                    x_clean = x.loc[common_idx]
                    y_clean = y.loc[common_idx]
                    
                    if len(x_clean) > 20:
                        # Point-biserial correlation (appropriate for binary outcome)
                        corr_r, corr_p = stats.pearsonr(x_clean, y_clean)
                        
                        # Also calculate Spearman
                        spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)
                        
                        results.append({
                            'feature': feature,
                            'threshold': threshold_name,
                            'threshold_value': threshold_value,
                            'point_biserial_r': corr_r,
                            'point_biserial_p': corr_p,
                            'spearman_r': spearman_r,
                            'spearman_p': spearman_p,
                            'abs_correlation': abs(float(corr_r)),
                            'success_rate': y_clean.mean()
                        })
                        
                except Exception as e:
                    print(f"    ⚠️ Threshold analysis error with {feature}: {e}")
    
    threshold_df = pd.DataFrame(results)
    if len(threshold_df) > 0:
        threshold_df = threshold_df.sort_values(['threshold', 'abs_correlation'], ascending=[True, False])
    
    return threshold_df

def perform_comprehensive_t_tests(df, metric, binary_features):
    """Perform comprehensive t-tests with effect sizes."""
    results = []
    
    for feature in binary_features:
        if feature in df.columns:
            try:
                # Get groups
                group_1 = df[df[feature] == 1][metric].dropna()
                group_0 = df[df[feature] == 0][metric].dropna()
                
                if len(group_1) > 10 and len(group_0) > 10:
                    # Independent t-test (Welch's t-test - unequal variances)
                    t_stat, p_val = stats.ttest_ind(group_1, group_0, equal_var=False)
                    
                    # Mann-Whitney U test (non-parametric alternative)
                    u_stat, u_p = stats.mannwhitneyu(group_1, group_0, alternative='two-sided')
                    
                    # Effect sizes
                    # Cohen's d
                    pooled_std = np.sqrt(
                        ((len(group_1) - 1) * group_1.var() + 
                         (len(group_0) - 1) * group_0.var()) / 
                        (len(group_1) + len(group_0) - 2)
                    )
                    cohens_d = (group_1.mean() - group_0.mean()) / pooled_std
                    
                    # Common language effect size
                    n1, n2 = len(group_1), len(group_0)
                    u1 = u_stat
                    cles = u1 / (n1 * n2)
                    
                    results.append({
                        'feature': feature.replace('has_', ''),
                        'has_mean': group_1.mean(),
                        'has_std': group_1.std(),
                        'has_median': group_1.median(),
                        'has_count': len(group_1),
                        'no_mean': group_0.mean(),
                        'no_std': group_0.std(),
                        'no_median': group_0.median(),
                        'no_count': len(group_0),
                        'mean_diff': group_1.mean() - group_0.mean(),
                        'median_diff': group_1.median() - group_0.median(),
                        't_stat': t_stat,
                        't_p_value': p_val,
                        'u_stat': u_stat,
                        'u_p_value': u_p,
                        'cohens_d': cohens_d,
                        'cles': cles,
                        't_significant': float(p_val) < 0.05,
                        'u_significant': u_p < 0.05,
                        'effect_size_magnitude': classify_effect_size(abs(cohens_d))
                    })
                    
            except Exception as e:
                print(f"    ⚠️ T-test error with {feature}: {e}")
    
    return results

def classify_effect_size(magnitude):
    """Classify effect size magnitude according to Cohen's conventions."""
    if magnitude < 0.2:
        return 'negligible'
    elif magnitude < 0.5:
        return 'small'
    elif magnitude < 0.8:
        return 'medium'
    else:
        return 'large'

def create_comprehensive_visualizations(correlations, log_correlations, percentile_correlations, 
                                      threshold_analysis, t_test_results, metric, label, output_dir):
    """Create comprehensive visualization suite."""
    
    # 1. Main Spearman correlation heatmap
    create_spearman_heatmap(correlations, metric, label, output_dir)
    
    # 2. Transformation comparison plot
    create_transformation_comparison(correlations, log_correlations, percentile_correlations, 
                                   metric, label, output_dir)
    
    # 3. Success threshold analysis plot
    create_threshold_plot(threshold_analysis, metric, label, output_dir)
    
    # 4. Effect size visualization
    create_effect_size_plot(t_test_results, metric, label, output_dir)

def create_spearman_heatmap(correlations, metric, label, output_dir):
    """Create Spearman-only correlation heatmap with ALL 20 features (Pearson removed due to assumption violations)."""
    if len(correlations) == 0:
        return
    
    # Use ALL 20 features in consistent order (not just top 15)
    all_features = correlations
    
    plt.figure(figsize=(20, 6))
    
    # Create matrix for heatmap - SPEARMAN ONLY (Pearson violates assumptions)
    feature_names = list(all_features['feature'])
    spearman_values = list(all_features['spearman_r'])
    
    # Single row matrix with only Spearman correlations
    matrix = np.array([spearman_values])
    
    # Determine scale
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    vmax = max(vmax, 0.08)
    
    # Create heatmap - Spearman only
    sns.heatmap(matrix,
                annot=True,
                fmt='.4f',
                cmap='RdBu_r',
                center=0,
                vmin=-vmax,
                vmax=vmax,
                xticklabels=feature_names,
                yticklabels=['Spearman ρ (Robust Method)'],
                cbar_kws={'label': 'Spearman Correlation Coefficient'})
    
    plt.title(f'All 20 Features: Spearman Correlation Analysis - {label}\n(Pearson Excluded: Violates Normality & Outlier Assumptions)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontweight='bold')
    plt.ylabel('Correlation Method', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_dir / 'spearman_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_transformation_comparison(correlations, log_correlations, percentile_correlations, 
                                   metric, label, output_dir):
    """Compare correlations across different transformations."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Original correlations
    if len(correlations) > 0:
        all_orig = correlations
        axes[0].barh(range(len(all_orig)), all_orig['spearman_r'],
                    color=['red' if x < 0 else 'blue' for x in all_orig['spearman_r']])
        axes[0].set_yticks(range(len(all_orig)))
        axes[0].set_yticklabels(all_orig['feature'])
        axes[0].set_title('Original Variables')
        axes[0].set_xlabel('Spearman ρ')
        axes[0].grid(True, alpha=0.3)
    
    # Log-transformed correlations - ALL 20 features
    if len(log_correlations) > 0:
        all_log = log_correlations
        axes[1].barh(range(len(all_log)), all_log['spearman_r'],
                     color=['red' if x < 0 else 'green' for x in all_log['spearman_r']])
        axes[1].set_yticks(range(len(all_log)))
        axes[1].set_yticklabels(all_log['feature'])
        axes[1].set_title('Log-Transformed')
        axes[1].set_xlabel('Spearman ρ')
        axes[1].grid(True, alpha=0.3)
    
    # Percentile-ranked correlations - ALL 20 features
    if len(percentile_correlations) > 0:
        all_perc = percentile_correlations
        axes[2].barh(range(len(all_perc)), all_perc['spearman_r'],
                     color=['red' if x < 0 else 'purple' for x in all_perc['spearman_r']])
        axes[2].set_yticks(range(len(all_perc)))
        axes[2].set_yticklabels(all_perc['feature'])
        axes[2].set_title('Percentile-Ranked')
        axes[2].set_xlabel('Spearman ρ')
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Transformation Effects on Correlations: {label}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'transformation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_threshold_plot(threshold_analysis, metric, label, output_dir):
    """Create success threshold analysis visualization."""
    if len(threshold_analysis) == 0:
        return
    
    # Get top features for each threshold
    thresholds = threshold_analysis['threshold'].unique()
    
    fig, axes = plt.subplots(len(thresholds), 1, figsize=(14, 4 * len(thresholds)))
    if len(thresholds) == 1:
        axes = [axes]
    
    colors = ['gold', 'orange', 'red']
    
    for i, threshold in enumerate(thresholds):
        thresh_data = threshold_analysis[threshold_analysis['threshold'] == threshold]
        
        axes[i].barh(range(len(thresh_data)), thresh_data['point_biserial_r'],
                     color=colors[i], alpha=0.7)
        axes[i].set_yticks(range(len(thresh_data)))
        axes[i].set_yticklabels(thresh_data['feature'])
        axes[i].set_title(f'{threshold.replace("_", " ").title()} Success')
        axes[i].set_xlabel('Point-Biserial Correlation')
        axes[i].grid(True, alpha=0.3)
        
        # Add success rate annotations
        for j, (_, row) in enumerate(thresh_data.iterrows()):
            axes[i].text(row['point_biserial_r'] + 0.005, j, 
                        f"{row['success_rate']:.1%}", 
                        va='center', fontsize=8)
    
    plt.suptitle(f'Success Threshold Analysis: {label}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_effect_size_plot(t_test_results, metric, label, output_dir):
    """Create effect size visualization."""
    if len(t_test_results) == 0:
        return
    
    results_df = pd.DataFrame(t_test_results)
    significant = results_df[results_df['t_significant']].copy()
    
    if len(significant) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Cohen's d plot
    colors = significant['cohens_d'].apply(lambda x: 
        'red' if x < -0.2 else 'lightcoral' if x < 0 else 'lightblue' if x < 0.2 else 'blue')
    
    ax1.barh(range(len(significant)), significant['cohens_d'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(significant)))
    ax1.set_yticklabels(significant['feature'])
    ax1.set_xlabel("Cohen's d (Effect Size)")
    ax1.set_title('Effect Sizes for Significant Differences')
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.8)
    ax1.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Medium')
    ax1.axvline(x=0.8, color='gray', linestyle='--', alpha=0.9, label='Large')
    ax1.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.7)
    ax1.axvline(x=-0.8, color='gray', linestyle='--', alpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Mean differences plot
    ax2.barh(range(len(significant)), significant['mean_diff'], 
             color=['red' if x < 0 else 'green' for x in significant['mean_diff']], alpha=0.7)
    ax2.set_yticks(range(len(significant)))
    ax2.set_yticklabels(significant['feature'])
    ax2.set_xlabel('Mean Difference (Has - No)')
    ax2.set_title('Mean Differences for Significant Effects')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.8)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Effect Size Analysis: {label}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'effect_sizes.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_report(correlations, log_correlations, percentile_correlations,
                              threshold_analysis, t_test_results, metric, label, output_dir):
    """Create comprehensive analysis report."""
    report = []
    report.append(f"# REFINED Correlation Analysis Report: {label}")
    report.append("=" * 70)
    report.append("")
    report.append("## 📊 Analysis Overview")
    report.append("- Primary Method: Spearman correlations (robust to outliers)")
    report.append("- Feature Set: 20 refined variables (10 boolean + 10 count)")
    report.append("- Advanced Techniques: Log transformations, percentile rankings, success thresholds")
    report.append("")
    
    # Top Spearman correlations
    if len(correlations) > 0:
        report.append("## 🔗 Top Spearman Correlations")
        report.append("")
        
        top_10 = correlations.head(10)
        for _, row in top_10.iterrows():
            sig = "***" if row['spearman_p'] < 0.001 else "**" if row['spearman_p'] < 0.01 else "*" if row['spearman_p'] < 0.05 else ""
            effect = row['effect_size']
            report.append(f"• **{row['feature']}**: ρ = {row['spearman_r']:.4f} (p = {row['spearman_p']:.6f}){sig} [{effect}]")
        
        report.append("")
    
    # Log transformation insights
    if len(log_correlations) > 0:
        report.append("## 🔄 Log Transformation Effects")
        report.append("")
        
        top_log = log_correlations.head(5)
        for _, row in top_log.iterrows():
            transformation_note = f" ({row['transformation']})" if 'transformation' in row else ""
            report.append(f"• **{row['feature']}**: ρ = {row['spearman_r']:.4f}{transformation_note}")
        
        report.append("")
    
    # Success threshold analysis
    if len(threshold_analysis) > 0:
        report.append("## 🎯 Success Threshold Analysis")
        report.append("")
        
        for threshold in ['top_10_percent', 'top_25_percent', 'top_50_percent']:
            thresh_data = threshold_analysis[threshold_analysis['threshold'] == threshold]
            if len(thresh_data) > 0:
                top_thresh = thresh_data.head(3)
                report.append(f"### {threshold.replace('_', ' ').title()}:")
                for _, row in top_thresh.iterrows():
                    report.append(f"  • {row['feature']}: r = {row['point_biserial_r']:.4f}")
                report.append("")
    
    # Significant category effects
    if len(t_test_results) > 0:
        significant = [r for r in t_test_results if r['t_significant']]
        
        if len(significant) > 0:
            report.append("## 🧪 Significant Category Effects")
            report.append("")
            
            for result in significant:
                direction = "higher" if result['mean_diff'] > 0 else "lower"
                sig = "***" if result['t_p_value'] < 0.001 else "**" if result['t_p_value'] < 0.01 else "*"
                
                report.append(f"**{result['feature'].title()}**: {direction} engagement{sig}")
                report.append(f"  Has: {result['has_mean']:.1f} ± {result['has_std']:.1f} (n={result['has_count']:,})")
                report.append(f"  No:  {result['no_mean']:.1f} ± {result['no_std']:.1f} (n={result['no_count']:,})")
                report.append(f"  Effect size: {result['cohens_d']:.3f} ({result['effect_size_magnitude']})")
                report.append(f"  Mann-Whitney p: {result['u_p_value']:.6f}")
                report.append("")
        else:
            report.append("## ⚠️ No Statistically Significant Category Effects")
            report.append("")
    
    # Methodology notes
    report.append("## 📋 Methodology Notes")
    report.append("")
    report.append("**Statistical Methods:**")
    report.append("- Spearman ρ: Robust to outliers, captures monotonic relationships")
    report.append("- Log transformations: log(x+1) for count variables")
    report.append("- Percentile rankings: Rank-based transformations")
    report.append("- Point-biserial: Correlations with binary success thresholds")
    report.append("- Mann-Whitney U: Non-parametric group comparisons")
    report.append("")
    report.append("**Effect Size Interpretations:**")
    report.append("- Negligible: |ρ| < 0.2")
    report.append("- Small: 0.2 ≤ |ρ| < 0.5")
    report.append("- Medium: 0.5 ≤ |ρ| < 0.8")
    report.append("- Large: |ρ| ≥ 0.8")
    report.append("")
    
    # Save report
    with open(output_dir / 'comprehensive_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"    ✅ Comprehensive report saved")

if __name__ == "__main__":
    main()