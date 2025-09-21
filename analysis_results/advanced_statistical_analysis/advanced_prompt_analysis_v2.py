#!/usr/bin/env python3
"""
Advanced Statistical Analysis for Text-to-Image Prompt Engagement
================================================================

This script implements comprehensive statistical analysis using robust methods
appropriate for social media engagement data with heavy-tailed distributions.

Implements 7 advanced statistical techniques:
1. Median contrasts with Mann-Whitney U tests
2. Bucket analysis for optimal ranges
3. Synergy analysis for feature interactions
4. Co-occurrence analysis 
5. Partial correlations controlling for confounds
6. Top combinations analysis
7. Comprehensive reporting with visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
import warnings
from pathlib import Path
import itertools
import pingouin as pg

warnings.filterwarnings('ignore')

class AdvancedPromptAnalyzer:
    def __init__(self, data_file):
        """Initialize the analyzer with data file"""
        self.data_file = data_file
        self.df = None
        self.output_dir = Path("analysis_results/advanced_statistical_analysis")
        self.setup_directories()
        
    def setup_directories(self):
        """Create output directories"""
        dirs = [
            'median_contrasts', 'bucket_analysis', 'synergy_analysis',
            'co_occurrence', 'partial_correlations', 'top_terms', 'plots'
        ]
        
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
            
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("Loading and preparing data...")
        
        # Load data with proper separator and encoding handling
        try:
            self.df = pd.read_csv(self.data_file, sep=';', encoding='utf-8')
        except:
            self.df = pd.read_csv(self.data_file, sep=';', encoding='latin-1')
        
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
        # Convert date column and calculate time differences
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], errors='coerce')
        current_time = pd.Timestamp.now()
        
        # Calculate time differences properly
        time_diffs = []
        for created_date in self.df['created_at']:
            if pd.isna(created_date):
                time_diffs.append(365)
            else:
                days_diff = (current_time - created_date).days
                time_diffs.append(days_diff)
        
        self.df['days_since_posting'] = time_diffs
        self.df['months_since_posting'] = np.maximum(np.array(time_diffs) / 30.44, 0.1)
        
    def create_boolean_features(self):
        """Create boolean features from count columns"""
        print("Creating boolean features...")
        
        # Define count columns available in the dataset
        count_features = [
            'subjects_count', 'style_modifiers_count', 'quality_boosters_count', 
            'camera_composition_count', 'lighting_color_count', 'artists_count',
            'actions_verbs_count', 'style_codes_count'
        ]
        
        # Create boolean versions (has_*)
        for feature in count_features:
            if feature in self.df.columns:
                bool_feature = f"has_{feature.replace('_count', '')}"
                self.df[bool_feature] = (self.df[feature] > 0).astype(bool)
        
        # Handle weights column - convert to boolean properly
        if 'weights' in self.df.columns:
            # Convert weights column to boolean
            self.df['has_weights'] = self.df['weights'].astype(str).str.upper() == 'TRUE'
        else:
            self.df['has_weights'] = False
            
        # Ensure we have required engagement metrics
        if 'likes_per_month' not in self.df.columns:
            self.df['likes_per_month'] = self.df['like_count'] / self.df['months_since_posting']
        if 'reactions_per_month' not in self.df.columns:
            self.df['reactions_per_month'] = self.df['total_reactions'] / self.df['months_since_posting']
            
    def calculate_cliffs_delta(self, group1, group2):
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
    
    def analyze_median_contrasts(self):
        """Analyze median differences with effect sizes"""
        print("Performing median contrasts analysis...")
        
        # Define features to analyze - now including has_weights
        boolean_features = [
            'has_subjects', 'has_style_modifiers', 'has_quality_boosters',
            'has_camera_composition', 'has_lighting_color', 'has_artists', 
            'has_actions_verbs', 'has_style_codes', 'has_weights'
        ]
        
        engagement_metrics = ['likes_per_month', 'reactions_per_month']
        
        for metric in engagement_metrics:
            results = []
            
            for feature in boolean_features:
                if feature not in self.df.columns:
                    continue
                    
                # Split data by feature presence
                has_feature = self.df[self.df[feature] == True][metric].dropna()
                no_feature = self.df[self.df[feature] == False][metric].dropna()
                
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
                cliffs_delta = self.calculate_cliffs_delta(has_feature, no_feature)
                
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
            output_file = self.output_dir / 'median_contrasts' / f'{metric}_median_contrasts.csv'
            results_df.to_csv(output_file, index=False)
            print(f"Saved {metric} median contrasts to {output_file}")
            
    def analyze_bucket_effects(self):
        """Analyze optimal ranges for count variables"""
        print("Performing bucket analysis...")
        
        # Define count features for bucket analysis as requested
        count_features = [
            'prompt_word_count', 'negative_word_count', 'actions_verbs_count', 
            'artists_count', 'camera_composition_count', 'lighting_color_count', 
            'quality_boosters_count', 'style_codes_count', 'style_modifiers_count', 
            'subjects_count'
        ]
        
        results = []
        
        for feature in count_features:
            if feature not in self.df.columns:
                continue
                
            # Create buckets based on distribution
            feature_data = self.df[feature].dropna()
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
            self.df[bucket_col] = pd.cut(self.df[feature], bins=buckets, labels=labels, include_lowest=True)
            
            # Calculate engagement by bucket
            for metric in ['likes_per_month', 'reactions_per_month']:
                bucket_stats = self.df.groupby(bucket_col)[metric].agg([
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
            output_file = self.output_dir / 'bucket_analysis' / 'bucket_statistics.csv'
            all_results.to_csv(output_file, index=False)
            print(f"Saved bucket analysis to {output_file}")
            
    def analyze_synergy_effects(self):
        """Analyze 2x2 feature interactions"""
        print("Performing synergy analysis...")
        
        # Define features for synergy analysis
        boolean_features = [
            'has_subjects', 'has_style_modifiers', 'has_quality_boosters',
            'has_camera_composition', 'has_lighting_color', 'has_artists', 
            'has_actions_verbs', 'has_style_codes'
        ]
        
        # Filter features that exist in the dataset
        available_features = [f for f in boolean_features if f in self.df.columns]
        
        results = []
        
        # Test all pairs of features - limit to top 6 combinations
        feature_pairs = list(itertools.combinations(available_features, 2))
        
        for metric in ['likes_per_month', 'reactions_per_month']:
            pair_results = []
            
            for feature1, feature2 in feature_pairs:
                # Create 2x2 combinations
                neither = self.df[(~self.df[feature1]) & (~self.df[feature2])][metric].dropna()
                only_f1 = self.df[(self.df[feature1]) & (~self.df[feature2])][metric].dropna()
                only_f2 = self.df[(~self.df[feature1]) & (self.df[feature2])][metric].dropna()
                both = self.df[(self.df[feature1]) & (self.df[feature2])][metric].dropna()
                
                if len(neither) < 10 or len(both) < 10:
                    continue
                
                # Calculate medians
                median_neither = neither.median()
                median_only_f1 = only_f1.median() if len(only_f1) > 0 else median_neither
                median_only_f2 = only_f2.median() if len(only_f2) > 0 else median_neither
                median_both = both.median()
                
                # Calculate expected additive effect
                f1_effect = median_only_f1 - median_neither
                f2_effect = median_only_f2 - median_neither
                expected_both = median_neither + f1_effect + f2_effect
                
                # Calculate synergy (actual vs expected)
                synergy_absolute = median_both - expected_both
                synergy_percentage = (synergy_absolute / median_neither * 100) if median_neither > 0 else 0
                
                pair_results.append({
                    'feature_1': feature1.replace('has_', '').replace('_', ' ').title(),
                    'feature_2': feature2.replace('has_', '').replace('_', ' ').title(),
                    'median_neither': median_neither,
                    'median_both': median_both,
                    'expected_both': expected_both,
                    'synergy_absolute': synergy_absolute,
                    'synergy_percentage': synergy_percentage,
                    'n_neither': len(neither),
                    'n_both': len(both)
                })
            
            # Sort by synergy percentage and take top 6
            pair_results_df = pd.DataFrame(pair_results)
            if not pair_results_df.empty:
                pair_results_df = pair_results_df.sort_values('synergy_percentage', ascending=False).head(6)
                pair_results_df['engagement_metric'] = metric
                results.append(pair_results_df)
        
        # Save results
        if results:
            all_results = pd.concat(results, ignore_index=True)
            output_file = self.output_dir / 'synergy_analysis' / 'synergy_effects.csv'
            all_results.to_csv(output_file, index=False)
            print(f"Saved synergy analysis to {output_file}")
    
    def analyze_co_occurrence(self):
        """Analyze feature co-occurrence patterns - specific features only"""
        print("Performing co-occurrence analysis...")
        
        # Only analyze these specific features as requested
        features_to_analyze = [
            'has_actions_verbs', 'has_artists', 'has_camera_composition', 
            'has_lighting_color', 'has_quality_boosters', 'has_style_codes', 
            'has_style_modifiers', 'has_subjects'
        ]
        
        # Filter to features that exist in dataset
        available_features = [f for f in features_to_analyze if f in self.df.columns]
        
        if len(available_features) < 2:
            print("Not enough features available for co-occurrence analysis")
            return
        
        # Create correlation matrix
        feature_df = self.df[available_features].astype(float)
        correlation_matrix = feature_df.corr(method='spearman')
        
        # Clean up feature names for display
        clean_names = [f.replace('has_', '').replace('_', ' ').title() for f in available_features]
        correlation_matrix.columns = clean_names
        correlation_matrix.index = pd.Index(clean_names)
        
        # Save correlation matrix
        output_file = self.output_dir / 'co_occurrence' / 'feature_correlations.csv'
        correlation_matrix.to_csv(output_file)
        print(f"Saved co-occurrence analysis to {output_file}")
        
    def analyze_partial_correlations(self):
        """Analyze partial correlations controlling for confounds"""
        print("Performing partial correlations analysis...")
        
        # Define features to analyze - ensure has_weights is included correctly
        boolean_features = [
            'has_subjects', 'has_style_modifiers', 'has_quality_boosters',
            'has_camera_composition', 'has_lighting_color', 'has_artists', 
            'has_actions_verbs', 'has_style_codes', 'has_weights'
        ]
        
        # Control variables
        control_vars = ['prompt_word_count', 'months_since_posting']
        
        # Filter to available features and controls
        available_features = [f for f in boolean_features if f in self.df.columns]
        available_controls = [c for c in control_vars if c in self.df.columns]
        
        for metric in ['likes_per_month', 'reactions_per_month']:
            results = []
            
            for feature in available_features:
                # Raw correlation
                valid_data = self.df[[metric, feature]].dropna()
                if len(valid_data) < 100:
                    continue
                    
                raw_corr, raw_p = stats.spearmanr(valid_data[metric], valid_data[feature])
                
                # Partial correlation controlling for available controls
                if available_controls:
                    try:
                        # Prepare data for partial correlation
                        partial_data = self.df[[metric, feature] + available_controls].dropna()
                        if len(partial_data) < 100:
                            partial_corr, partial_p = np.nan, np.nan
                            correlation_change = np.nan
                        else:
                            # Use pingouin for partial correlation
                            partial_corr_result = pg.partial_corr(
                                data=partial_data, x=feature, y=metric, covar=available_controls
                            )
                            partial_corr = float(partial_corr_result['r'].iloc[0])
                            partial_p = float(partial_corr_result['p-val'].iloc[0])
                            
                            # Calculate change
                            correlation_change = partial_corr - raw_corr
                    except:
                        partial_corr, partial_p = np.nan, np.nan
                        correlation_change = np.nan
                else:
                    partial_corr, partial_p = raw_corr, raw_p
                    correlation_change = 0
                
                results.append({
                    'metric': metric,
                    'feature': feature.replace('has_', '').replace('_', ' ').title(),
                    'raw_correlation': raw_corr,
                    'raw_p_value': raw_p,
                    'partial_correlation': partial_corr,
                    'partial_p_value': partial_p,
                    'correlation_change': correlation_change,
                    'controls_used': ', '.join(available_controls)
                })
            
            # Save results
            results_df = pd.DataFrame(results)
            if not results_df.empty:
                results_df = results_df.sort_values('partial_correlation', key=abs, ascending=False)
                output_file = self.output_dir / 'partial_correlations' / f'{metric}_partial_correlations.csv'
                results_df.to_csv(output_file, index=False)
                print(f"Saved {metric} partial correlations to {output_file}")
    
    def analyze_top_combinations(self):
        """Analyze top-performing feature combinations"""
        print("Performing top combinations analysis...")
        
        boolean_features = [
            'has_subjects', 'has_style_modifiers', 'has_quality_boosters',
            'has_camera_composition', 'has_lighting_color', 'has_artists', 
            'has_actions_verbs', 'has_style_codes'
        ]
        
        available_features = [f for f in boolean_features if f in self.df.columns]
        
        for metric in ['likes_per_month', 'reactions_per_month']:
            combination_results = []
            
            # Analyze combinations of 2-4 features
            for combo_size in [2, 3, 4]:
                for combination in itertools.combinations(available_features, combo_size):
                    # Find posts with this exact combination
                    mask = pd.Series(True, index=self.df.index)
                    for feature in combination:
                        mask &= self.df[feature]
                    
                    # Exclude posts with other features not in combination
                    for other_feature in available_features:
                        if other_feature not in combination:
                            mask &= ~self.df[other_feature]
                    
                    combo_data = self.df[mask][metric].dropna()
                    
                    if len(combo_data) >= 10:  # Minimum sample size
                        median_engagement = combo_data.median()
                        
                        # Clean up combination names
                        clean_combo = ' + '.join([
                            f.replace('has_', '').replace('_', ' ').title() 
                            for f in combination
                        ])
                        
                        combination_results.append({
                            'combination': clean_combo,
                            'median_engagement': median_engagement
                        })
            
            # Sort by median engagement and take top 10
            if combination_results:
                results_df = pd.DataFrame(combination_results)
                results_df = results_df.sort_values('median_engagement', ascending=False).head(10)
                
                output_file = self.output_dir / 'top_terms' / f'{metric}_top_combinations.csv'
                results_df.to_csv(output_file, index=False)
                print(f"Saved {metric} top combinations to {output_file}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("Set2")
        
        # 1. Median contrasts effect sizes
        self.plot_median_contrasts()
        
        # 2. Bucket analysis sweet spots  
        self.plot_bucket_analysis()
        
        # 3. Synergy analysis
        self.plot_synergy_analysis()
        
        # 4. Co-occurrence heatmap
        self.plot_co_occurrence_heatmap()
        
        # 5. Partial correlations comparison
        self.plot_partial_correlations()
        
        # 6. Top feature combinations table
        self.plot_top_combinations_table()
    
    def plot_median_contrasts(self):
        """Plot median contrasts with effect sizes - no negligible text"""
        try:
            # Load median contrasts data
            likes_file = self.output_dir / 'median_contrasts' / 'likes_per_month_median_contrasts.csv'
            if not likes_file.exists():
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
            plt.savefig(self.output_dir / 'plots' / 'median_contrasts_effect_sizes.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating median contrasts plot: {e}")
    
    def plot_bucket_analysis(self):
        """Plot bucket analysis sweet spots"""
        try:
            bucket_file = self.output_dir / 'bucket_analysis' / 'bucket_statistics.csv'
            if not bucket_file.exists():
                return
                
            df = pd.read_csv(bucket_file)
            df_likes = df[df['engagement_metric'] == 'likes_per_month']
            
            # Select key features to plot
            key_features = ['quality_boosters_count', 'camera_composition_count', 'prompt_word_count']
            df_plot = df_likes[df_likes['feature'].isin(key_features)]
            
            if df_plot.empty:
                return
                
            fig, axes = plt.subplots(1, len(key_features), figsize=(15, 5))
            if len(key_features) == 1:
                axes = [axes]
            
            for i, feature in enumerate(key_features):
                feature_data = df_plot[df_plot['feature'] == feature]
                
                bars = axes[i].bar(feature_data['bucket'], feature_data['pct_change_from_baseline'])
                axes[i].set_title(f'{feature.replace("_", " ").title()}', fontweight='bold')
                axes[i].set_ylabel('% Change from Baseline')
                axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                
                # Add value labels
                for bar, value in zip(bars, feature_data['pct_change_from_baseline']):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                               f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.suptitle('Optimal Ranges Analysis: Sweet Spots for Engagement', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'bucket_analysis_sweet_spots.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating bucket analysis plot: {e}")
    
    def plot_synergy_analysis(self):
        """Plot synergy analysis"""
        try:
            synergy_file = self.output_dir / 'synergy_analysis' / 'synergy_effects.csv'
            if not synergy_file.exists():
                return
                
            df = pd.read_csv(synergy_file)
            df_likes = df[df['engagement_metric'] == 'likes_per_month'].head(6)  # Top 6 as requested
            
            if df_likes.empty:
                return
                
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create combination labels
            df_likes['combination'] = df_likes['feature_1'] + ' + ' + df_likes['feature_2']
            
            # Create bars
            bars = ax.bar(range(len(df_likes)), df_likes['synergy_percentage'])
            
            # Customize plot
            ax.set_xlabel('Feature Combinations')
            ax.set_ylabel('Synergy Effect (%)')
            ax.set_title('Top 6 Feature Synergies: Combined Effects Greater Than Sum', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(df_likes)))
            ax.set_xticklabels(df_likes['combination'], rotation=45, ha='right')
            
            # Add value labels
            for bar, value in zip(bars, df_likes['synergy_percentage']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'synergy_analysis_2x2.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating synergy analysis plot: {e}")
    
    def plot_co_occurrence_heatmap(self):
        """Plot co-occurrence heatmap"""
        try:
            corr_file = self.output_dir / 'co_occurrence' / 'feature_correlations.csv'
            if not corr_file.exists():
                return
                
            corr_matrix = pd.read_csv(corr_file, index_col=0)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                       square=True, fmt='.2f', cbar_kws={'label': 'Spearman Correlation'})
            
            ax.set_title('Feature Co-occurrence Patterns', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'plots' / 'co_occurrence_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating co-occurrence heatmap: {e}")
    
    def plot_partial_correlations(self):
        """Plot partial correlations comparison"""
        try:
            likes_file = self.output_dir / 'partial_correlations' / 'likes_per_month_partial_correlations.csv'
            