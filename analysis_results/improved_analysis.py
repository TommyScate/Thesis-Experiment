#!/usr/bin/env python3
"""
Improved Descriptive Analysis for Text-to-Image Prompt Dataset
Includes time normalization and refined visualizations based on feedback.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import ast
import json
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up beautiful plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class ImprovedPromptAnalyzer:
    def __init__(self, data_path):
        """Initialize the analyzer with time filtering."""
        print("🔍 Loading dataset...")
        self.df = pd.read_csv(data_path, sep=';', encoding='utf-8', decimal=',')
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], format='%d/%m/%Y %H:%M')
        
        # Filter out prompts less than 3 months old
        cutoff_date = datetime.now() - timedelta(days=90)
        initial_count = len(self.df)
        self.df = self.df[self.df['created_at'] <= cutoff_date]
        filtered_count = len(self.df)
        print(f"📅 Filtered out {initial_count - filtered_count:,} prompts less than 3 months old")
        print(f"📊 Remaining dataset: {filtered_count:,} prompts")
        
        # Calculate months since posting
        current_time = pd.Timestamp.now()
        self.df['months_since_posting'] = self.df['created_at'].apply(lambda x: (current_time - x).days / 30.44)
        
        # Parse list columns
        self.list_columns = [
            'subjects', 'style_modifiers', 'quality_boosters',
            'camera_composition', 'lighting_color', 'artists',
            'actions_verbs', 'style_codes', 'negative_terms'
        ]
        
        print("📊 Processing AI categories...")
        for col in self.list_columns:
            self.df[col] = self.df[col].apply(self.safe_eval)
        
        # Calculate engagement metrics using pre-computed normalized values
        self.df['total_engagement'] = (
            self.df['like_count'] + self.df['heart_count'] +
            self.df['comment_count'] + self.df['laugh_count'] + self.df['cry_count']
        )
        
        # Use pre-computed normalized metrics from CSV (already computed with proper normalization)
        # likes_per_month and reactions_per_month columns are available
        self.df['engagement_per_month'] = self.df['reactions_per_month']
        
        print(f"✅ Dataset loaded: {len(self.df):,} prompts (3+ months old)")
        
        # Create output directories
        self.plots_dir = Path('analysis_results/improved_plots')
        self.stats_dir = Path('analysis_results/improved_statistics')
        self.plots_dir.mkdir(exist_ok=True)
        self.stats_dir.mkdir(exist_ok=True)
    
    def safe_eval(self, x):
        """Safely evaluate string representation of lists."""
        if pd.isna(x) or x == '[]':
            return []
        try:
            return ast.literal_eval(x)
        except:
            return []
    
    def basic_statistics(self):
        """Generate basic dataset statistics."""
        print("\n📈 PHASE 1: BASIC STATISTICS")
        print("=" * 50)
        
        stats = {
            'total_prompts': len(self.df),
            'date_range': {
                'earliest': self.df['created_at'].min().strftime('%Y-%m-%d'),
                'latest': self.df['created_at'].max().strftime('%Y-%m-%d'),
                'span_days': int((self.df['created_at'].max() - self.df['created_at'].min()).days)
            },
            'time_filtering': {
                'cutoff_date': (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                'average_months_since_posting': float(self.df['months_since_posting'].mean()),
                'median_months_since_posting': float(self.df['months_since_posting'].median())
            },
            'negative_prompt_usage': {
                'with_negative': int(self.df['has_negative'].sum()),
                'without_negative': int((~self.df['has_negative']).sum()),
                'percentage_with_negative': float(self.df['has_negative'].mean() * 100)
            }
        }
        
        print(f"📅 Date Range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
        print(f"⏱️  Time Span: {stats['date_range']['span_days']} days")
        print(f"📆 Average Age: {stats['time_filtering']['average_months_since_posting']:.1f} months")
        print(f"➖ Negative Prompts: {stats['negative_prompt_usage']['percentage_with_negative']:.1f}%")
        
        # Save stats
        with open(self.stats_dir / 'basic_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        return stats
    
    def engagement_analysis(self):
        """Analyze engagement metrics - both raw and time-normalized."""
        print("\n💫 PHASE 2: ENGAGEMENT METRICS ANALYSIS")
        print("=" * 50)
        
        # Focus on likes and total engagement (raw and normalized)
        key_metrics = ['like_count', 'total_engagement', 'likes_per_month', 'engagement_per_month']
        
        # Calculate descriptive statistics
        engagement_stats = self.df[key_metrics].describe()
        print("📊 Key Engagement Statistics:")
        print(engagement_stats.round(2))
        
        # Create engagement distribution plots (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        titles = ['Raw Likes', 'Raw Total Engagement', 'Likes per Month', 'Engagement per Month']
        
        for i, (col, color, title) in enumerate(zip(key_metrics, colors, titles)):
            row, col_idx = i // 2, i % 2
            
            # Use log scale for better visualization
            data = self.df[col][self.df[col] > 0]  # Remove zeros for log scale
            
            axes[row, col_idx].hist(data, bins=50, alpha=0.7, color=color, edgecolor='white')
            axes[row, col_idx].set_title(title, fontsize=14, fontweight='bold')
            axes[row, col_idx].set_xlabel('Count')
            axes[row, col_idx].set_ylabel('Frequency')
            axes[row, col_idx].set_yscale('log')
            axes[row, col_idx].grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            axes[row, col_idx].axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                                     label=f'Mean: {mean_val:.1f}')
            axes[row, col_idx].axvline(median_val, color='orange', linestyle='--', alpha=0.8, 
                                     label=f'Median: {median_val:.1f}')
            axes[row, col_idx].legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'engagement_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save engagement stats
        engagement_stats.to_csv(self.stats_dir / 'engagement_statistics.csv')
        
        return engagement_stats
    
    def category_analysis_improved(self):
        """Improved category analysis - averages only for prompts using each category."""
        print("\n🔤 PHASE 3: IMPROVED CATEGORY ANALYSIS")
        print("=" * 50)
        
        # Text characteristics
        text_stats = {
            'prompt_length': {
                'mean_words': float(self.df['prompt_word_count'].mean()),
                'median_words': float(self.df['prompt_word_count'].median())
            }
        }
        
        print(f"📝 Average prompt: {text_stats['prompt_length']['mean_words']:.1f} words")
        
        # IMPROVED category analysis - only for prompts that use each category
        category_stats = {}
        for col in self.list_columns:
            count_col = f"{col}_count"
            if count_col in self.df.columns:
                # Filter to only prompts that use this category
                using_category = self.df[self.df[count_col] > 0]
                
                if len(using_category) > 0:
                    category_stats[col] = {
                        'mean_count_among_users': float(using_category[count_col].mean()),
                        'median_count_among_users': float(using_category[count_col].median()),
                        'max_count': int(self.df[count_col].max()),
                        'usage_percentage': float((self.df[count_col] > 0).mean() * 100),
                        'total_users': int(len(using_category))
                    }
                    print(f"📊 {col}: {category_stats[col]['usage_percentage']:.1f}% use it, avg {category_stats[col]['mean_count_among_users']:.1f} terms when used")
        
        # Create improved category visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Average categories per prompt (among users only)
        category_names = [col.replace('_', ' ').title() for col in self.list_columns]
        mean_counts_users = [category_stats[col]['mean_count_among_users'] for col in self.list_columns]
        usage_percentages = [category_stats[col]['usage_percentage'] for col in self.list_columns]
        
        bars1 = ax1.bar(range(len(category_names)), mean_counts_users, 
                       color=sns.color_palette("husl", len(category_names)))
        ax1.set_title('Average Terms per Category\n(Among Prompts Using Each Category)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Category')
        ax1.set_ylabel('Average Count (Users Only)')
        ax1.set_xticks(range(len(category_names)))
        ax1.set_xticklabels(category_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, mean_counts_users):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Usage percentage
        bars2 = ax2.bar(range(len(category_names)), usage_percentages, 
                       color=sns.color_palette("viridis", len(category_names)))
        ax2.set_title('Category Usage Percentage', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Category')
        ax2.set_ylabel('Usage Percentage (%)')
        ax2.set_xticks(range(len(category_names)))
        ax2.set_xticklabels(category_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, usage_percentages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'category_analysis_improved.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Prompt length distribution (words only)
        plt.figure(figsize=(12, 6))
        
        plt.hist(self.df['prompt_word_count'], bins=50, alpha=0.7, color='skyblue', edgecolor='white')
        plt.title('Prompt Length Distribution (Words)', fontsize=16, fontweight='bold')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.axvline(self.df['prompt_word_count'].mean(), color='red', linestyle='--', alpha=0.8, 
                   label=f'Mean: {self.df["prompt_word_count"].mean():.1f}')
        plt.axvline(self.df['prompt_word_count'].median(), color='orange', linestyle='--', alpha=0.8,
                   label=f'Median: {self.df["prompt_word_count"].median():.1f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'prompt_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save stats
        combined_stats = {**text_stats, 'categories': category_stats}
        with open(self.stats_dir / 'category_stats_improved.json', 'w') as f:
            json.dump(combined_stats, f, indent=2)
        
        return combined_stats
    
    def top_terms_analysis_improved(self):
        """Improved top terms analysis with better spacing."""
        print("\n🏆 PHASE 4: TOP TERMS ANALYSIS")
        print("=" * 50)
        
        top_terms = {}
        
        for col in self.list_columns:
            if col in self.df.columns:
                # Flatten all terms from this category
                all_terms = []
                for term_list in self.df[col]:
                    if isinstance(term_list, list):
                        for term in term_list:
                            if isinstance(term, str):
                                all_terms.append(term)
                    elif isinstance(term_list, str):
                        all_terms.append(term_list)
                
                # Count frequency
                term_counts = Counter(all_terms)
                top_terms[col] = dict(term_counts.most_common(20))
                
                print(f"\n📊 Top 10 {col.replace('_', ' ').title()}:")
                for term, count in term_counts.most_common(10):
                    print(f"  {term}: {count:,}")
        
        # Create improved visualizations with better spacing
        fig, axes = plt.subplots(3, 3, figsize=(24, 20))  # Wider figure
        axes = axes.flatten()
        
        for i, col in enumerate(self.list_columns):
            if i >= len(axes):
                break
                
            if col in top_terms and top_terms[col]:
                terms = list(top_terms[col].keys())[:10]
                counts = list(top_terms[col].values())[:10]
                
                # Create horizontal bar plot with more space
                bars = axes[i].barh(range(len(terms)), counts, 
                                  color=sns.color_palette("husl", len(terms)))
                axes[i].set_title(f'Top {col.replace("_", " ").title()}', 
                                fontsize=13, fontweight='bold')
                axes[i].set_yticks(range(len(terms)))
                axes[i].set_yticklabels(terms, fontsize=10)
                axes[i].set_xlabel('Frequency')
                axes[i].grid(True, alpha=0.3)
                
                # Add value labels with better spacing
                max_count = max(counts)
                for j, (bar, count) in enumerate(zip(bars, counts)):
                    # Position text further from bar end
                    text_x = bar.get_width() + max_count * 0.02
                    axes[i].text(text_x, bar.get_y() + bar.get_height()/2,
                               f'{count:,}', ha='left', va='center', fontweight='bold', fontsize=9)
                
                # Invert y-axis and adjust margins
                axes[i].invert_yaxis()
                axes[i].set_xlim(0, max_count * 1.15)  # Extra space for labels
        
        # Remove empty subplots
        for i in range(len(self.list_columns), len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'top_terms_analysis_improved.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save top terms data
        with open(self.stats_dir / 'top_terms.json', 'w') as f:
            json.dump(top_terms, f, indent=2)
        
        return top_terms
    
    def engagement_vs_features_with_trends(self):
        """Analyze engagement vs features (4 separate analyses without trend lines)."""
        print("\n📈 PHASE 5: ENGAGEMENT VS FEATURES ANALYSIS")
        print("=" * 50)
        
        # All category count columns
        feature_cols = ['prompt_word_count'] + [f"{col}_count" for col in self.list_columns if f"{col}_count" in self.df.columns]
        
        # Create comprehensive scatter plots with trend lines
        n_features = len(feature_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # 1. Raw Total Engagement vs Features
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Raw Total Engagement vs Features', fontsize=16, fontweight='bold', y=0.95)
        
        for i, feature in enumerate(feature_cols):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            x_data = self.df[feature]
            y_data = self.df['total_engagement']
            
            ax.scatter(x_data, y_data, alpha=0.6, color='steelblue', s=20)
            ax.set_title(f'{feature.replace("_", " ").title()} vs Total Engagement', fontweight='bold', pad=10)
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Total Engagement')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(feature_cols), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].remove()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'raw_engagement_vs_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Raw Likes vs Features
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Raw Likes vs Features', fontsize=16, fontweight='bold', y=0.95)
        
        for i, feature in enumerate(feature_cols):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            x_data = self.df[feature]
            y_data = self.df['like_count']
            
            ax.scatter(x_data, y_data, alpha=0.6, color='crimson', s=20)
            ax.set_title(f'{feature.replace("_", " ").title()} vs Likes', fontweight='bold', pad=10)
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Like Count')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(feature_cols), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].remove()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'raw_likes_vs_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Normalized Total Engagement vs Features
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Normalized Total Engagement vs Features', fontsize=16, fontweight='bold', y=0.95)
        
        for i, feature in enumerate(feature_cols):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            x_data = self.df[feature]
            y_data = self.df['engagement_per_month']
            
            ax.scatter(x_data, y_data, alpha=0.6, color='forestgreen', s=20)
            ax.set_title(f'{feature.replace("_", " ").title()} vs Engagement per Month', fontweight='bold', pad=10)
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Engagement per Month')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(feature_cols), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].remove()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'normalized_engagement_vs_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Normalized Likes vs Features
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Normalized Likes vs Features', fontsize=16, fontweight='bold', y=0.95)
        
        for i, feature in enumerate(feature_cols):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            x_data = self.df[feature]
            y_data = self.df['likes_per_month']
            
            ax.scatter(x_data, y_data, alpha=0.6, color='darkorange', s=20)
            ax.set_title(f'{feature.replace("_", " ").title()} vs Likes per Month', fontweight='bold', pad=10)
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Likes per Month')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(feature_cols), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].remove()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'normalized_likes_vs_features.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Engagement vs features analysis complete!")
        return feature_cols
    
    def generate_improved_summary(self):
        """Generate improved summary with better layout."""
        print("\n📋 GENERATING IMPROVED SUMMARY REPORT")
        print("=" * 50)
        
        summary = {
            'dataset_overview': {
                'total_prompts': len(self.df),
                'date_range': f"{self.df['created_at'].min().strftime('%Y-%m-%d')} to {self.df['created_at'].max().strftime('%Y-%m-%d')}",
                'average_months_old': float(self.df['months_since_posting'].mean()),
                'raw_metrics': {
                    'average_likes': float(self.df['like_count'].mean()),
                    'average_total_engagement': float(self.df['total_engagement'].mean())
                },
                'normalized_metrics': {
                    'average_likes_per_month': float(self.df['likes_per_month'].mean()),
                    'average_engagement_per_month': float(self.df['engagement_per_month'].mean())
                }
            }
        }
        
        # Save summary
        with open(self.stats_dir / 'improved_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create improved summary visualization (no language dist, better spacing)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Weights usage pie chart
        weights_counts = self.df['weights'].value_counts()
        weights_labels = ['Uses Weights' if x else 'No Weights' for x in weights_counts.index]
        weights_colors = ['#66b3ff', '#ff9999']
        
        axes[0, 0].pie(weights_counts.values, labels=weights_labels, autopct='%1.1f%%',
                      colors=weights_colors, startangle=90)
        axes[0, 0].set_title('Weight Usage Distribution', fontweight='bold')
        
        # Raw vs normalized engagement comparison (with taller y-axis)
        raw_avg = self.df['total_engagement'].mean()
        norm_avg = self.df['engagement_per_month'].mean()
        bars = axes[0, 1].bar(['Raw Total\nEngagement', 'Engagement\nper Month'],
                             [raw_avg, norm_avg], color=['lightblue', 'orange'])
        axes[0, 1].set_title('Raw vs Time-Normalized Engagement', fontweight='bold')
        axes[0, 1].set_ylabel('Average Engagement')
        # Fix y-axis limit to prevent cutoff
        y_max = max(raw_avg, norm_avg) * 1.15
        axes[0, 1].set_ylim(0, y_max)
        
        # Add value labels with better spacing
        for bar, value in zip(bars, [raw_avg, norm_avg]):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_max * 0.02,
                           f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Category usage overview
        categories = [col.replace('_', '\n') for col in self.list_columns]
        usage_pcts = [(self.df[f"{col}_count"] > 0).mean() * 100 for col in self.list_columns]
        bars = axes[0, 2].bar(range(len(categories)), usage_pcts, 
                             color=sns.color_palette("husl", len(categories)))
        axes[0, 2].set_title('Category Usage %', fontweight='bold')
        axes[0, 2].set_xticks(range(len(categories)))
        axes[0, 2].set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
        axes[0, 2].set_ylabel('Usage Percentage')
        
        # Monthly posting trend
        monthly_posts = self.df.groupby(self.df['created_at'].dt.to_period('M')).size()
        monthly_posts.plot(kind='line', color='steelblue', linewidth=2, ax=axes[1, 0])
        axes[1, 0].set_title('Monthly Posting Trend', fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Number of Posts')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Time normalization impact (with taller y-axis)
        raw_likes = self.df['like_count'].mean()
        norm_likes = self.df['likes_per_month'].mean()
        bars = axes[1, 1].bar(['Raw Likes', 'Likes per Month'],
                             [raw_likes, norm_likes], color=['skyblue', 'lightgreen'])
        axes[1, 1].set_title('Time Normalization Impact', fontweight='bold')
        axes[1, 1].set_ylabel('Average Likes')
        # Fix y-axis limit to prevent cutoff
        y_max_likes = max(raw_likes, norm_likes) * 1.15
        axes[1, 1].set_ylim(0, y_max_likes)
        
        # Add value labels
        for bar, value in zip(bars, [raw_likes, norm_likes]):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_max_likes * 0.02,
                           f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Negative prompt usage pie chart
        neg_counts = self.df['has_negative'].value_counts()
        neg_labels = ['With Negative', 'Without Negative']
        neg_colors = ['#ff7f7f', '#87ceeb']
        
        axes[1, 2].pie(neg_counts.values, labels=neg_labels, autopct='%1.1f%%',
                      colors=neg_colors, startangle=90)
        axes[1, 2].set_title('Negative Prompt Usage', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'improved_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Improved summary report generated!")
        return summary
    

def main():
    """Run improved comprehensive descriptive analysis."""
    print("🎯 STARTING IMPROVED DESCRIPTIVE ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ImprovedPromptAnalyzer('final_dataset/complete_analysis_py_adjusted_csv_normalized.csv')
    
    # Run all analysis phases
    basic_stats = analyzer.basic_statistics()
    engagement_stats = analyzer.engagement_analysis()
    category_stats = analyzer.category_analysis_improved()
    top_terms = analyzer.top_terms_analysis_improved()
    features = analyzer.engagement_vs_features_with_trends()
    summary = analyzer.generate_improved_summary()
    
    print("\n🎉 IMPROVED ANALYSIS COMPLETE!")
    print("=" * 60)
    print("📁 Results saved in:")
    print("   📊 Plots: analysis_results/improved_plots/")
    print("   📈 Statistics: analysis_results/improved_statistics/")
    print("\n📋 Generated files:")
    print("   • engagement_distributions.png (Raw + Time-normalized)")
    print("   • category_analysis_improved.png")
    print("   • top_terms_analysis_improved.png (Better spacing)")
    print("   • raw_engagement_vs_features.png (Total engagement vs features)")
    print("   • raw_likes_vs_features.png (Likes vs features)")
    print("   • normalized_engagement_vs_features.png (Engagement per month)")
    print("   • normalized_likes_vs_features.png (Likes per month)")
    print("   • prompt_length_distribution.png (Words only)")
    print("   • improved_summary.png (With weights pie chart)")
    print("   • Various improved CSV and JSON files")
    
    print(f"\n📊 Dataset: {len(analyzer.df):,} prompts (3+ months old)")
    print(f"📅 Average age: {analyzer.df['months_since_posting'].mean():.1f} months")

if __name__ == "__main__":
    main()