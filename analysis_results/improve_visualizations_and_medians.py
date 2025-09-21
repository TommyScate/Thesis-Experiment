#!/usr/bin/env python3
"""
Improve visualizations and add median-based analysis
1. Add median charts to improved summary
2. Create median-based category comparison tables
3. Fix overlapping titles in engagement vs features charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up beautiful plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class VisualizationImprover:
    def __init__(self, data_path):
        """Initialize with the dataset."""
        print("🔍 Loading dataset for visualization improvements...")
        self.df = pd.read_csv(data_path, sep=';', encoding='utf-8', decimal=',')
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], format='%d/%m/%Y %H:%M')
        
        # Filter out prompts less than 3 months old (same as improved_analysis.py)
        cutoff_date = datetime.now() - timedelta(days=90)
        self.df = self.df[self.df['created_at'] <= cutoff_date]
        
        # Calculate months since posting
        current_time = pd.Timestamp.now()
        self.df['months_since_posting'] = self.df['created_at'].apply(lambda x: (current_time - x).days / 30.44)
        
        # Calculate total engagement
        self.df['total_engagement'] = (
            self.df['like_count'] + self.df['heart_count'] +
            self.df['comment_count'] + self.df['laugh_count'] + self.df['cry_count']
        )
        
        # Categories and metrics
        self.categories = [
            'subjects', 'style_modifiers', 'quality_boosters',
            'camera_composition', 'lighting_color', 'artists',
            'actions_verbs', 'style_codes', 'negative_terms', 'weights'
        ]
        
        self.list_columns = [
            'subjects', 'style_modifiers', 'quality_boosters',
            'camera_composition', 'lighting_color', 'artists',
            'actions_verbs', 'style_codes', 'negative_terms'
        ]
        
        # Create output directories
        self.plots_dir = Path('analysis_results/improved_plots')
        self.tables_dir = Path('analysis_results/category_comparison')
        self.plots_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)
        
        print(f"✅ Dataset loaded: {len(self.df):,} prompts")

    def safe_eval(self, x):
        """Safely evaluate string representation of lists."""
        if pd.isna(x) or x == '[]':
            return []
        try:
            return ast.literal_eval(x)
        except:
            return []

    def has_category(self, category_name):
        """Determine if a prompt has a given category."""
        if category_name == 'weights':
            return self.df['weights'] == True
        else:
            count_col = f"{category_name}_count"
            if count_col in self.df.columns:
                return self.df[count_col] > 0
            else:
                category_col = self.df[category_name].apply(self.safe_eval)
                return category_col.apply(lambda x: len(x) > 0)

    def add_median_charts_to_summary(self):
        """Add 2 median-based charts next to the existing average charts."""
        print("\n📊 Creating median-based summary charts...")
        
        # Create the median charts only (2 new charts)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Raw vs normalized engagement comparison (MEDIAN)
        raw_median = self.df['total_engagement'].median()
        norm_median = self.df['reactions_per_month'].median()
        bars = axes[0].bar(['Raw Total\nEngagement', 'Engagement\nper Month'],
                          [raw_median, norm_median], color=['lightblue', 'orange'])
        axes[0].set_title('Raw vs Time-Normalized Engagement (Median)', fontweight='bold')
        axes[0].set_ylabel('Median Engagement')
        
        # Fix y-axis limit to prevent cutoff
        y_max = max(raw_median, norm_median) * 1.15
        axes[0].set_ylim(0, y_max)
        
        # Add value labels
        for bar, value in zip(bars, [raw_median, norm_median]):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_max * 0.02,
                        f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Time normalization impact (MEDIAN)
        raw_likes_median = self.df['like_count'].median()
        norm_likes_median = self.df['likes_per_month'].median()
        bars = axes[1].bar(['Raw Likes', 'Likes per Month'],
                          [raw_likes_median, norm_likes_median], color=['skyblue', 'lightgreen'])
        axes[1].set_title('Time Normalization Impact (Median)', fontweight='bold')
        axes[1].set_ylabel('Median Likes')
        
        # Fix y-axis limit to prevent cutoff
        y_max_likes = max(raw_likes_median, norm_likes_median) * 1.15
        axes[1].set_ylim(0, y_max_likes)
        
        # Add value labels
        for bar, value in zip(bars, [raw_likes_median, norm_likes_median]):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_max_likes * 0.02,
                        f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'median_summary_charts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Median summary charts created!")

    def create_median_category_comparison_tables(self):
        """Create 4 category comparison tables using median values."""
        print("\n📊 Creating median-based category comparison tables...")
        
        engagement_metrics = [
            ('like_count', 'Median Like Count'),
            ('total_engagement', 'Median Total Engagement'), 
            ('likes_per_month', 'Median Likes Per Month'),
            ('reactions_per_month', 'Median Reactions Per Month')
        ]
        
        for metric_col, metric_name in engagement_metrics:
            print(f"   Creating table for {metric_name}...")
            
            results = []
            
            for category in self.categories:
                # Determine which prompts have this category
                has_cat = self.has_category(category)
                
                # Split data into two groups
                with_category = self.df[has_cat]
                without_category = self.df[~has_cat]
                
                # Calculate statistics for each group (MEDIAN instead of MEAN)
                for group_name, group_data in [("Has " + category.replace('_', ' ').title(), with_category),
                                             ("No " + category.replace('_', ' ').title(), without_category)]:
                    
                    if len(group_data) > 0:
                        median_value = group_data[metric_col].median()
                        percentage_change = 0
                        
                        # Calculate percentage change
                        if group_name.startswith("Has"):
                            # This is the "with category" group
                            without_median = without_category[metric_col].median() if len(without_category) > 0 else 0
                            if without_median != 0:
                                percentage_change = ((median_value - without_median) / without_median) * 100
                        else:
                            # This is the "without category" group
                            with_median = with_category[metric_col].median() if len(with_category) > 0 else 0
                            if median_value != 0:
                                percentage_change = ((with_median - median_value) / median_value) * 100
                        
                        row = {
                            'Category': category.replace('_', ' ').title(),
                            'Group': group_name,
                            'Count': len(group_data),
                            'Median Value': round(median_value, 1),
                            'Percentage Change': f"{percentage_change:+.1f}%"
                        }
                        results.append(row)
                    else:
                        row = {
                            'Category': category.replace('_', ' ').title(),
                            'Group': group_name,
                            'Count': 0,
                            'Median Value': 0,
                            'Percentage Change': "0.0%"
                        }
                        results.append(row)
            
            # Convert to DataFrame and save
            comparison_df = pd.DataFrame(results)
            
            # Save as CSV and display format
            safe_metric_name = metric_name.lower().replace(' ', '_').replace('median_', '')
            csv_filename = f'median_category_comparison_{safe_metric_name}.csv'
            comparison_df.to_csv(self.tables_dir / csv_filename, index=False)
            
            print(f"     ✅ Saved {csv_filename}")

        print("✅ All median category comparison tables created!")

    def fix_overlapping_titles_in_engagement_charts(self):
        """Fix overlapping titles in the 4 engagement vs features charts."""
        print("\n🔧 Fixing overlapping titles in engagement vs features charts...")
        
        # All category count columns
        feature_cols = ['prompt_word_count'] + [f"{col}_count" for col in self.list_columns if f"{col}_count" in self.df.columns]
        
        # Calculate grid dimensions
        n_features = len(feature_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        chart_configs = [
            ('raw_engagement_vs_features_fixed.png', 'total_engagement', 'Total Engagement', 'steelblue'),
            ('raw_likes_vs_features_fixed.png', 'like_count', 'Like Count', 'crimson'),
            ('normalized_engagement_vs_features_fixed.png', 'reactions_per_month', 'Engagement per Month', 'forestgreen'),
            ('normalized_likes_vs_features_fixed.png', 'likes_per_month', 'Likes per Month', 'darkorange')
        ]
        
        for filename, y_column, y_label, color in chart_configs:
            print(f"   Fixing {filename}...")
            
            # Create figure with proper spacing
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            # MOVE TITLE HIGHER UP to avoid overlap
            fig.suptitle(f'{y_label} vs Features', fontsize=18, fontweight='bold', y=0.98)
            
            for i, feature in enumerate(feature_cols):
                row, col = i // n_cols, i % n_cols
                ax = axes[row, col]
                
                x_data = self.df[feature]
                y_data = self.df[y_column]
                
                ax.scatter(x_data, y_data, alpha=0.6, color=color, s=20)
                ax.set_title(f'{feature.replace("_", " ").title()} vs {y_label}', 
                           fontweight='bold', pad=15)  # Increased padding
                ax.set_xlabel(feature.replace('_', ' ').title())
                ax.set_ylabel(y_label)
                ax.grid(True, alpha=0.3)
            
            # Remove empty subplots
            for i in range(len(feature_cols), n_rows * n_cols):
                row, col = i // n_cols, i % n_cols
                axes[row, col].remove()
            
            # Use more spacing to prevent overlap
            plt.tight_layout(rect=(0, 0.03, 1, 0.94))
            plt.savefig(self.plots_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"     ✅ Fixed {filename}")
        
        print("✅ All engagement vs features charts fixed!")

    def run_all_improvements(self):
        """Run all visualization improvements."""
        print("🚀 Starting visualization improvements...")
        print("=" * 60)
        
        # 1. Add median charts to summary
        self.add_median_charts_to_summary()
        
        # 2. Create median category comparison tables
        self.create_median_category_comparison_tables()
        
        # 3. Fix overlapping titles in engagement charts
        self.fix_overlapping_titles_in_engagement_charts()
        
        print("\n✅ ALL VISUALIZATION IMPROVEMENTS COMPLETE!")
        print("=" * 60)
        print("📁 Generated files:")
        print("   📊 New plots:")
        print("      • median_summary_charts.png (2 median-based charts)")
        print("      • *_vs_features_fixed.png (4 charts with fixed titles)")
        print("   📋 New tables:")
        print("      • median_category_comparison_like_count.csv")
        print("      • median_category_comparison_total_engagement.csv") 
        print("      • median_category_comparison_likes_per_month.csv")
        print("      • median_category_comparison_reactions_per_month.csv")

def main():
    """Main execution function."""
    data_path = 'final_dataset/complete_analysis_py_adjusted_csv_normalized.csv'
    
    try:
        improver = VisualizationImprover(data_path)
        improver.run_all_improvements()
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file at {data_path}")
        print("Please ensure the dataset file exists in the correct location.")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()