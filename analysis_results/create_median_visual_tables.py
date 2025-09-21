#!/usr/bin/env python3
"""
Create beautiful visual comparison tables using median values
(Same style as the existing average tables but with median calculations)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ast
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MedianVisualTableCreator:
    def __init__(self, data_path):
        """Initialize with the dataset."""
        print("🔍 Loading dataset for median visual tables...")
        self.df = pd.read_csv(data_path, sep=';', encoding='utf-8', decimal=',')
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], format='%d/%m/%Y %H:%M')
        
        # Filter out prompts less than 3 months old
        cutoff_date = datetime.now() - timedelta(days=90)
        self.df = self.df[self.df['created_at'] <= cutoff_date]
        
        # Calculate total engagement
        self.df['total_engagement'] = (
            self.df['like_count'] + self.df['heart_count'] +
            self.df['comment_count'] + self.df['laugh_count'] + self.df['cry_count']
        )
        
        # Categories to analyze
        self.categories = [
            'subjects', 'style_modifiers', 'quality_boosters',
            'camera_composition', 'lighting_color', 'artists',
            'actions_verbs', 'style_codes', 'negative_terms', 'weights'
        ]
        
        # Create output directory
        self.output_dir = Path('analysis_results/category_comparison')
        self.output_dir.mkdir(exist_ok=True)
        
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

    def create_visual_median_table(self, metric_col, metric_name, output_filename):
        """Create a beautiful visual table using median values."""
        print(f"   Creating {metric_name} median table...")
        
        # Prepare data
        table_data = []
        
        for category in self.categories:
            has_cat = self.has_category(category)
            
            with_category = self.df[has_cat]
            without_category = self.df[~has_cat]
            
            if len(with_category) > 0 and len(without_category) > 0:
                with_median = with_category[metric_col].median()
                without_median = without_category[metric_col].median()
                
                # Calculate percentage change
                if without_median != 0:
                    pct_change = ((with_median - without_median) / without_median) * 100
                else:
                    pct_change = 0
                
                # Add rows for this category
                table_data.append({
                    'Category': category.replace('_', ' ').title(),
                    'Group': f'Has {category.replace("_", " ").title()}',
                    'Count': len(with_category),
                    'Median_Value': with_median,
                    'Percentage_Change': pct_change,
                    'Group_Type': 'has'
                })
                
                table_data.append({
                    'Category': category.replace('_', ' ').title(),
                    'Group': f'No {category.replace("_", " ").title()}',
                    'Count': len(without_category),
                    'Median_Value': without_median,
                    'Percentage_Change': -pct_change,  # Opposite sign
                    'Group_Type': 'no'
                })
        
        # Create the visual table
        self.create_beautiful_table_image(table_data, metric_name, output_filename)

    def create_beautiful_table_image(self, table_data, metric_name, output_filename):
        """Create a beautiful table image matching the style of the original."""
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 5)
        ax.set_ylim(0, len(table_data) + 1)
        ax.axis('off')
        
        # Define colors
        header_color = '#4472C4'  # Blue header
        row_colors = ['#FFFFFF', '#F2F2F2']  # Alternating white and light gray
        has_color = '#70AD47'  # Green for "Has" groups
        no_color = '#C5504B'   # Red for "No" groups
        
        # Column definitions
        cols = [
            ('Category', 0.8),
            ('Group', 1.2), 
            ('Count', 0.6),
            (f'Median Value', 1.0),
            ('Percentage Change', 1.4)
        ]
        
        col_positions = []
        x_start = 0
        for col_name, width in cols:
            col_positions.append((x_start, width))
            x_start += width
        
        # Draw header
        y_pos = len(table_data)
        header_rect = patches.Rectangle((0, y_pos), sum(width for _, width in cols), 1, 
                                       facecolor=header_color, edgecolor='black', linewidth=1)
        ax.add_patch(header_rect)
        
        # Add header text
        for i, ((col_name, _), (x_pos, width)) in enumerate(zip(cols, col_positions)):
            ax.text(x_pos + width/2, y_pos + 0.5, col_name, 
                   ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        # Draw data rows
        for row_idx, row_data in enumerate(table_data):
            y_pos = len(table_data) - row_idx - 1
            
            # Alternate row colors
            row_color = row_colors[row_idx % 2]
            
            # Draw row background
            row_rect = patches.Rectangle((0, y_pos), sum(width for _, width in cols), 1,
                                       facecolor=row_color, edgecolor='black', linewidth=0.5)
            ax.add_patch(row_rect)
            
            # Add cell content
            cell_data = [
                row_data['Category'],
                row_data['Group'],
                f"{row_data['Count']:,}",
                f"{row_data['Median_Value']:.1f}",
                f"{row_data['Percentage_Change']:+.1f}%"
            ]
            
            for i, ((col_name, _), (x_pos, width), content) in enumerate(zip(cols, col_positions, cell_data)):
                # Determine text color
                if i == 1:  # Group column
                    text_color = has_color if row_data['Group_Type'] == 'has' else no_color
                elif i == 4:  # Percentage change column
                    text_color = has_color if row_data['Percentage_Change'] > 0 else no_color
                else:
                    text_color = 'black'
                
                # Add text
                fontweight = 'bold' if i == 1 or i == 4 else 'normal'
                ax.text(x_pos + width/2, y_pos + 0.5, content,
                       ha='center', va='center', fontsize=11, fontweight=fontweight, color=text_color)
        
        # Add title
        plt.suptitle(f'{metric_name} Category Comparison (Median Values)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Save the image
        plt.savefig(self.output_dir / output_filename, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

    def create_all_median_visual_tables(self):
        """Create all 4 median-based visual tables."""
        print("\n📊 Creating beautiful median comparison tables...")
        
        # Define the 4 metrics and their corresponding files
        metrics = [
            ('like_count', 'Median Like Count', 'median_like_count_comparison_table.png'),
            ('total_engagement', 'Median Total Engagement', 'median_total_engagement_comparison_table.png'),
            ('likes_per_month', 'Median Likes Per Month', 'median_likes_per_month_comparison_table.png'),
            ('reactions_per_month', 'Median Reactions Per Month', 'median_reactions_per_month_comparison_table.png')
        ]
        
        for metric_col, metric_name, output_filename in metrics:
            self.create_visual_median_table(metric_col, metric_name, output_filename)
        
        print("✅ All 4 median visual tables created!")
        print("Generated files:")
        for _, metric_name, output_filename in metrics:
            print(f"   • {output_filename}")

def main():
    """Main execution function."""
    data_path = 'final_dataset/complete_analysis_py_adjusted_csv_normalized.csv'
    
    try:
        creator = MedianVisualTableCreator(data_path)
        creator.create_all_median_visual_tables()
        
        print("\n✅ MEDIAN VISUAL TABLES COMPLETE!")
        print("📁 All files saved to: analysis_results/category_comparison/")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file at {data_path}")
        print("Please ensure the dataset file exists in the correct location.")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()