#!/usr/bin/env python3
"""
Create 4 improved visual tables with proper formatting and percentage columns.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def create_improved_tables():
    """Create 4 separate tables for each engagement metric."""
    
    # Load the comparison data
    df_comparison = pd.read_csv('analysis_results/category_comparison/category_comparison_table.csv')
    df_summary = pd.read_csv('analysis_results/category_comparison/category_summary_statistics.csv')
    
    # Create output directory
    output_dir = Path('analysis_results/category_comparison')
    output_dir.mkdir(exist_ok=True)
    
    # Define the 4 metrics and their percentage columns
    metrics = [
        ('Mean Like Count', 'Like Count_Percentage_Change', 'Raw Likes Comparison'),
        ('Mean Total Reactions', 'Total Reactions_Percentage_Change', 'Raw Total Engagement Comparison'),
        ('Mean Likes Per Month', 'Likes Per Month_Percentage_Change', 'Normalized Likes Comparison'),
        ('Mean Reactions Per Month', 'Reactions Per Month_Percentage_Change', 'Normalized Engagement Comparison')
    ]
    
    for metric_col, pct_col, title in metrics:
        create_single_table(df_comparison, df_summary, metric_col, pct_col, title, output_dir)
    
    print("✅ All 4 improved visual tables created successfully!")
    print("📁 Files saved:")
    print("   • raw_likes_table.png")
    print("   • raw_engagement_table.png") 
    print("   • normalized_likes_table.png")
    print("   • normalized_engagement_table.png")

def create_single_table(df_comparison, df_summary, metric_col, pct_col, title, output_dir):
    """Create a single table for one metric."""
    
    # Prepare data for this metric
    table_data = []
    
    # Get percentage changes from summary data
    pct_dict = dict(zip(df_summary['Category'], df_summary[pct_col]))
    
    categories = df_comparison['Category'].unique()
    
    for i, category in enumerate(categories):
        cat_data = df_comparison[df_comparison['Category'] == category]
        
        # Get data for "Has" and "No" groups
        has_row = cat_data[cat_data['Group'].str.contains('Has')].iloc[0]
        no_row = cat_data[cat_data['Group'].str.contains('No')].iloc[0]
        
        # Get percentage change for this category
        pct_change = pct_dict.get(category, 0)
        
        # Add rows with alternating backgrounds
        has_group = 'Has ' + category.replace('_', ' ')
        no_group = 'No ' + category.replace('_', ' ')
        
        table_data.append([
            category.replace('_', ' ').title(),
            has_group,
            f"{has_row['Count']:,}",
            f"{has_row[metric_col]:.1f}",
            f"+{pct_change:.1f}%" if pct_change > 0 else f"{pct_change:.1f}%"
        ])
        
        table_data.append([
            '',  # Empty category for second row
            no_group,
            f"{no_row['Count']:,}",
            f"{no_row[metric_col]:.1f}",
            f"-{abs(pct_change):.1f}%" if pct_change > 0 else f"+{abs(pct_change):.1f}%"
        ])
    
    # Create figure without title
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create alternating gray backgrounds
    colors = []
    for i in range(len(table_data)):
        if i % 4 < 2:  # First pair of rows for each category
            if i % 2 == 0:
                colors.append(['#f5f5f5', '#f5f5f5', '#f5f5f5', '#f5f5f5', '#f5f5f5'])  # Light gray
            else:
                colors.append(['#e8e8e8', '#e8e8e8', '#e8e8e8', '#e8e8e8', '#e8e8e8'])  # Darker gray
        else:  # Second pair of rows for each category
            if i % 2 == 0:
                colors.append(['#f5f5f5', '#f5f5f5', '#f5f5f5', '#f5f5f5', '#f5f5f5'])  # Light gray
            else:
                colors.append(['#e8e8e8', '#e8e8e8', '#e8e8e8', '#e8e8e8', '#e8e8e8'])  # Darker gray
    
    # Create the table
    table = ax.table(cellText=table_data,
                    colLabels=['Category', 'Group', 'Count', 'Average Value', 'Percentage Change'],
                    cellLoc='center',
                    loc='center',
                    cellColours=colors)
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#1976d2')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.08)
    
    # Style data cells
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            table[(i, j)].set_height(0.06)
            if j == 0:  # Category column
                if table_data[i-1][j]:  # Only for non-empty category cells
                    table[(i, j)].set_text_props(weight='bold')
            elif j == 1:  # Group column
                if 'Has' in table_data[i-1][j]:
                    table[(i, j)].set_text_props(color='#2e7d32', weight='bold')  # Green for "Has"
                else:
                    table[(i, j)].set_text_props(color='#d32f2f', weight='bold')  # Red for "No"
            elif j == 4:  # Percentage column
                val_str = table_data[i-1][j]
                if val_str.startswith('+'):
                    table[(i, j)].set_text_props(color='#2e7d32', weight='bold')  # Green for positive
                else:
                    table[(i, j)].set_text_props(color='#d32f2f', weight='bold')  # Red for negative
    
    # Save the table
    filename = title.lower().replace(' ', '_').replace(':', '') + '_table.png'
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    create_improved_tables()