#!/usr/bin/env python3
"""
Category Comparison Analysis for Text-to-Image Prompt Dataset
Creates a comparison table showing engagement metrics for prompts that include vs do not include each category.
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
import json

class CategoryComparisonAnalyzer:
    def __init__(self, data_path):
        """Initialize the analyzer."""
        print("🔍 Loading dataset for category comparison analysis...")
        self.df = pd.read_csv(data_path, sep=';', encoding='utf-8', decimal=',')
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], format='%d/%m/%Y %H:%M')
        
        print(f"📊 Dataset loaded: {len(self.df):,} prompts")
        
        # Define categories to analyze
        self.categories = [
            'subjects',
            'style_modifiers', 
            'quality_boosters',
            'camera_composition',
            'lighting_color',
            'artists',
            'actions_verbs',
            'style_codes',
            'negative_terms',
            'weights'
        ]
        
        # Define engagement metrics
        self.engagement_metrics = [
            'like_count',           # Raw likes
            'total_reactions',      # Raw total engagement  
            'likes_per_month',      # Normalized likes
            'reactions_per_month'   # Normalized total engagement
        ]
        
        # Create output directory
        self.output_dir = Path('analysis_results/category_comparison')
        self.output_dir.mkdir(exist_ok=True)
    
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
            # For weights, check if the weights column is True (has weights)
            return self.df['weights'] == True
        else:
            # For other categories, check if the count column > 0
            count_col = f"{category_name}_count"
            if count_col in self.df.columns:
                return self.df[count_col] > 0
            else:
                # Fallback: parse the actual category column
                category_col = self.df[category_name].apply(self.safe_eval)
                return category_col.apply(lambda x: len(x) > 0)
    
    def create_comparison_table(self):
        """Create the category comparison analysis table."""
        print("\n📊 Creating category comparison analysis table...")
        
        results = []
        
        for category in self.categories:
            print(f"   Analyzing {category}...")
            
            # Determine which prompts have this category
            has_cat = self.has_category(category)
            
            # Split data into two groups
            with_category = self.df[has_cat]
            without_category = self.df[~has_cat]
            
            # Calculate statistics for each group
            for group_name, group_data in [("Has " + category.replace('_', ' ').title(), with_category),
                                         ("No " + category.replace('_', ' ').title(), without_category)]:
                
                if len(group_data) > 0:
                    row = {
                        'Category': category.replace('_', ' ').title(),
                        'Group': group_name,
                        'Count': len(group_data)
                    }
                    
                    # Calculate mean values for each engagement metric
                    for metric in self.engagement_metrics:
                        if metric in group_data.columns:
                            mean_value = group_data[metric].mean()
                            row[f'Mean {metric.replace("_", " ").title()}'] = round(mean_value, 2)
                        else:
                            row[f'Mean {metric.replace("_", " ").title()}'] = 0
                    
                    results.append(row)
                else:
                    # Handle empty groups
                    row = {
                        'Category': category.replace('_', ' ').title(),
                        'Group': group_name,
                        'Count': 0
                    }
                    for metric in self.engagement_metrics:
                        row[f'Mean {metric.replace("_", " ").title()}'] = 0
                    results.append(row)
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Reorder columns for better readability
        column_order = ['Category', 'Group', 'Count'] + [f'Mean {metric.replace("_", " ").title()}' for metric in self.engagement_metrics]
        comparison_df = comparison_df[column_order]
        
        return comparison_df
    
    def calculate_category_summary(self):
        """Calculate summary statistics for each category."""
        print("\n📈 Calculating category summary statistics...")
        
        summary_results = []
        
        for category in self.categories:
            has_cat = self.has_category(category)
            
            with_category = self.df[has_cat]
            without_category = self.df[~has_cat]
            
            summary = {
                'Category': category.replace('_', ' ').title(),
                'Total_Prompts': len(self.df),
                'With_Category': len(with_category),
                'Without_Category': len(without_category),
                'Usage_Percentage': round((len(with_category) / len(self.df)) * 100, 2)
            }
            
            # Calculate engagement differences (with vs without)
            for metric in self.engagement_metrics:
                if metric in self.df.columns:
                    with_mean = with_category[metric].mean() if len(with_category) > 0 else 0
                    without_mean = without_category[metric].mean() if len(without_category) > 0 else 0
                    difference = with_mean - without_mean
                    percentage_change = ((with_mean - without_mean) / without_mean * 100) if without_mean != 0 else 0
                    
                    summary[f'{metric.replace("_", " ").title()}_Difference'] = round(difference, 2)
                    summary[f'{metric.replace("_", " ").title()}_Percentage_Change'] = round(percentage_change, 2)
            
            summary_results.append(summary)
        
        return pd.DataFrame(summary_results)
    
    def save_results(self, comparison_df, summary_df):
        """Save the analysis results."""
        print("\n💾 Saving results...")
        
        # Save comparison table
        comparison_df.to_csv(self.output_dir / 'category_comparison_table.csv', index=False)
        comparison_df.to_json(self.output_dir / 'category_comparison_table.json', orient='records', indent=2)
        
        # Save summary statistics
        summary_df.to_csv(self.output_dir / 'category_summary_statistics.csv', index=False)
        summary_df.to_json(self.output_dir / 'category_summary_statistics.json', orient='records', indent=2)
        
        print(f"✅ Results saved to {self.output_dir}")
        
        return comparison_df, summary_df
    
    def display_results(self, comparison_df, summary_df):
        """Display the results in a readable format."""
        print("\n" + "="*80)
        print("📊 CATEGORY COMPARISON ANALYSIS RESULTS")
        print("="*80)
        
        print("\n🔍 MAIN COMPARISON TABLE")
        print("-" * 50)
        print("Showing engagement metrics for prompts that include vs do not include each category:")
        print("\nColumns explained:")
        print("- Like Count: Raw number of likes")
        print("- Total Reactions: Raw total engagement (likes + hearts + comments + etc.)")
        print("- Likes Per Month: Time-normalized likes")
        print("- Reactions Per Month: Time-normalized total engagement")
        print()
        
        # Display the main table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(comparison_df.to_string(index=False))
        
        print("\n\n📈 CATEGORY SUMMARY STATISTICS")
        print("-" * 50)
        print("Summary showing usage percentages and engagement differences:")
        print()
        print(summary_df.to_string(index=False))
        
        print("\n\n📋 KEY INSIGHTS:")
        print("-" * 20)
        
        # Find categories with highest positive impact
        best_categories = []
        for _, row in summary_df.iterrows():
            like_change = row.get('Like Count Percentage Change', 0)
            if like_change > 10:  # More than 10% improvement
                best_categories.append((row['Category'], like_change))
        
        if best_categories:
            best_categories.sort(key=lambda x: x[1], reverse=True)
            print("🚀 Categories with highest positive impact on likes:")
            for cat, change in best_categories[:5]:
                print(f"   • {cat}: +{change:.1f}% improvement")
        
        # Find most/least used categories
        most_used = summary_df.loc[summary_df['Usage_Percentage'].idxmax()]
        least_used = summary_df.loc[summary_df['Usage_Percentage'].idxmin()]
        
        print(f"\n📊 Most used category: {most_used['Category']} ({most_used['Usage_Percentage']:.1f}%)")
        print(f"📊 Least used category: {least_used['Category']} ({least_used['Usage_Percentage']:.1f}%)")
    
    def run_analysis(self):
        """Run the complete category comparison analysis."""
        print("🚀 Starting Category Comparison Analysis")
        print("="*80)
        
        # Create comparison table
        comparison_df = self.create_comparison_table()
        
        # Calculate summary statistics
        summary_df = self.calculate_category_summary()
        
        # Save results
        self.save_results(comparison_df, summary_df)
        
        # Display results
        self.display_results(comparison_df, summary_df)
        
        print("\n✅ Category comparison analysis complete!")
        
        return comparison_df, summary_df

def main():
    """Main execution function."""
    data_path = 'final_dataset/complete_analysis_py_adjusted_csv_normalized.csv'
    
    try:
        analyzer = CategoryComparisonAnalyzer(data_path)
        comparison_df, summary_df = analyzer.run_analysis()
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find data file at {data_path}")
        print("Please ensure the dataset file exists in the correct location.")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()