#!/usr/bin/env python3
"""
Simple script to run the advanced analysis
"""

from advanced_prompt_analysis_v2 import AdvancedPromptAnalyzer

def main():
    # Path to the data file
    data_file = "../../final_dataset/complete_analysis_py_adjusted_csv_normalized.csv"
    
    # Initialize analyzer
    analyzer = AdvancedPromptAnalyzer(data_file)
    
    try:
        # Load and prepare data
        print("Loading data...")
        analyzer.load_and_prepare_data()
        print(f"Loaded {len(analyzer.df)} rows")
        print(f"Columns: {analyzer.df.columns.tolist()}")
        
        # Create boolean features
        analyzer.create_boolean_features()
        print("Created boolean features")
        
        # Run all analyses
        analyzer.analyze_median_contrasts()
        analyzer.analyze_bucket_effects()
        analyzer.analyze_synergy_effects()
        analyzer.analyze_co_occurrence()
        analyzer.analyze_partial_correlations()
        analyzer.analyze_top_combinations()
        
        # Create visualizations
        analyzer.create_visualizations()
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()