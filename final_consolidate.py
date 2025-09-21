#!/usr/bin/env python3
"""
Simple final consolidation - loads only the complete dataset from progress_batch_291.json
"""

import pandas as pd
import json
import time
from pathlib import Path

def clean_data_types(df):
    """Clean data types for consistency."""
    print("🔧 Cleaning data types...")
    
    # List columns that should always be lists
    list_columns = [
        'subjects', 'style_modifiers', 'quality_boosters',
        'camera_composition', 'lighting_color', 'artists',
        'actions_verbs', 'style_codes', 'negative_terms'
    ]
    
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [] if pd.isna(x) or x is None else [x])
    
    # Boolean columns
    bool_columns = ['weights', 'has_negative']
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    # Numeric columns
    numeric_columns = [
        'subjects_count', 'style_modifiers_count', 'quality_boosters_count',
        'camera_composition_count', 'lighting_color_count', 'artists_count',
        'actions_verbs_count', 'style_codes_count', 'negative_terms_count',
        'prompt_word_count', 'negative_word_count', 'prompt_char_count'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    return df

def main():
    print("🎯 FINAL DATASET CONSOLIDATION")
    print("=" * 50)
    
    # Paths
    input_file = Path("thesis_analysis_final/progress_batch_291.json")
    output_dir = Path("final_dataset")
    output_dir.mkdir(exist_ok=True)
    
    print(f"📄 Loading complete dataset from: {input_file}")
    
    # Load the complete dataset
    start_time = time.time()
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            complete_data = json.load(f)
        
        print(f"✅ Loaded {len(complete_data)} records")
        
        # Convert to DataFrame
        print("📊 Converting to DataFrame...")
        df = pd.DataFrame(complete_data)
        
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        # Clean data types
        df = clean_data_types(df)
        
        # Remove any potential duplicates based on prompt_id
        initial_rows = len(df)
        if 'prompt_id' in df.columns:
            df = df.drop_duplicates(subset=['prompt_id'])
            final_rows = len(df)
            if initial_rows != final_rows:
                print(f"🔄 Removed {initial_rows - final_rows} duplicate rows")
        
        # Save files
        print("💾 Saving files...")
        
        # 1. Main CSV (Python-friendly with lists)
        csv_path = output_dir / 'complete_analysis.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✅ Python CSV: {csv_path}")
        
        # 2. Excel-friendly CSV (lists converted to strings)
        excel_df = df.copy()
        list_columns = [
            'subjects', 'style_modifiers', 'quality_boosters',
            'camera_composition', 'lighting_color', 'artists',
            'actions_verbs', 'style_codes', 'negative_terms'
        ]
        
        for col in list_columns:
            if col in excel_df.columns:
                excel_df[col] = excel_df[col].apply(
                    lambda x: '; '.join(str(item) for item in x) if isinstance(x, list) else str(x)
                )
        
        excel_path = output_dir / 'complete_analysis_excel.csv'
        excel_df.to_csv(excel_path, index=False, encoding='utf-8')
        print(f"✅ Excel CSV: {excel_path}")
        
        # 3. Parquet (fast loading)
        try:
            parquet_path = output_dir / 'complete_analysis.parquet'
            df.to_parquet(parquet_path, index=False)
            print(f"✅ Parquet: {parquet_path}")
        except Exception as e:
            print(f"⚠️  Parquet save failed: {e}")
        
        # 4. Sample JSON (first 100 rows)
        sample_path = output_dir / 'sample_data.json'
        sample_data = df.head(100).to_dict('records')
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Sample JSON: {sample_path}")
        
        # 5. Summary statistics
        summary = {
            'total_prompts': len(df),
            'processing_time_seconds': time.time() - start_time,
            'file_size_mb': round(csv_path.stat().st_size / 1024 / 1024, 2),
            'columns': list(df.columns),
            'column_count': len(df.columns),
            'unique_platforms': df['platform'].nunique() if 'platform' in df.columns else 0,
            'language_distribution': df['prompt_lang'].value_counts().head(5).to_dict() if 'prompt_lang' in df.columns else {},
            'average_prompt_length': df['prompt_word_count'].mean() if 'prompt_word_count' in df.columns else 0,
            'categories_avg': {
                'subjects': df['subjects_count'].mean() if 'subjects_count' in df.columns else 0,
                'style_modifiers': df['style_modifiers_count'].mean() if 'style_modifiers_count' in df.columns else 0,
                'artists': df['artists_count'].mean() if 'artists_count' in df.columns else 0,
            }
        }
        
        summary_path = output_dir / 'dataset_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ Summary: {summary_path}")
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 50)
        print("🎉 CONSOLIDATION COMPLETE!")
        print(f"📊 Final dataset: {len(df):,} prompts")
        print(f"📁 Columns: {len(df.columns)}")
        print(f"⏱️  Processing time: {elapsed_time:.1f} seconds")
        print(f"💾 Main file size: {csv_path.stat().st_size / 1024 / 1024:.1f} MB")
        print("\n📁 Output files created:")
        print(f"   • complete_analysis.csv (Python)")
        print(f"   • complete_analysis_excel.csv (Excel)")
        print(f"   • complete_analysis.parquet (fast)")
        print(f"   • sample_data.json (preview)")
        print(f"   • dataset_summary.json (stats)")
        print("\n✅ Ready for analysis!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🧹 Cleaning up temporary files...")
        
        # List of files to remove
        temp_files = [
            'consolidate_results.py',
            'stream_consolidate.py', 
            'debug_consolidate.py',
            'final_dataset/complete_analysis_streaming.csv'  # The huge incorrect file
        ]
        
        for file_path in temp_files:
            try:
                Path(file_path).unlink(missing_ok=True)
                print(f"🗑️  Removed: {file_path}")
            except Exception as e:
                print(f"⚠️  Could not remove {file_path}: {e}")
        
        print("✅ Cleanup complete!")