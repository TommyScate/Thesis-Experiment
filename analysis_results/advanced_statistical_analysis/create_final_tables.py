#!/usr/bin/env python3
"""
Create final table visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def create_top_combinations_table():
    """Create top combinations table visualization"""
    print("Creating top combinations table...")
    
    def wrap_combo(combo: str, max_terms_per_line: int = 4) -> str:
        """Wrap long combination strings on ' + ' boundaries for readability."""
        parts = [p.strip() for p in str(combo).split(' + ')]
        if len(parts) <= max_terms_per_line:
            return ' + '.join(parts)
        lines = []
        for i in range(0, len(parts), max_terms_per_line):
            lines.append(' + '.join(parts[i:i + max_terms_per_line]))
        return '\n'.join(lines)
    
    for metric in ['likes_per_month', 'reactions_per_month']:
        try:
            # Load the combinations data
            combo_file = f'top_terms/{metric}_top_combinations.csv'
            if not Path(combo_file).exists():
                continue
                
            df = pd.read_csv(combo_file)
            
            # Create table visualization
            # Make figure ~10% wider
            fig, ax = plt.subplots(figsize=(17.6, 10))
            ax.axis('tight')
            ax.axis('off')
            
            # Prepare data for table - handle both legacy and new column names
            # Accept either 'combination' or 'feature_combination'
            if 'combination' in df.columns:
                combo_col = 'combination'
            elif 'feature_combination' in df.columns:
                combo_col = 'feature_combination'
            else:
                print(f"Skipping {metric}: no combination column found in {combo_file}")
                plt.close(fig)
                continue

            # Ensure median column exists
            if 'median_engagement' not in df.columns:
                print(f"Skipping {metric}: 'median_engagement' column missing in {combo_file}")
                plt.close(fig)
                continue

            table_data = df[[combo_col, 'median_engagement']].head(10).copy()
            table_data.columns = ['Feature Combination', f'Median {metric.replace("_", " ").title()}']
            
            # Round the median values
            table_data.iloc[:, 1] = table_data.iloc[:, 1].round(2)

            # Keep feature combinations on a single line (no wrapping)
            
            # Create table with custom column widths (wider feature column, narrower median column)
            col_widths = [0.80, 0.20]
            table = ax.table(cellText=table_data.values.tolist(),
                             colLabels=table_data.columns.tolist(),
                             colWidths=col_widths,
                             cellLoc='left',
                             loc='center')
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.02, 1.65)
            
            # Style header
            for i in range(len(table_data.columns)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white', ha='center')
            # Move the first header cell text slightly to the left
            table[(0, 0)].set_text_props(ha='left', va='center')
            # Nudge header text left inside the cell (public API)
            try:
                table[(0, 0)].get_text().set_x(0.02)
            except Exception:
                pass
            
            # Alternate row colors and align numeric column to center
            n_rows = len(table_data)
            for r in range(1, n_rows + 1):
                # Zebra striping
                bg = '#f1f1f2' if r % 2 == 0 else 'white'
                table[(r, 0)].set_facecolor(bg)
                table[(r, 1)].set_facecolor(bg)
                # Alignments
                table[(r, 0)].set_text_props(ha='left', va='center')
                table[(r, 1)].set_text_props(ha='center', va='center')
                # Nudge the first column text slightly left inside the cell (public API)
                try:
                    table[(r, 0)].get_text().set_x(0.02)
                except Exception:
                    pass
            
            title_metric = metric.replace("_", " ").title()
            plt.title(f'Top 10 Feature Combinations by {title_metric}',
                      fontsize=16, fontweight='bold', pad=20)
            
            plt.savefig(f'plots/top_combinations_table_{metric}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Created plots/top_combinations_table_{metric}.png")
            
        except Exception as e:
            print(f"Error creating {metric} combinations table: {e}")

def check_weights_in_contrasts():
    """Check if weights is included in median contrasts"""
    print("Checking median contrasts files...")
    
    for metric in ['likes_per_month', 'reactions_per_month']:
        file_path = f'median_contrasts/{metric}_median_contrasts.csv'
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            print(f"\n{metric} features:")
            print(df['feature'].tolist())
            
            if 'Weights' in df['feature'].values:
                print(f"✓ Weights feature found in {metric}")
            else:
                print(f"✗ Weights feature NOT found in {metric}")

def create_synergy_summary():
    """Create summary of synergy effects (robust to schema variants)"""
    synergy_file = 'synergy_analysis/synergy_effects.csv'
    if Path(synergy_file).exists():
        df = pd.read_csv(synergy_file)

        # Determine metric column name
        metric_col = None
        if 'metric' in df.columns:
            metric_col = 'metric'
        elif 'engagement_metric' in df.columns:
            metric_col = 'engagement_metric'
        else:
            print("Synergy summary skipped: no metric column found in synergy_effects.csv")
            return

        # Determine feature name columns
        f1_col = 'feature1_name' if 'feature1_name' in df.columns else ('feature_1' if 'feature_1' in df.columns else None)
        f2_col = 'feature2_name' if 'feature2_name' in df.columns else ('feature_2' if 'feature_2' in df.columns else None)
        if f1_col is None or f2_col is None:
            print("Synergy summary skipped: feature name columns not found")
            return

        # Determine synergy percentage column
        syn_col = 'synergy_percent' if 'synergy_percent' in df.columns else ('synergy_percentage' if 'synergy_percentage' in df.columns else None)
        if syn_col is None:
            print("Synergy summary skipped: synergy percentage column not found")
            return

        # Prefer engagement/month view for summary
        target_metric = 'reactions_per_month' if (df[metric_col] == 'reactions_per_month').any() else 'likes_per_month'
        top_synergies = df[df[metric_col] == target_metric].head(6)

        label = 'Engagement per Month' if target_metric == 'reactions_per_month' else 'Likes per Month'
        print(f"\nTop 6 Synergy Effects ({label}):")
        for _, row in top_synergies.iterrows():
            try:
                print(f"{row[f1_col]} + {row[f2_col]}: {row[syn_col]:.1f}% synergy")
            except Exception:
                print(f"{row[f1_col]} + {row[f2_col]}: {row[syn_col]}% synergy")

def main():
    """Run table creation"""
    print("Creating final visualization tables...")
    
    # Create directories if needed
    Path('plots').mkdir(exist_ok=True)
    
    create_top_combinations_table()
    check_weights_in_contrasts()
    create_synergy_summary()
    
    print("\nAll table visualizations completed!")

if __name__ == "__main__":
    main()