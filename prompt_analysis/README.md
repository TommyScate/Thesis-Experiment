# Text-to-Image Prompt Analysis Pipeline

A comprehensive analysis tool for extracting linguistic features from text-to-image prompts and analyzing their relationship with community engagement.

## Overview

This pipeline processes text-to-image prompts and extracts features based on Oppenlaender's taxonomy and prompt engineering best practices:

- **Subject Terms**: Character/person references, objects, scenes
- **Style Modifiers**: Art movements, media types, aesthetic styles  
- **Image Prompts**: URL/reference detection
- **Quality Boosters**: Technical and artistic quality terms
- **Repeating Terms**: Emphasis patterns and repetition
- **Magic Terms**: Platform-specific boosters
- **Camera/Photography Terms**: Technical photography language
- **Negatives**: Analysis of negative prompt patterns

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your Excel file has these columns:
   - `prompt` (required)
   - `negative_prompt` (optional)
   - `width`, `height` (for aspect ratio)
   - Engagement columns: `like_count`, `heart_count`, `comment_count`, etc.
   - `model`, `username`, `created_at` (optional)

## Usage

Basic usage:
```bash
python analyze_prompts.py --input "path/to/your/data.xlsx"
```

With options:
```bash
python analyze_prompts.py --input "data.xlsx" --sheet 0 --outdir results --sample 1000
```

### Parameters

- `--input`: Path to Excel file (required)
- `--sheet`: Sheet name or index (default: 0)
- `--outdir`: Output directory (default: outputs)
- `--sample`: Sample size for quick testing (optional)

## Output Files

The pipeline generates:

### Main Results
- `features.parquet`: Complete feature dataset
- `features_sample.csv`: First 100 rows for quick inspection

### Summaries
- `summaries/top_artists.csv`: Most referenced artists
- `summaries/top_styles.csv`: Most used style terms
- `summaries/top_lora_tags.csv`: Most common LoRA references
- `summaries/models_usage.csv`: Model usage statistics

### Analysis
- `correlations_spearman.csv`: Feature-engagement correlations

### Visualizations
- `plots/engagement_distribution.png`: Engagement distribution
- `plots/top_artists.png`: Top 20 artists bar chart
- `plots/length_vs_engagement.png`: Prompt length vs engagement
- `plots/style_families.png`: Style category usage

## Feature Categories

### Extracted Features

**Text Analysis:**
- Word counts, character counts, unique words
- Language detection
- Prompt complexity metrics

**Content Classification:**
- Artist names and LoRA references
- Style modifiers by category
- Quality and technical terms
- Camera/photography terminology

**Engagement Metrics:**
- Total engagement (likes + hearts + comments + etc.)
- Engagement per megapixel
- Aspect ratio and image size analysis

**Temporal & User Patterns:**
- Time-based analysis (if created_at available)
- User behavior patterns

## Research Applications

This pipeline supports analysis of:
- Prompt engineering effectiveness
- Style preference trends
- Artist influence on engagement
- Technical quality vs. community response
- Platform-specific optimization patterns
- Temporal evolution of prompt styles

## Notes

- All prompts are assumed to be SFW content
- Language detection uses `langdetect` library
- Rule-based extraction for reproducibility
- Designed for ~10k+ prompt datasets
- Optimized for research and academic use

## Example Output

```
=== ANALYSIS SUMMARY ===
Total prompts analyzed: 25,847
Average prompt length: 23.4 words
Average engagement: 156.2
Prompts with artists: 8,234
Prompts with styles: 19,567
Prompts with negatives: 22,103
Most common language: en