#!/usr/bin/env python3
"""
AI-Powered Text-to-Image Prompt Analysis Pipeline
Uses Ollama local AI to categorize prompt features with high accuracy.
"""

import pandas as pd
import numpy as np
import re
import json
import time
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from langdetect import detect
import warnings
warnings.filterwarnings('ignore')

# Ollama integration
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    print("Warning: Ollama not installed. Install with: pip install ollama")

def get_empty_categories():
    """Return empty categories structure."""
    return {
        'subjects': [],
        'style_modifiers': [],
        'quality_boosters': [],
        'camera_composition': [],
        'lighting_color': [],
        'artists': [],
        'actions_verbs': [],
        'style_codes': [],
        'weights': False
    }

def create_ai_prompt(prompt_text):
    """Create detailed AI categorization prompt with extensive examples but no whitelists."""
    return f"""You are a JSON extraction tool. You MUST respond with ONLY valid JSON. Extract ONLY terms that literally appear in the prompt.

CRITICAL RULES:
- ONLY extract terms that are LITERALLY PRESENT in the prompt
- Do NOT add terms that are not explicitly written
- Be extremely conservative - when in doubt, exclude

Extract terms into these categories:

1. SUBJECTS (what is depicted - nouns only):
   INCLUDE: People ("1girl", "woman", "man", "child"), body parts ("face", "hair", "hands"), animals ("cat", "dragon", "bird"), objects ("flower", "sword", "building", "car"), clothing ("dress", "hat", "shoes"), etc.
   EXCLUDE: adjectives ("beautiful woman" → only "woman", "cute cat" → only "cat"), colors ("blue hair" → only "hair", "red dress" → only "dress"), sizes ("large breasts" → only "breasts", "small hands" → only "hands"), styles ("anime girl" → only "girl")

2. STYLE_MODIFIERS (art styles and techniques):
   INCLUDE: Art styles ("anime", "realistic", "cartoon"), techniques ("oil painting", "digital art", "watercolor"), movements ("impressionist", "surreal", "abstract")
   EXCLUDE: quality terms ("detailed style", "high quality art"), technical codes ("k4_illu", "4ndaerz_illu", "cksc"), color themes ("blue theme", "dark style"), descriptive words ("beautiful", "amazing")

3. QUALITY_BOOSTERS (enhancement terms):
   INCLUDE: Quality words ("masterpiece", "best quality", "detailed", "intricate"), resolution terms ("8k", "4k", "high resolution", "absurdres"), enhancement ("ultra detailed", "highly detailed")
   EXCLUDE: camera terms ("depth of field"), lighting ("dramatic lighting")

4. CAMERA_COMPOSITION (photography and framing):
   INCLUDE: Shots ("close up", "wide shot", "full body", "portrait"), angles ("low angle", "high angle", "overhead"), focus ("depth of field", "bokeh", "shallow focus")
   EXCLUDE: lighting terms ("soft light"), quality terms ("sharp focus")

5. LIGHTING_COLOR (lighting setups only):
   INCLUDE: Lighting ("golden hour", "studio lighting", "dramatic lighting", "soft light", "rim light", "volumetric lighting", "god rays")
   EXCLUDE: color descriptions ("blue theme", "orange hue", "red tint")

6. ARTISTS (real human names only, max 3 words):
   INCLUDE: Real artist names ("Van Gogh", "Picasso", "Greg Rutkowski", "Thomas Kinkade", "Enki Bilal", "Norman Rockwell")
   EXCLUDE: Technical names ("Forest_Decay_Style_ichi_sk", "ka_marukogedago", "FluxMythP0rtr4itStyle", "FLUX-daubrez-DB4RZ-v2", "SabrinaPokemonAnime_IXL_v1"), LoRA usernames, style codes, anything with underscores/numbers/colons

7. ACTIONS_VERBS (single action words in -ing form):
   INCLUDE: Simple actions converted to -ing form ("sit" → "sitting", "run" → "running", "smile" → "smiling", "roar" → "roaring")
   EXCLUDE: Complex phrases ("looking at viewer", "sitting on chair", "flying through jungle", "walking down street"), multi-word actions, non-actions ("beautiful", "detailed", "quality")

8. STYLE_CODES (technical references):
   INCLUDE: LoRA codes ("<lora:anything:number>"), technical identifiers ("k4_illu", "cksc", "mythp0rt"), model names ("SDXL", "FLUX")
   EXCLUDE: artist names, quality terms, style descriptions

9. WEIGHTS: true if you find:
   - Numeric weights: "(word:1.3)", "(term:0.8)", "[word:1.5]"
   - Repetition: "very very", "highly highly"

DETAILED EXAMPLES:

Input: "1girl, anime, Van Gogh style, sitting, masterpiece, close up, dramatic lighting, <lora:style:1.2>"
Output: {{"subjects": ["1girl"], "style_modifiers": ["anime"], "quality_boosters": ["masterpiece"], "camera_composition": ["close up"], "lighting_color": ["dramatic lighting"], "artists": ["Van Gogh"], "actions_verbs": ["sitting"], "style_codes": ["<lora:style:1.2>"], "weights": true}}

Input: "dragon roaring, Forest_Decay_Style_ichi_sk, detailed, low angle, golden hour"
Output: {{"subjects": ["dragon"], "style_modifiers": [], "quality_boosters": ["detailed"], "camera_composition": ["low angle"], "lighting_color": ["golden hour"], "artists": [], "actions_verbs": ["roaring"], "style_codes": [], "weights": false}}

Input: "beautiful woman with blue hair, Picasso, painting, 8k, portrait, studio lighting"
Output: {{"subjects": ["woman", "hair"], "style_modifiers": ["painting"], "quality_boosters": ["8k"], "camera_composition": ["portrait"], "lighting_color": ["studio lighting"], "artists": ["Picasso"], "actions_verbs": [], "style_codes": [], "weights": false}}

Input: "cat sleeping on bed, photorealistic, Greg Rutkowski, highly detailed, wide shot, soft light, (detailed:1.3)"
Output: {{"subjects": ["cat", "bed"], "style_modifiers": ["photorealistic"], "quality_boosters": ["highly detailed"], "camera_composition": ["wide shot"], "lighting_color": ["soft light"], "artists": ["Greg Rutkowski"], "actions_verbs": ["sleeping"], "style_codes": [], "weights": true}}

Input: "spaceship flying through nebula, scifi art, 4k, dramatic angle, rim lighting, very very detailed"
Output: {{"subjects": ["spaceship", "nebula"], "style_modifiers": ["scifi art"], "quality_boosters": ["4k", "detailed"], "camera_composition": ["dramatic angle"], "lighting_color": ["rim lighting"], "artists": [], "actions_verbs": ["flying"], "style_codes": [], "weights": true}}

Now analyze this prompt: "{prompt_text}"

Respond with ONLY valid JSON:"""

def clean_artists(artists_list):
    """Clean artist list by filtering out technical patterns, not by whitelist."""
    if not artists_list:
        return []
    
    cleaned = []
    for artist in artists_list:
        if not artist or not isinstance(artist, str):
            continue
            
        artist_clean = artist.strip()
        
        # Skip if too long (more than 3 words)
        if len(artist_clean.split()) > 3:
            continue
            
        # Skip technical patterns and LoRA codes
        skip_patterns = [
            '_', ':', '<', '>', 'lora', 'flux', 'sdxl', 'xl', 'v1', 'v2', 'v3',
            'style_', '_style', 'cksc', 'cknc', 'ckpf', 'k4_', '_illu',
            'mythp0rt', 'ne0nfant4sy', 'midjourney', 'dalle', 'stable',
            'diffusion', 'checkpoint', 'model', 'epoch', 'steps'
        ]
        
        artist_lower = artist_clean.lower()
        if any(pattern in artist_lower for pattern in skip_patterns):
            continue
            
        # Skip if contains numbers
        if re.search(r'\d', artist_clean):
            continue
            
        # Skip single letter or very short names (likely technical)
        if len(artist_clean) < 3:
            continue
            
        # Skip common non-artist words
        non_artist_words = [
            'quality', 'detailed', 'resolution', 'render', 'painting',
            'digital', 'photo', 'image', 'picture', 'artwork', 'drawing'
        ]
        if artist_lower in non_artist_words:
            continue
            
        cleaned.append(artist_clean)
    
    return list(set(cleaned))  # Remove duplicates

def clean_actions_verbs(verbs_list):
    """Clean verbs list - extract only -ing words from multi-word phrases."""
    if not verbs_list:
        return []
    
    cleaned = []
    for verb in verbs_list:
        if not verb or not isinstance(verb, str):
            continue
            
        verb_clean = verb.strip()
        words = verb_clean.split()
        
        if len(words) == 1:
            # Single word - convert to -ing if needed
            word = words[0].lower()
            
            # Skip if too short or contains numbers/technical patterns
            if len(word) < 2 or re.search(r'\d|_|:', word):
                continue
                
            # Skip common non-action words
            non_action_words = [
                'and', 'or', 'the', 'a', 'an', 'with', 'at', 'in', 'on', 'of', 'to', 'for',
                'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
                'very', 'really', 'quite', 'much', 'more', 'most', 'less', 'quality', 'detailed'
            ]
            
            if word in non_action_words:
                continue
                
            # Convert to -ing form if not already
            if word.endswith('ing'):
                cleaned.append(word)
            else:
                # Basic -ing conversion
                if word.endswith('e') and not word.endswith('ee'):
                    converted = word[:-1] + 'ing'  # 'dance' -> 'dancing'
                elif len(word) >= 3 and word[-1] in 'ptgnm' and word[-2] in 'aeiou' and word[-3] not in 'aeiou':
                    converted = word + word[-1] + 'ing'  # 'run' -> 'running'
                elif word.endswith('y'):
                    converted = word[:-1] + 'ying'  # 'cry' -> 'crying'
                else:
                    converted = word + 'ing'  # Default
                    
                cleaned.append(converted)
        else:
            # Multi-word phrase - only extract words ending in -ing
            for word in words:
                word_clean = word.lower().strip('.,!?;:')
                if word_clean.endswith('ing') and len(word_clean) > 3:
                    cleaned.append(word_clean)
    
    return list(set(cleaned))  # Remove duplicates

def safe_detect_language(text):
    """Safely detect language with error handling."""
    if not text or pd.isna(text):
        return 'unknown'
    
    try:
        text_str = str(text).strip()
        if len(text_str) < 3:  # Too short for detection
            return 'unknown'
        return detect(text_str)
    except:
        # Return 'unknown' for any detection errors
        return 'unknown'

def normalize_term(term, category):
    """Normalize terms for consistency across all categories except style_codes."""
    if category == 'style_codes':
        return term  # Don't normalize style codes
    
    if not term or not isinstance(term, str):
        return term
        
    # Remove parentheses
    term = re.sub(r'[()]', '', term)
    
    # Convert to lowercase
    term = term.lower().strip()
    
    # Replace underscores with spaces
    term = term.replace('_', ' ')
    
    # Replace hyphens with spaces for multi-word terms
    term = re.sub(r'-', ' ', term)
    
    # Clean up multiple spaces
    term = re.sub(r'\s+', ' ', term).strip()
    
    return term

def parse_ai_response(response_text):
    """Safely parse AI response with robust JSON extraction and post-processing."""
    try:
        # Remove any extra text and find JSON
        response_text = response_text.strip()
        
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
            
            # Validate structure
            expected_keys = ['subjects', 'style_modifiers', 'quality_boosters',
                           'camera_composition', 'lighting_color', 'artists',
                           'actions_verbs', 'style_codes', 'weights']
            
            if all(key in result for key in expected_keys):
                # Apply post-processing filters
                if 'artists' in result:
                    result['artists'] = clean_artists(result['artists'])
                if 'actions_verbs' in result:
                    result['actions_verbs'] = clean_actions_verbs(result['actions_verbs'])
                
                # Apply normalization to all categories except style_codes
                for category in expected_keys:
                    if category != 'weights' and category != 'style_codes' and category in result:
                        if isinstance(result[category], list):
                            result[category] = [normalize_term(term, category) for term in result[category] if term]
                
                return result
        
        # If parsing fails, return empty
        return get_empty_categories()
        
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return get_empty_categories()

def categorize_with_ollama(prompt_text, model="llama3:8b", max_retries=3):
    """Categorize prompt using Ollama with progressive retry logic."""
    if not HAS_OLLAMA:
        print("CRITICAL ERROR: Ollama not installed or available")
        print("Please install Ollama and ensure the service is running")
        raise SystemExit("Analysis stopped - Ollama service unavailable")
    
    if not prompt_text or pd.isna(prompt_text):
        return get_empty_categories()
    
    ai_prompt = create_ai_prompt(str(prompt_text))
    
    # Progressive delays: 10s, 30s, 60s
    delays = [10, 30, 60]
    
    for attempt in range(max_retries):
        try:
            response = ollama.generate(
                model=model,
                prompt=ai_prompt,
                options={
                    'temperature': 0.1,  # Low temperature for consistency
                    'top_p': 0.9,
                    'num_predict': 500   # Limit response length
                }
            )
            
            if 'response' in response:
                result = parse_ai_response(response['response'])
                
                # Validate result has expected structure
                if isinstance(result, dict) and 'subjects' in result:
                    return result
            
        except Exception as e:
            print(f"Ollama error (attempt {attempt+1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                delay = delays[attempt] if attempt < len(delays) else 60
                print(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                # All retries failed - stop the entire analysis
                print(f"\nCRITICAL: Ollama failed after {max_retries} attempts")
                print("Please check Ollama service status and restart the analysis manually")
                print(f"Last error: {e}")
                raise SystemExit("Analysis stopped - Ollama service unavailable after retries")
    
    # This should never be reached due to SystemExit above
    raise SystemExit("Analysis stopped - Ollama service unavailable")


def process_negative_prompt(text):
    """Process negative prompt by splitting on commas and normalizing."""
    if not text or pd.isna(text):
        return []
    
    text = str(text)
    # Split by commas
    terms = [term.strip() for term in text.split(',')]
    
    # Apply full normalization to negative terms
    normalized_terms = []
    for term in terms:
        if term:
            # Apply same normalization as other categories (not style_codes)
            normalized = normalize_term(term, 'negative_terms')
            if normalized:
                normalized_terms.append(normalized)
    
    return normalized_terms

def save_progress(results, output_dir, batch_num):
    """Save intermediate results."""
    progress_file = output_dir / f'progress_batch_{batch_num}.json'
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def load_progress(output_dir):
    """Load previous progress if exists."""
    progress_files = list(output_dir.glob('progress_batch_*.json'))
    if not progress_files:
        return []
    
    # Load the latest progress file
    latest_file = max(progress_files, key=lambda x: int(x.stem.split('_')[-1]))
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_prompts_with_ai(df, output_dir, model="llama3:8b", batch_size=100):
    """Process all prompts with AI categorization and progress saving."""
    print(f"Processing {len(df)} prompts with AI categorization...")
    
    # Check for existing progress
    existing_results = load_progress(output_dir)
    start_idx = len(existing_results)
    
    if start_idx > 0:
        print(f"Resuming from prompt {start_idx + 1}")
        results = existing_results
    else:
        results = []
    
    # Global vocabulary accumulators
    global_vocabularies = defaultdict(Counter)
    
    for idx in range(start_idx, len(df)):
        row = df.iloc[idx]
        
        # Progress indicator
        if idx % 10 == 0:
            progress = (idx / len(df)) * 100
            print(f"Processing prompt {idx+1}/{len(df)} ({progress:.1f}%)")
        
        prompt_text = row.get('prompt', '')
        negative_text = row.get('negative_prompt', '')
        
        # Categorize prompt with AI
        prompt_categories = categorize_with_ollama(prompt_text, model)
        
        # Process negative prompt
        negative_terms = process_negative_prompt(negative_text)
        
        # Create result record with ALL original data + AI results
        result = {}
        
        # Include ALL original data from the Excel row
        for col in df.columns:
            try:
                value = row[col]
                # Convert pandas types to Python types for JSON serialization
                if pd.isna(value):
                    result[col] = None
                elif isinstance(value, (pd.Timestamp, np.datetime64)):
                    result[col] = str(value)
                elif isinstance(value, (np.integer, np.floating)):
                    result[col] = float(value) if np.isfinite(value) else None
                else:
                    result[col] = str(value) if value is not None else None
            except:
                result[col] = None
        
        # Add AI analysis results
        result.update({
            'prompt_id': idx,
            'prompt_original': str(prompt_text) if prompt_text else '',
            'negative_original': str(negative_text) if negative_text else '',
            **prompt_categories,  # All 8 categories + weights
            'negative_terms': negative_terms,
            
            # Counts for analysis
            'subjects_count': len(prompt_categories['subjects']),
            'style_modifiers_count': len(prompt_categories['style_modifiers']),
            'quality_boosters_count': len(prompt_categories['quality_boosters']),
            'camera_composition_count': len(prompt_categories['camera_composition']),
            'lighting_color_count': len(prompt_categories['lighting_color']),
            'artists_count': len(prompt_categories['artists']),
            'actions_verbs_count': len(prompt_categories['actions_verbs']),
            'style_codes_count': len(prompt_categories['style_codes']),
            'negative_terms_count': len(negative_terms),
            
            # Basic text metrics
            'prompt_word_count': len(str(prompt_text).split()) if prompt_text else 0,
            'negative_word_count': len(str(negative_text).split()) if negative_text else 0,
            'prompt_char_count': len(str(prompt_text)) if prompt_text else 0,
            'has_negative': bool(negative_text and str(negative_text).strip()),
            'prompt_lang': safe_detect_language(prompt_text)
        })
        
        results.append(result)
        
        # Accumulate vocabulary
        for category in ['subjects', 'style_modifiers', 'quality_boosters',
                        'camera_composition', 'lighting_color', 'artists',
                        'actions_verbs', 'style_codes']:
            category_terms = prompt_categories.get(category, [])
            # Ensure it's a list
            if isinstance(category_terms, list):
                for term in category_terms:
                    if isinstance(term, str):  # Only add string terms
                        global_vocabularies[category][term] += 1
        
        # Accumulate negative terms
        for term in negative_terms:
            global_vocabularies['negative_terms'][term] += 1
        
        # Save progress every batch_size prompts
        if (idx + 1) % batch_size == 0:
            save_progress(results, output_dir, (idx + 1) // batch_size)
            save_vocabularies(global_vocabularies, output_dir)
            print(f"Progress saved at prompt {idx + 1}")
    
    return pd.DataFrame(results), global_vocabularies

def save_vocabularies(global_vocabularies, output_dir):
    """Save global vocabulary files."""
    vocab_dir = output_dir / 'vocabularies'
    vocab_dir.mkdir(exist_ok=True)
    
    for category, counter in global_vocabularies.items():
        if counter:
            vocab_df = pd.DataFrame(counter.most_common(), columns=['term', 'frequency'])
            vocab_df.to_csv(vocab_dir / f'{category}_all.csv', index=False)

def calculate_engagement_metrics(df):
    """Calculate engagement metrics."""
    print("Calculating engagement metrics...")
    
    # Basic engagement
    engagement_cols = ['like_count', 'heart_count', 'comment_count', 'laugh_count', 'cry_count']
    available_cols = [col for col in engagement_cols if col in df.columns]
    
    if available_cols:
        df['engagement_total'] = df[available_cols].fillna(0).sum(axis=1)
    elif 'total_reactions' in df.columns:
        df['engagement_total'] = df['total_reactions'].fillna(0)
    else:
        df['engagement_total'] = 0
    
    # Image metrics
    if 'width' in df.columns and 'height' in df.columns:
        df['aspect_ratio'] = df['width'] / df['height']
        df['megapixels'] = (df['width'] * df['height']) / 1e6
        df['engagement_per_mp'] = df['engagement_total'] / np.maximum(df['megapixels'], 0.1)
    
    return df

def calculate_correlations(df, output_dir):
    """Calculate correlations between categories and engagement."""
    print("Calculating category-engagement correlations...")
    
    # Category count columns
    category_cols = [
        'subjects_count', 'style_modifiers_count', 'quality_boosters_count',
        'camera_composition_count', 'lighting_color_count', 'artists_count',
        'actions_verbs_count', 'style_codes_count', 'negative_terms_count'
    ]
    
    # Boolean columns
    boolean_cols = ['weights', 'has_negative']
    
    # Basic metrics
    basic_cols = ['prompt_word_count', 'negative_word_count']
    
    if 'aspect_ratio' in df.columns:
        basic_cols.extend(['aspect_ratio', 'megapixels'])
    
    all_features = category_cols + boolean_cols + basic_cols
    target_cols = ['engagement_total']
    if 'engagement_per_mp' in df.columns:
        target_cols.append('engagement_per_mp')
    
    # Calculate correlations
    corr_data = []
    for feature in all_features:
        if feature in df.columns:
            for target in target_cols:
                if target in df.columns:
                    try:
                        feature_data = df[feature].fillna(0)
                        target_data = df[target].fillna(0)
                        
                        # Very simple correlation calculation
                        try:
                            corr_result = spearmanr(feature_data, target_data)
                            
                            # Safe extraction
                            corr_val = 0.0
                            p_val = 1.0
                            
                            if corr_result is not None:
                                if hasattr(corr_result, '__len__') and len(corr_result) >= 2:
                                    try:
                                        corr_val = corr_result[0]
                                        p_val = corr_result[1]
                                        
                                        # Convert strings and check for valid numbers
                                        if isinstance(corr_val, (int, float)) and np.isfinite(corr_val):
                                            corr_val = corr_val
                                        else:
                                            corr_val = 0.0
                                            
                                        if isinstance(p_val, (int, float)) and np.isfinite(p_val):
                                            p_val = p_val
                                        else:
                                            p_val = 1.0
                                    except:
                                        corr_val = 0.0
                                        p_val = 1.0
                                        
                        except Exception as corr_error:
                            print(f"Correlation calculation error for {feature} vs {target}: {corr_error}")
                            corr_val = 0.0
                            p_val = 1.0
                        
                        corr_data.append({
                            'feature': feature,
                            'target': target,
                            'correlation': corr_val,
                            'p_value': p_val,
                            'significant': p_val < 0.05
                        })
                    except Exception as e:
                        print(f"Error calculating correlation for {feature}: {e}")
                        continue
    
    if corr_data:
        corr_df = pd.DataFrame(corr_data)
        corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
        corr_df.to_csv(output_dir / 'category_engagement_correlations.csv', index=False)

def create_plots(df, output_dir):
    """Create visualization plots."""
    print("Creating plots...")
    
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Category usage
    category_counts = {
        'Subjects': df['subjects_count'].sum(),
        'Style Modifiers': df['style_modifiers_count'].sum(),
        'Quality Boosters': df['quality_boosters_count'].sum(),
        'Camera & Composition': df['camera_composition_count'].sum(),
        'Lighting & Color': df['lighting_color_count'].sum(),
        'Artists': df['artists_count'].sum(),
        'Actions & Verbs': df['actions_verbs_count'].sum(),
        'Style Codes': df['style_codes_count'].sum()
    }
    
    plt.figure(figsize=(12, 8))
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    plt.barh(categories, counts)
    plt.title('AI-Categorized Feature Usage Across All Prompts')
    plt.xlabel('Total Occurrences')
    plt.tight_layout()
    plt.savefig(plots_dir / 'ai_category_usage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Weights usage
    if 'weights' in df.columns:
        weights_counts = df['weights'].value_counts()
        plt.figure(figsize=(8, 6))
        labels = ['No Weights/Emphasis', 'Has Weights/Emphasis']
        plt.pie(weights_counts.values, labels=labels, autopct='%1.1f%%')
        plt.title('Prompts Using Weights or Emphasis Patterns')
        plt.savefig(plots_dir / 'weights_usage.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='AI-powered prompt analysis')
    parser.add_argument('--input', required=True, help='Path to Excel file')
    parser.add_argument('--sheet', default=0, help='Sheet name or index')
    parser.add_argument('--outdir', default='outputs', help='Output directory')
    parser.add_argument('--sample', type=int, help='Sample size for testing')
    parser.add_argument('--model', default='llama3:8b', help='Ollama model to use')
    parser.add_argument('--batch_size', type=int, default=100, help='Save progress every N prompts')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from {args.input}...")
    
    # Load data
    try:
        df = pd.read_excel(args.input, sheet_name=args.sheet)
        print(f"Loaded {len(df)} rows")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Sample if requested
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42)
        print(f"Sampled {len(df)} rows for analysis")
    
    # Parse created_at if present
    if 'created_at' in df.columns:
        try:
            df['created_at'] = pd.to_datetime(df['created_at'], format='%d/%m/%Y %H:%M', errors='coerce')
        except:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    
    # Process prompts with AI (already includes all original data)
    result_df, global_vocabularies = process_prompts_with_ai(
        df, output_dir, args.model, args.batch_size
    )
    
    # Calculate engagement metrics
    result_df = calculate_engagement_metrics(result_df)
    
    # Save main results
    print("Saving final results...")
    result_df.to_parquet(output_dir / 'ai_categorized_features.parquet', index=False)
    result_df.head(100).to_csv(output_dir / 'ai_categorized_features_sample.csv', index=False)
    
    # Save final vocabularies
    save_vocabularies(global_vocabularies, output_dir)
    
    # Calculate correlations
    calculate_correlations(result_df, output_dir)
    
    # Create plots
    create_plots(result_df, output_dir)
    
    # Clean up progress files
    for progress_file in output_dir.glob('progress_batch_*.json'):
        progress_file.unlink()
    
    # Print summary
    print("\n=== AI-POWERED ANALYSIS SUMMARY ===")
    print(f"Total prompts analyzed: {len(result_df)}")
    print(f"Average prompt length: {result_df['prompt_word_count'].mean():.1f} words")
    if 'engagement_total' in result_df.columns:
        print(f"Average engagement: {result_df['engagement_total'].mean():.1f}")
    
    print(f"\nAI-categorized features:")
    for category in ['subjects', 'style_modifiers', 'quality_boosters', 'camera_composition',
                    'lighting_color', 'artists', 'actions_verbs', 'style_codes']:
        count_col = f"{category}_count"
        if count_col in result_df.columns:
            prompts_with_category = (result_df[count_col] > 0).sum()
            total_terms = result_df[count_col].sum()
            print(f"  {category}: {prompts_with_category} prompts ({total_terms} total terms)")
    
    weights_count = result_df['weights'].sum() if 'weights' in result_df.columns else 0
    print(f"\nPrompts with weights/emphasis: {weights_count}")
    
    print(f"\nGlobal vocabularies saved to: {output_dir}/vocabularies/")
    print("AI-powered analysis complete!")

if __name__ == "__main__":
    main()