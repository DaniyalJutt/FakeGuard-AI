"""
Reporter Module - Adapted from Phase 8
Generates visualizations and reports
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Import matplotlib with error handling
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    try:
        sns.set_palette("husl")
    except:
        pass
except Exception as e:
    MATPLOTLIB_AVAILABLE = False
    print(f"Warning: Matplotlib not available: {e}")
    # Create dummy functions
    plt = None
    sns = None

# ========================
# STATISTICS FUNCTIONS
# ========================

def compute_statistics(df_clean, df_removed, df_original):
    """Compute comprehensive statistics"""
    
    total = len(df_original)
    
    stats = {
        'total_reviews': total,
        'clean_reviews': len(df_clean),
        'removed_reviews': len(df_removed),
        'clean_percentage': len(df_clean) / total * 100 if total > 0 else 0,
        'removed_percentage': len(df_removed) / total * 100 if total > 0 else 0,
        'retention_rate': len(df_clean) / total * 100 if total > 0 else 0
    }
    
    # Rating distribution
    if 'rating' in df_clean.columns:
        stats['clean_rating_dist'] = df_clean['rating'].value_counts().sort_index().to_dict()
    if 'rating' in df_original.columns:
        stats['original_rating_dist'] = df_original['rating'].value_counts().sort_index().to_dict()
    
    # Category distribution
    if 'category' in df_clean.columns:
        stats['clean_category_dist'] = df_clean['category'].value_counts().to_dict()
    
    # Score statistics
    if 'ensemble_score' in df_removed.columns and len(df_removed) > 0:
        stats['avg_removed_score'] = df_removed['ensemble_score'].mean()
        stats['avg_removed_rule_score'] = df_removed['rule_score'].mean()
        stats['avg_removed_lgbm'] = df_removed['lgbm_proba'].mean()
    
    return stats

# ========================
# VISUALIZATION FUNCTIONS
# ========================

def create_summary_plot(df_clean, df_removed, df_original):
    """Create comprehensive summary visualization"""
    
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Fake Review Detection Summary', fontsize=16, fontweight='bold')
    
    # 1. Before/After Comparison
    ax1 = axes[0, 0]
    categories = ['Original', 'Clean', 'Removed']
    counts = [len(df_original), len(df_clean), len(df_removed)]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Reviews', fontweight='bold')
    ax1.set_title('Before/After Comparison', fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    for i, v in enumerate(counts):
        ax1.text(i, v + max(counts)*0.01, f'{v:,}', ha='center', fontweight='bold')
    
    # 2. Rating Distribution
    ax2 = axes[0, 1]
    if 'rating' in df_original.columns and 'rating' in df_clean.columns:
        ratings = [1, 2, 3, 4, 5]
        original_counts = [df_original[df_original['rating'] == r].shape[0] for r in ratings]
        clean_counts = [df_clean[df_clean['rating'] == r].shape[0] for r in ratings]
        
        x = np.arange(len(ratings))
        width = 0.35
        ax2.bar(x - width/2, original_counts, width, label='Original', color='#3498db', alpha=0.7)
        ax2.bar(x + width/2, clean_counts, width, label='Clean', color='#2ecc71', alpha=0.7)
        ax2.set_xlabel('Rating', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title('Rating Distribution', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(ratings)
        ax2.legend()
        ax2.grid(alpha=0.3, axis='y')
    
    # 3. Score Distribution
    ax3 = axes[1, 0]
    if 'ensemble_score' in df_removed.columns and len(df_removed) > 0:
        ax3.hist(df_removed['ensemble_score'], bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax3.axvline(3.5, color='red', linestyle='--', linewidth=2, label='Threshold: 3.5')
        ax3.set_xlabel('Ensemble Score', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Removed Reviews Score Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3, axis='y')
    
    # 4. Decision Breakdown
    ax4 = axes[1, 1]
    if 'decision' in df_original.columns:
        decisions = df_original['decision'].value_counts()
        colors_pie = ['#2ecc71', '#e74c3c', '#f39c12']
        ax4.pie(decisions.values, labels=decisions.index, autopct='%1.1f%%', 
                colors=colors_pie[:len(decisions)], startangle=90)
        ax4.set_title('Decision Breakdown', fontweight='bold')
    else:
        # Fallback: show clean vs removed
        clean_count = len(df_clean)
        removed_count = len(df_removed)
        ax4.pie([clean_count, removed_count], labels=['Clean', 'Removed'], 
                autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
        ax4.set_title('Clean vs Removed', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_score_breakdown_plot(df_removed):
    """Create score component breakdown plot"""
    
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    if len(df_removed) == 0:
        return None
    
    # Get score columns
    score_cols = [col for col in df_removed.columns if col.startswith('score_')]
    
    if not score_cols:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate average contribution of each score component
    component_means = df_removed[score_cols].mean().sort_values(ascending=False)
    
    ax.barh(range(len(component_means)), component_means.values, color='#e67e22', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(component_means)))
    ax.set_yticklabels([col.replace('score_', '').replace('_', ' ').title() for col in component_means.index])
    ax.set_xlabel('Average Score Contribution', fontweight='bold')
    ax.set_title('Top Score Components in Removed Reviews', fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig

# ========================
# REPORT GENERATION
# ========================

def generate_report(df_clean, df_removed, df_original, output_dir='outputs'):
    """
    Generate comprehensive report
    
    Args:
        df_clean: Clean reviews DataFrame
        df_removed: Removed reviews DataFrame
        df_original: Original reviews DataFrame
        output_dir: Output directory for reports
    
    Returns:
        dict with statistics, plots, and report text
    """
    
    # Compute statistics
    stats = compute_statistics(df_clean, df_removed, df_original)
    
    # Create visualizations
    summary_plot = create_summary_plot(df_clean, df_removed, df_original)
    score_plot = create_score_breakdown_plot(df_removed)
    
    # Generate report text
    report_text = f"""
FAKE REVIEW DETECTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

OVERALL STATISTICS:
-------------------
Total Reviews (Original): {stats['total_reviews']:,}
Clean Reviews: {stats['clean_reviews']:,} ({stats['clean_percentage']:.1f}%)
Removed Reviews: {stats['removed_reviews']:,} ({stats['removed_percentage']:.1f}%)
Retention Rate: {stats['retention_rate']:.1f}%

"""
    
    if 'avg_removed_score' in stats:
        report_text += f"""
REMOVED REVIEWS ANALYSIS:
-------------------------
Average Ensemble Score: {stats['avg_removed_score']:.2f}
Average Rule Score: {stats['avg_removed_rule_score']:.2f}
Average ML Probability: {stats['avg_removed_lgbm']:.3f}

"""
    
    if 'clean_rating_dist' in stats:
        report_text += "CLEAN REVIEWS RATING DISTRIBUTION:\n"
        report_text += "-----------------------------------\n"
        for rating, count in sorted(stats['clean_rating_dist'].items()):
            pct = count / stats['clean_reviews'] * 100 if stats['clean_reviews'] > 0 else 0
            report_text += f"  {rating}★: {count:,} ({pct:.1f}%)\n"
        report_text += "\n"
    
    report_text += f"""
CONCLUSION:
-----------
The detection pipeline successfully identified and removed {stats['removed_reviews']:,} 
fake reviews ({stats['removed_percentage']:.1f}% of total). The clean dataset 
contains {stats['clean_reviews']:,} reviews ready for further analysis.

"""
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return {
        'statistics': stats,
        'summary_plot': summary_plot,
        'score_plot': score_plot,
        'report_text': report_text,
        'report_file': report_file
    }

# ========================
# MAIN FUNCTION
# ========================

def generate_report_simple(df_clean, df_removed, df_original):
    """
    Simplified report generation for web app
    
    Returns:
        dict with statistics and plots
    """
    stats = compute_statistics(df_clean, df_removed, df_original)
    summary_plot = create_summary_plot(df_clean, df_removed, df_original)
    score_plot = create_score_breakdown_plot(df_removed)
    
    return {
        'statistics': stats,
        'summary_plot': summary_plot,
        'score_plot': score_plot
    }

