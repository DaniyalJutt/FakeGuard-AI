"""
PHASE 9: VISUALIZATION
Creates charts and visualizations for sentiment analysis results
"""

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import os


def setup_visualization():
    """Setup visualization libraries and style"""
    print("\n" + "="*80)
    print("PHASE 9: VISUALIZATION SETUP")
    print("="*80)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create exports directory if it doesn't exist
    os.makedirs('data/exports', exist_ok=True)
    
    print("✅ Visualization libraries loaded")


def plot_sentiment_distribution(df):
    """Create sentiment distribution charts"""
    print("\n📊 Creating Sentiment Distribution Charts...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart
    sentiment_counts = df['sentiment_final'].value_counts()
    ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, 
            autopct='%1.1f%%', startangle=90)
    ax1.set_title('Sentiment Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    # Bar chart
    sentiment_counts.plot(kind='bar', ax=ax2, color=['#2ecc71', '#e74c3c', '#95a5a6'])
    ax2.set_title('Sentiment Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sentiment')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('data/exports/sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Sentiment distribution charts saved")


def plot_rating_sentiment_heatmap(df):
    """Create rating vs sentiment heatmap"""
    print("\n📊 Creating Rating vs Sentiment Heatmap...")
    
    if 'rating' in df.columns:
        # Create crosstab
        rating_sentiment = pd.crosstab(df['rating'], df['sentiment_final'])
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(rating_sentiment, annot=True, fmt='d', cmap='YlGnBu', 
                    cbar_kws={'label': 'Count'})
        plt.title('Rating vs Sentiment Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment')
        plt.ylabel('Rating')
        plt.tight_layout()
        plt.savefig('data/exports/rating_sentiment_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Rating vs Sentiment heatmap saved")
    else:
        print("⚠️  Rating column not available")


def plot_wordclouds(df):
    """Create wordclouds for each sentiment"""
    print("\n📊 Creating Wordclouds...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (sentiment, ax) in enumerate(zip(['positive', 'negative', 'neutral'], axes)):
        if sentiment in df['sentiment_final'].values:
            text = ' '.join(df[df['sentiment_final'] == sentiment]['text_cleaned'].values)
            
            if text.strip():
                wordcloud = WordCloud(width=600, height=400, 
                                    background_color='white',
                                    colormap='viridis').generate(text)
                
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'{sentiment.capitalize()} Reviews', 
                            fontsize=14, fontweight='bold')
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'No {sentiment} reviews', 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('data/exports/wordclouds.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Wordclouds saved")


def plot_confidence_distribution(df):
    """Create confidence distribution histogram"""
    print("\n📊 Creating Confidence Distribution...")
    
    plt.figure(figsize=(10, 6))
    plt.hist(df['prediction_confidence'], bins=30, edgecolor='black', 
             color='skyblue', alpha=0.7)
    plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.axvline(df['prediction_confidence'].mean(), color='red', 
               linestyle='--', label=f'Mean: {df["prediction_confidence"].mean():.2f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('data/exports/confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Confidence distribution saved")


def plot_sarcasm_analytics(df):
    """Create sarcasm analytics charts"""
    print("\n📊 Creating Sarcasm Analytics...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sarcasm count
    sarcasm_counts = df['sarcasm_detected'].value_counts()
    ax1.bar(['Not Sarcastic', 'Sarcastic'], 
            [sarcasm_counts.get(0, 0), sarcasm_counts.get(1, 0)],
            color=['#3498db', '#e74c3c'])
    ax1.set_title('Sarcasm Detection', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    
    # Sarcasm by rating
    if 'rating' in df.columns:
        sarcasm_by_rating = df.groupby('rating')['sarcasm_detected'].mean() * 100
        sarcasm_by_rating.plot(kind='bar', ax=ax2, color='coral')
        ax2.set_title('Sarcasm % by Rating', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Rating')
        ax2.set_ylabel('Sarcasm %')
        ax2.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('data/exports/sarcasm_analytics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Sarcasm analytics saved")
    print(f"✅ Total sarcastic reviews: {sarcasm_counts.get(1, 0)} ({df['sarcasm_detected'].mean()*100:.1f}%)")


def generate_all_visualizations(df):
    """Generate all visualizations"""
    setup_visualization()
    plot_sentiment_distribution(df)
    plot_rating_sentiment_heatmap(df)
    plot_wordclouds(df)
    plot_confidence_distribution(df)
    plot_sarcasm_analytics(df)

