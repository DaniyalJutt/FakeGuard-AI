"""
PHASE 9: STREAMLIT VISUALIZATIONS
Interactive Plotly visualizations for Streamlit app
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.validators.scatter3d import MarkerValidator
# ✅ Working correctly
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import io
import base64
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def create_sentiment_pie_chart(df):
    """Create interactive sentiment pie chart"""
    if 'sentiment_final' not in df.columns:
        return None
    
    sentiment_counts = df['sentiment_final'].value_counts()
    
    colors = {
        'positive': '#2ecc71',
        'negative': '#e74c3c',
        'neutral': '#95a5a6'
    }
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_map=colors
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True, height=400)
    
    return fig


def create_sentiment_bar_chart(df):
    """Create interactive sentiment bar chart"""
    if 'sentiment_final' not in df.columns:
        return None
    
    sentiment_counts = df['sentiment_final'].value_counts()
    
    colors = {
        'positive': '#2ecc71',
        'negative': '#e74c3c',
        'neutral': '#95a5a6'
    }
    
    fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        title="Sentiment Counts",
        labels={'x': 'Sentiment', 'y': 'Count'},
        color=sentiment_counts.index,
        color_discrete_map=colors
    )
    fig.update_layout(showlegend=False, height=400)
    
    return fig


def create_rating_sentiment_heatmap(df):
    """Create rating vs sentiment heatmap"""
    if 'rating' not in df.columns or 'sentiment_final' not in df.columns:
        return None
    
    rating_sentiment = pd.crosstab(df['rating'], df['sentiment_final'])
    
    fig = px.imshow(
        rating_sentiment,
        labels=dict(x="Sentiment", y="Rating", color="Count"),
        title="Rating vs Sentiment Heatmap",
        aspect="auto",
        text_auto=True,
        color_continuous_scale='YlGnBu'
    )
    fig.update_layout(height=500)
    
    return fig


def create_wordcloud_image(text, sentiment):
    """Create wordcloud image for a sentiment"""
    if not text or len(text.strip()) < 10:
        return None
    
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)
        
        # Convert to image
        img = wordcloud.to_image()
        return img
    except Exception as e:
        print(f"Error creating wordcloud: {e}")
        return None


def create_confidence_distribution(df):
    """Create confidence distribution histogram"""
    if 'prediction_confidence' not in df.columns:
        return None
    
    fig = px.histogram(
        df,
        x='prediction_confidence',
        nbins=30,
        title="Prediction Confidence Distribution",
        labels={'prediction_confidence': 'Confidence Score', 'count': 'Frequency'},
        color_discrete_sequence=['skyblue']
    )
    
    # Add mean line
    mean_conf = df['prediction_confidence'].mean()
    fig.add_vline(
        x=mean_conf,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_conf:.2f}"
    )
    
    fig.update_layout(height=400)
    return fig


def create_sarcasm_analytics(df):
    """Create sarcasm analytics charts"""
    if 'sarcasm_detected' not in df.columns:
        return None
    
    sarcasm_counts = df['sarcasm_detected'].value_counts()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Sarcasm Detection', 'Sarcasm % by Rating'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Sarcasm count
    fig.add_trace(
        go.Bar(
            x=['Not Sarcastic', 'Sarcastic'],
            y=[sarcasm_counts.get(0, 0), sarcasm_counts.get(1, 0)],
            marker_color=['#3498db', '#e74c3c'],
            name='Count'
        ),
        row=1, col=1
    )
    
    # Sarcasm by rating
    if 'rating' in df.columns:
        sarcasm_by_rating = df.groupby('rating')['sarcasm_detected'].mean() * 100
        fig.add_trace(
            go.Bar(
                x=sarcasm_by_rating.index.astype(str),
                y=sarcasm_by_rating.values,
                marker_color='coral',
                name='Sarcasm %'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title_text="Sarcasm Analytics",
        showlegend=False,
        height=400
    )
    fig.update_xaxes(title_text="Category", row=1, col=1)
    fig.update_xaxes(title_text="Rating", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Sarcasm %", row=1, col=2)
    
    return fig


def create_language_distribution(df):
    """Create language distribution chart"""
    if 'language' not in df.columns:
        return None
    
    lang_counts = df['language'].value_counts()
    
    fig = px.bar(
        x=lang_counts.index,
        y=lang_counts.values,
        title="Language Distribution",
        labels={'x': 'Language', 'y': 'Count'},
        color=lang_counts.index,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(showlegend=False, height=400)
    
    return fig


def create_tsne_visualization(df, max_samples=1000):
    """Create t-SNE 2D visualization of embeddings"""
    if 'transformer_emb_mean' not in df.columns or 'transformer_emb_std' not in df.columns:
        return None
    
    # Sample data if too large
    sample_df = df.sample(min(max_samples, len(df))) if len(df) > max_samples else df
    
    # Prepare features for t-SNE
    features = sample_df[['transformer_emb_mean', 'transformer_emb_std', 
                          'transformer_sentiment_score', 'transformer_confidence']].values
    
    # Use PCA first for dimensionality reduction if needed
    if features.shape[1] > 50:
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)
    
    # Apply t-SNE
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sample_df)-1))
        embeddings_2d = tsne.fit_transform(features)
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'sentiment': sample_df['sentiment_final'].values if 'sentiment_final' in sample_df.columns else 'unknown'
        })
        
        # Create scatter plot
        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='sentiment',
            title="t-SNE 2D Embedding Visualization",
            labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'},
            color_discrete_map={
                'positive': '#2ecc71',
                'negative': '#e74c3c',
                'neutral': '#95a5a6'
            }
        )
        fig.update_layout(height=600)
        
        return fig
    except Exception as e:
        print(f"Error creating t-SNE visualization: {e}")
        return None


def create_emotion_distribution(df):
    """Create emotion/confidence distribution"""
    if 'emotion_confidence' not in df.columns and 'prediction_confidence' not in df.columns:
        return None
    
    conf_col = 'emotion_confidence' if 'emotion_confidence' in df.columns else 'prediction_confidence'
    
    fig = px.box(
        df,
        x='sentiment_final' if 'sentiment_final' in df.columns else None,
        y=conf_col,
        title="Emotion Confidence Distribution by Sentiment",
        labels={conf_col: 'Confidence Score'},
        color='sentiment_final' if 'sentiment_final' in df.columns else None,
        color_discrete_map={
            'positive': '#2ecc71',
            'negative': '#e74c3c',
            'neutral': '#95a5a6'
        } if 'sentiment_final' in df.columns else None
    )
    fig.update_layout(height=400)
    
    return fig


def create_metadata_analysis(df):
    """Create metadata analysis charts"""
    metadata_cols = ['meta_text_length', 'meta_word_count', 'meta_emoji_count']
    available_cols = [col for col in metadata_cols if col in df.columns]
    
    if not available_cols:
        return None
    
    fig = make_subplots(
        rows=1, cols=len(available_cols),
        subplot_titles=available_cols
    )
    
    for idx, col in enumerate(available_cols):
        fig.add_trace(
            go.Histogram(x=df[col], name=col, showlegend=False),
            row=1, col=idx+1
        )
    
    fig.update_layout(
        title_text="Metadata Analysis",
        height=400
    )
    
    return fig
