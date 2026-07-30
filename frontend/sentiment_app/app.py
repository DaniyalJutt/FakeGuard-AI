"""
SENTIMENT ANALYSIS SYSTEM - STREAMLIT APP
Complete 10-phase pipeline with interactive visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
import warnings
import io
import base64
from PIL import Image
warnings.filterwarnings('ignore')

# Import modules
from utils.file_handler import load_csv, export_results
from modules import (
    apply_language_detection,
    apply_text_preprocessing,
    apply_metadata_extraction,
    apply_transformer_features,
    apply_lexicon_engine,
    apply_rule_engine,
    assemble_features,
    load_models,
    make_predictions,
)

# Import Streamlit visualization functions
from modules.phase9_visualization_streamlit import (
    create_sentiment_pie_chart,
    create_sentiment_bar_chart,
    create_rating_sentiment_heatmap,
    create_confidence_distribution,
    create_sarcasm_analytics,
    create_language_distribution,
    create_tsne_visualization,
    create_emotion_distribution,
    create_metadata_analysis,
    create_wordcloud_image,
)

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .phase-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_cached():
    """Load models with caching"""
    return load_models(verbose=False)


def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'df': None,
        'processed_df': None,
        'current_phase': 0,
        'phases_complete': [],
        'models_loaded': False,
        'models': None,
        'X_exp2': None,
        'feature_columns': None,
        'visualizations_generated': False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session():
    """Reset session state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()


def run_phase(phase_num, phase_name, phase_func, *args, **kwargs):
    """Run a phase with progress tracking"""
    # Prevent duplicate phase additions
    if phase_num in st.session_state.phases_complete:
        return st.session_state.processed_df
    
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with status_placeholder:
        st.info(f"🔄 Running {phase_name}...")
    
    try:
        if args:
            result = phase_func(*args)
        else:
            result = phase_func(st.session_state.processed_df)
        
        if result is not None:
            st.session_state.processed_df = result
            # Only add if not already present (prevent duplicates)
            if phase_num not in st.session_state.phases_complete:
                st.session_state.phases_complete.append(phase_num)
            status_placeholder.success(f"✅ {phase_name} completed!")
            return result
        else:
            status_placeholder.error(f"❌ {phase_name} returned None")
            return None
    except Exception as e:
        status_placeholder.error(f"❌ Error in {phase_name}: {str(e)}")
        st.exception(e)
        return None


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">📊 Sentiment Analysis System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        # Logo
        logo_path = "assets/logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
        
        st.title("🧭 Navigation")
        
        page = st.radio(
            "Select Page",
            ["📤 Upload Data", "⚙️ Process Pipeline", "📊 Results Dashboard", "📥 Export Data"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.subheader("📈 Progress Tracker")
        
        phases = [
            (0, "Phase 0: Upload"),
            (1, "Phase 1: Language Detection"),
            (2, "Phase 2: Text Cleaning"),
            (3, "Phase 3: Metadata Extraction"),
            (4, "Phase 4: Transformer Features"),
            (5, "Phase 5: Lexicon Engine"),
            (6, "Phase 6: Rule Engine"),
            (7, "Phase 7: Feature Assembly"),
            (8, "Phase 8: Prediction"),
            (9, "Phase 9: Visualization"),
            (10, "Phase 10: Export")
        ]
        
        for phase_num, phase_name in phases:
            if phase_num in st.session_state.phases_complete:
                st.markdown(f"✅ {phase_name}")
            else:
                st.markdown(f"⏳ {phase_name}")
        
        # Overall progress
        total_phases = len(phases)
        completed = len(st.session_state.phases_complete)
        progress_pct = min(completed / total_phases, 1.0) if total_phases > 0 else 0.0
        st.progress(progress_pct, text=f"Overall: {completed}/{total_phases} phases")
        
        st.markdown("---")
        if st.button("🔄 Reset Session", use_container_width=True):
            reset_session()
            st.rerun()
    
    # Main content based on selected page
    if page == "📤 Upload Data":
        show_upload_page()
    elif page == "⚙️ Process Pipeline":
        show_processing_page()
    elif page == "📊 Results Dashboard":
        show_results_page()
    elif page == "📥 Export Data":
        show_export_page()


def show_upload_page():
    """Phase 0: Upload and validate CSV"""
    st.header("📤 Phase 0: Upload Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file with 'text' or 'text_original' column"
    )
    
    if uploaded_file is not None:
        try:
            # Load CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
            
            # Validate columns
            if 'text' not in df.columns and 'text_original' not in df.columns:
                st.error("❌ Error: 'text' or 'text_original' column required!")
                return
            
            # Standardize column name
            if 'text_original' in df.columns:
                df['text'] = df['text_original']
            
            st.session_state.df = df
            st.session_state.processed_df = df.copy()
            if 0 not in st.session_state.phases_complete:
                st.session_state.phases_complete.append(0)
            
            # Preview
            st.subheader("📝 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Show column info
            st.subheader("📋 Column Information")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Text Column", "✅ Found" if 'text' in df.columns else "❌ Missing")
            with col4:
                if 'rating' in df.columns:
                    st.metric("Has Rating", "✅ Yes")
                else:
                    st.metric("Has Rating", "❌ No")
            
            st.subheader("📊 Column Names")
            st.write(list(df.columns))
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.exception(e)


def show_processing_page():
    """Phases 1-8: Processing pipeline"""
    st.header("⚙️ Processing Pipeline")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a CSV file first in the Upload Data page.")
        return
    
    # Progress bar
    total_phases = 8
    completed = len([p for p in st.session_state.phases_complete if 1 <= p <= 8])
    progress = min(completed / total_phases, 1.0) if total_phases > 0 else 0.0
    st.progress(progress, text=f"Processing Progress: {completed}/{total_phases} phases complete")
    
    # Phase buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Run All Phases", type="primary", use_container_width=True):
            run_all_phases()
    
    with col2:
        if st.button("▶️ Run Next Phase", use_container_width=True):
            run_next_phase()
    
    with col3:
        if st.button("⏸️ Pause", use_container_width=True):
            st.info("Processing paused. Click 'Run Next Phase' to continue.")
    
    st.markdown("---")
    
    # Individual phase controls
    st.subheader("Individual Phase Controls")
    
    phases_config = [
        (1, "Phase 1: Language Detection", apply_language_detection),
        (2, "Phase 2: Text Cleaning", apply_text_preprocessing),
        (3, "Phase 3: Metadata Extraction", apply_metadata_extraction),
        (4, "Phase 4: Transformer Features", apply_transformer_features),
        (5, "Phase 5: Lexicon Engine", apply_lexicon_engine),
        (6, "Phase 6: Rule Engine", apply_rule_engine),
    ]
    
    for phase_num, phase_name, phase_func in phases_config:
        col1, col2 = st.columns([4, 1])
        with col1:
            status = "✅" if phase_num in st.session_state.phases_complete else "⏳"
            st.markdown(f"**{status} {phase_name}**")
        with col2:
            if st.button(f"Run", key=f"phase_{phase_num}", use_container_width=True):
                run_phase(phase_num, phase_name, phase_func)
                st.rerun()
    
    # Phase 7: Feature Assembly
    col1, col2 = st.columns([4, 1])
    with col1:
        status = "✅" if 7 in st.session_state.phases_complete else "⏳"
        st.markdown(f"**{status} Phase 7: Feature Assembly**")
    with col2:
        if st.button("Run Feature Assembly", key="phase_7", use_container_width=True):
            if st.session_state.processed_df is not None:
                # Prevent duplicate execution
                if 7 not in st.session_state.phases_complete:
                    try:
                        X_exp2, feature_columns, df = assemble_features(st.session_state.processed_df)
                        st.session_state.processed_df = df
                        st.session_state.X_exp2 = X_exp2
                        st.session_state.feature_columns = feature_columns
                        st.session_state.phases_complete.append(7)
                        st.success("✅ Feature Assembly completed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
                else:
                    st.info("✅ Phase 7 already completed")
    
    # Phase 8: Prediction
    col1, col2 = st.columns([4, 1])
    with col1:
        status = "✅" if 8 in st.session_state.phases_complete else "⏳"
        st.markdown(f"**{status} Phase 8: Prediction**")
    with col2:
        if st.button("Run Prediction", key="phase_8", use_container_width=True):
            # Prevent duplicate execution
            if 8 in st.session_state.phases_complete:
                st.info("✅ Phase 8 already completed")
                return
                
            if 7 in st.session_state.phases_complete:
                if not st.session_state.models_loaded:
                    with st.spinner("Loading models..."):
                        try:
                            st.session_state.models = load_models_cached()
                            st.session_state.models_loaded = True
                        except Exception as e:
                            st.error(f"❌ Error loading models: {str(e)}")
                            return
                
                if st.session_state.models and st.session_state.models[0] is not None:
                    try:
                        with st.spinner("Making predictions..."):
                            model, tfidf, svd, scaler1, scaler2, le = st.session_state.models
                            df = make_predictions(
                                st.session_state.processed_df,
                                st.session_state.X_exp2,
                                model, tfidf, svd, scaler1, scaler2, le,
                                verbose=False
                            )
                            st.session_state.processed_df = df
                            if 8 not in st.session_state.phases_complete:
                                st.session_state.phases_complete.append(8)
                            st.success("✅ Prediction completed!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
                        st.exception(e)
                else:
                    st.error("❌ Models not loaded. Please check model files.")
            else:
                st.warning("⚠️ Please complete Phase 7 first.")


def run_all_phases():
    """Run all phases sequentially"""
    if st.session_state.df is None:
        st.error("❌ Please upload a CSV file first.")
        return
    
    phases = [
        (1, "Language Detection", apply_language_detection),
        (2, "Text Cleaning", apply_text_preprocessing),
        (3, "Metadata Extraction", apply_metadata_extraction),
        (4, "Transformer Features", apply_transformer_features),
        (5, "Lexicon Engine", apply_lexicon_engine),
        (6, "Rule Engine", apply_rule_engine),
    ]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (phase_num, phase_name, phase_func) in enumerate(phases):
        status_text.text(f"Running {phase_name}... ({idx+1}/{len(phases)})")
        result = run_phase(phase_num, phase_name, phase_func)
        if result is None:
            st.error(f"Failed at {phase_name}")
            return
        progress_value = min((idx + 1) / (len(phases) + 2), 1.0)
        progress_bar.progress(progress_value)
    
    # Phase 7: Feature Assembly
    if 7 not in st.session_state.phases_complete:
        status_text.text("Running Feature Assembly...")
        if st.session_state.processed_df is not None:
            try:
                X_exp2, feature_columns, df = assemble_features(st.session_state.processed_df)
                st.session_state.processed_df = df
                st.session_state.X_exp2 = X_exp2
                st.session_state.feature_columns = feature_columns
                if 7 not in st.session_state.phases_complete:
                    st.session_state.phases_complete.append(7)
            except Exception as e:
                st.error(f"Error in Feature Assembly: {str(e)}")
                return
    
    progress_bar.progress(min(7 / 8, 1.0))
    
    # Phase 8: Prediction
    if 8 not in st.session_state.phases_complete:
        status_text.text("Loading models and making predictions...")
        if not st.session_state.models_loaded:
            try:
                st.session_state.models = load_models_cached()
                st.session_state.models_loaded = True
            except Exception as e:
                st.error(f"Error loading models: {str(e)}")
                return
        
        if st.session_state.models and st.session_state.models[0] is not None:
            try:
                model, tfidf, svd, scaler1, scaler2, le = st.session_state.models
                df = make_predictions(
                    st.session_state.processed_df,
                    st.session_state.X_exp2,
                    model, tfidf, svd, scaler1, scaler2, le,
                    verbose=False
                )
                st.session_state.processed_df = df
                if 8 not in st.session_state.phases_complete:
                    st.session_state.phases_complete.append(8)
            except Exception as e:
                st.error(f"Error in Prediction: {str(e)}")
                return
    
    progress_bar.progress(1.0)
    status_text.text("✅ All phases completed!")
    st.success("🎉 Pipeline execution complete!")
    st.rerun()


def run_next_phase():
    """Run the next incomplete phase"""
    if st.session_state.df is None:
        st.error("❌ Please upload a CSV file first.")
        return
    
    next_phase = None
    for i in range(1, 9):
        if i not in st.session_state.phases_complete:
            next_phase = i
            break
    
    if next_phase is None:
        st.info("✅ All phases are complete!")
        return
    
    phases_map = {
        1: ("Language Detection", apply_language_detection),
        2: ("Text Cleaning", apply_text_preprocessing),
        3: ("Metadata Extraction", apply_metadata_extraction),
        4: ("Transformer Features", apply_transformer_features),
        5: ("Lexicon Engine", apply_lexicon_engine),
        6: ("Rule Engine", apply_rule_engine),
    }
    
    if next_phase in phases_map:
        phase_name, phase_func = phases_map[next_phase]
        run_phase(next_phase, phase_name, phase_func)
    elif next_phase == 7:
        if 7 not in st.session_state.phases_complete:
            try:
                X_exp2, feature_columns, df = assemble_features(st.session_state.processed_df)
                st.session_state.processed_df = df
                st.session_state.X_exp2 = X_exp2
                st.session_state.feature_columns = feature_columns
                if 7 not in st.session_state.phases_complete:
                    st.session_state.phases_complete.append(7)
                st.success("✅ Feature Assembly completed!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    elif next_phase == 8:
        if 8 not in st.session_state.phases_complete:
            if not st.session_state.models_loaded:
                try:
                    st.session_state.models = load_models_cached()
                    st.session_state.models_loaded = True
                except Exception as e:
                    st.error(f"Error loading models: {str(e)}")
                    return
            
            if st.session_state.models and st.session_state.models[0] is not None:
                try:
                    model, tfidf, svd, scaler1, scaler2, le = st.session_state.models
                    df = make_predictions(
                        st.session_state.processed_df,
                        st.session_state.X_exp2,
                        model, tfidf, svd, scaler1, scaler2, le,
                        verbose=False
                    )
                    st.session_state.processed_df = df
                    if 8 not in st.session_state.phases_complete:
                        st.session_state.phases_complete.append(8)
                    st.success("✅ Prediction completed!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    st.rerun()


def show_results_page():
    """Phase 9: Results Dashboard with Visualizations"""
    st.header("📊 Results Dashboard")
    
    if st.session_state.processed_df is None or 8 not in st.session_state.phases_complete:
        st.warning("⚠️ Please complete the processing pipeline first.")
        return
    
    df = st.session_state.processed_df
    
    # Metrics
    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total = len(df)
        st.metric("Total Reviews", f"{total:,}")
    
    with col2:
        if 'sentiment_final' in df.columns:
            positive = (df['sentiment_final'] == 'positive').sum()
            st.metric("Positive", f"{positive:,}", f"{positive/total*100:.1f}%")
    
    with col3:
        if 'sentiment_final' in df.columns:
            negative = (df['sentiment_final'] == 'negative').sum()
            st.metric("Negative", f"{negative:,}", f"{negative/total*100:.1f}%")
    
    with col4:
        if 'sentiment_final' in df.columns:
            neutral = (df['sentiment_final'] == 'neutral').sum()
            st.metric("Neutral", f"{neutral:,}", f"{neutral/total*100:.1f}%")
    
    with col5:
        if 'prediction_confidence' in df.columns:
            avg_conf = df['prediction_confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.2%}")
    
    st.markdown("---")
    
    # Visualizations
    if 'sentiment_final' in df.columns:
        # Sentiment Distribution
        st.subheader("📊 Sentiment Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = create_sentiment_pie_chart(df)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = create_sentiment_bar_chart(df)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Rating vs Sentiment Heatmap
        if 'rating' in df.columns:
            st.subheader("🔥 Rating vs Sentiment Heatmap")
            fig_heatmap = create_rating_sentiment_heatmap(df)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Wordclouds
        st.subheader("☁️ Wordclouds by Sentiment")
        sentiments = ['positive', 'negative', 'neutral']
        cols = st.columns(3)
        
        for idx, sentiment in enumerate(sentiments):
            if sentiment in df['sentiment_final'].values:
                with cols[idx]:
                    text = ' '.join(df[df['sentiment_final'] == sentiment]['text_cleaned'].astype(str).values)
                    if text.strip():
                        img = create_wordcloud_image(text, sentiment)
                        if img:
                            st.image(img, caption=f"{sentiment.capitalize()} Reviews")
        
        # t-SNE Visualization
        st.subheader("🎯 t-SNE 2D Embedding Visualization")
        with st.spinner("Computing t-SNE visualization (this may take a moment)..."):
            fig_tsne = create_tsne_visualization(df, max_samples=1000)
            if fig_tsne:
                st.plotly_chart(fig_tsne, use_container_width=True)
            else:
                st.info("t-SNE visualization not available. Ensure transformer features are computed.")
        
        # Confidence Distribution
        if 'prediction_confidence' in df.columns:
            st.subheader("📊 Confidence Distribution")
            fig_conf = create_confidence_distribution(df)
            if fig_conf:
                st.plotly_chart(fig_conf, use_container_width=True)
        
        # Emotion Distribution
        st.subheader("😊 Emotion Distribution")
        fig_emotion = create_emotion_distribution(df)
        if fig_emotion:
            st.plotly_chart(fig_emotion, use_container_width=True)
        
        # Sarcasm Analytics
        if 'sarcasm_detected' in df.columns:
            st.subheader("😏 Sarcasm Analytics")
            fig_sarc = create_sarcasm_analytics(df)
            if fig_sarc:
                st.plotly_chart(fig_sarc, use_container_width=True)
        
        # Language Distribution
        if 'language' in df.columns:
            st.subheader("🌐 Language Distribution")
            fig_lang = create_language_distribution(df)
            if fig_lang:
                st.plotly_chart(fig_lang, use_container_width=True)
        
        # Metadata Analysis
        st.subheader("📈 Metadata Analysis")
        fig_meta = create_metadata_analysis(df)
        if fig_meta:
            st.plotly_chart(fig_meta, use_container_width=True)
    
    # Data Explorer
    st.markdown("---")
    st.subheader("🔍 Data Explorer")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if 'sentiment_final' in df.columns:
            selected_sentiments = st.multiselect(
                "Filter by Sentiment",
                options=df['sentiment_final'].unique(),
                default=df['sentiment_final'].unique()
            )
        else:
            selected_sentiments = []
    
    with col2:
        if 'language' in df.columns:
            selected_languages = st.multiselect(
                "Filter by Language",
                options=df['language'].unique(),
                default=df['language'].unique()
            )
        else:
            selected_languages = []
    
    with col3:
        if 'prediction_confidence' in df.columns:
            min_confidence = st.slider(
                "Min Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )
        else:
            min_confidence = 0.0
    
    with col4:
        max_rows = st.number_input("Max Rows to Display", min_value=10, max_value=1000, value=100, step=10)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_sentiments and 'sentiment_final' in df.columns:
        filtered_df = filtered_df[filtered_df['sentiment_final'].isin(selected_sentiments)]
    if selected_languages and 'language' in df.columns:
        filtered_df = filtered_df[filtered_df['language'].isin(selected_languages)]
    if 'prediction_confidence' in df.columns:
        filtered_df = filtered_df[filtered_df['prediction_confidence'] >= min_confidence]
    
    display_cols = ['text', 'sentiment_final', 'prediction_confidence']
    if 'language' in filtered_df.columns:
        display_cols.append('language')
    if 'rating' in filtered_df.columns:
        display_cols.append('rating')
    
    available_cols = [col for col in display_cols if col in filtered_df.columns]
    
    st.dataframe(
        filtered_df[available_cols].head(max_rows),
        use_container_width=True,
        height=400
    )


def show_export_page():
    """Phase 10: Export Results"""
    st.header("📥 Phase 10: Export Results")
    
    if st.session_state.processed_df is None or 8 not in st.session_state.phases_complete:
        st.warning("⚠️ Please complete the processing pipeline first.")
        return
    
    df = st.session_state.processed_df
    
    # Export options
    st.subheader("📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Full dataset CSV
        csv_full = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=csv_full,
            file_name="processed_dataset_full.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Predictions only CSV
        if 'sentiment_final' in df.columns and 'prediction_confidence' in df.columns:
            csv_pred = df[['text', 'sentiment_final', 'prediction_confidence']].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions Only (CSV)",
                data=csv_pred,
                file_name="predictions_only.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    
    if 'sentiment_final' in df.columns:
        summary = {
            'total_reviews': len(df),
            'sentiment_distribution': df['sentiment_final'].value_counts().to_dict(),
            'avg_confidence': float(df['prediction_confidence'].mean()) if 'prediction_confidence' in df.columns else 0.0,
            'sarcasm_rate': float(df['sarcasm_detected'].mean()) if 'sarcasm_detected' in df.columns else 0.0,
            'language_distribution': df['language'].value_counts().to_dict() if 'language' in df.columns else {}
        }
        
        st.json(summary)
        
        # Download summary
        summary_pkl = pickle.dumps(summary)
        st.download_button(
            label="📥 Download Summary (PKL)",
            data=summary_pkl,
            file_name="summary_stats.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )
    
    # Export to data/exports directory
    st.subheader("💾 Save to Local Directory")
    if st.button("💾 Export All Results to data/exports/", use_container_width=True):
        try:
            os.makedirs('data/exports', exist_ok=True)
            export_results(df)
            st.success("✅ Results exported to data/exports/ directory!")
        except Exception as e:
            st.error(f"❌ Error exporting: {str(e)}")


if __name__ == "__main__":
    main()
