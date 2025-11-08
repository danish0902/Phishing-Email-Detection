"""
Unified Streamlit App: Quick Prediction + Explainable AI
Combines both streamlit_multimodel.py and streamlit_xai.py into a single interface with tabs.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import for Quick Prediction tab
from src.app.model_loader import ModelLoader

# Import for XAI tab
from src.xai.lime_baseline import LimeBaseline
from src.xai.lime_keras import LimeKeras
from src.xai.lime_bert import LimeBERT
from src.xai.lime_hybrid import LimeHybrid
import streamlit.components.v1 as components
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Phishing Email Detection - Unified",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SHARED CACHE FOR MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load all models once and cache them"""
    return ModelLoader()

# ============================================================================
# TAB 1: QUICK PREDICTION
# ============================================================================

def quick_prediction_tab():
    """Quick Prediction Interface (from streamlit_multimodel.py)"""
    
    st.title("⚡ Quick Phishing Email Detection")
    st.markdown("*Fast, production-ready predictions across multiple models*")
    
    # Load models
    loader = load_models()
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Model Selection")
        st.markdown("Select one or more models:")
        
        model_selection = {
            "baseline": st.checkbox("TF-IDF + Logistic Regression", value=True, help="Fast baseline model"),
            "cnn": st.checkbox("CNN", value=True, help="Convolutional Neural Network"),
            "lstm": st.checkbox("LSTM", value=True, help="Long Short-Term Memory"),
            "bert": st.checkbox("BERT (DistilBERT)", value=True, help="Transformer-based model"),
            "hybrid": st.checkbox("Hybrid CNN+LSTM+URL", value=True, help="Multi-input model")
        }
        
        selected_models = [k for k, v in model_selection.items() if v]
        
        if not selected_models:
            st.warning("⚠️ Please select at least one model")
            return
        
        st.markdown("---")
        st.markdown("### 📊 Model Performance")
        st.markdown("""
        **Accuracy Scores:**
        - TF-IDF+LR: ~96%
        - CNN: ~97%
        - LSTM: ~98%
        - BERT: ~99%
        - Hybrid: ~98%
        """)
        
        st.markdown("---")
        st.markdown("### 🎚️ Prediction Threshold")
        threshold = st.slider(
            "Classification threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Probability threshold for classifying as PHISHING. Lower = more sensitive to phishing, Higher = more conservative"
        )
        st.caption(f"Current: {threshold:.2f} - Emails with phishing probability ≥ {threshold:.2f} will be classified as PHISHING")
    
    # Main area
    st.markdown("### 📧 Enter Email Content")
    email_text = st.text_area(
        "Paste your email text below:",
        height=200,
        placeholder="Enter the email content you want to analyze...",
        help="The email content will be analyzed by all selected models"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
    
    if analyze_button:
        if not email_text.strip():
            st.error("❌ Please enter some email text to analyze")
            return
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        total_models = len(selected_models)
        
        # Map UI model names to internal model names
        model_name_map = {
            "baseline": "tfidf",
            "cnn": "cnn",
            "lstm": "lstm",
            "bert": "bert",
            "hybrid": "hybrid"
        }
        
        for idx, model_name in enumerate(selected_models):
            status_text.text(f"Analyzing with {model_name.upper()}...")
            progress_bar.progress((idx + 1) / total_models)
            
            # Map to internal model name
            internal_model_name = model_name_map.get(model_name, model_name)
            
            # Get probability from model (returns phishing probability)
            proba = loader.predict(email_text, internal_model_name)
            
            # Convert to label using user-defined threshold
            pred = 1 if proba >= threshold else 0
            label = "PHISHING" if pred == 1 else "LEGITIMATE"
            
            results[model_name] = {
                'prediction': pred,
                'probability': proba,
                'label': label
            }
        
        status_text.text("✅ Analysis Complete!")
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        
        
        st.markdown("### 🎯 Individual Model Predictions")
        
        cols = st.columns(min(len(selected_models), 3))
        
        for idx, (model_name, result) in enumerate(results.items()):
            col_idx = idx % 3
            with cols[col_idx]:
                pred_class = result['label']
                confidence = result['probability'] * 100
                
                # Color coding
                if pred_class == "PHISHING":
                    color = "🔴"
                    bg_color = "#414040"
                else:
                    color = "🟢"
                    bg_color = "#414040"
                
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                    <h4>{color} {model_name.upper()}</h4>
                    <p style="font-size: 24px; margin: 0;"><b>{pred_class}</b></p>
                    <p style="margin: 0;">Confidence: <b>{confidence:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
        
        # Close the light grey container
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Enhanced Ensemble
        st.markdown("### 🎯 Enhanced Ensemble Prediction")
        st.markdown("*Weighted average across all selected models*")
        
        # Calculate weighted ensemble
        weights = {
            'baseline': 0.15,
            'cnn': 0.20,
            'lstm': 0.20,
            'bert': 0.25,
            'hybrid': 0.20
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for model_name, result in results.items():
            weight = weights.get(model_name, 0.2)
            weighted_sum += result['probability'] * weight
            total_weight += weight
        
        ensemble_prob = weighted_sum / total_weight if total_weight > 0 else 0.5
        ensemble_label = "PHISHING" if ensemble_prob >= threshold else "LEGITIMATE"
        ensemble_confidence = max(ensemble_prob, 1 - ensemble_prob) * 100
        
        # Display ensemble result
        if ensemble_label == "PHISHING":
            st.error(f"🔴 **PHISHING EMAIL DETECTED** (Confidence: {ensemble_confidence:.1f}%)")
        else:
            st.success(f"🟢 **LEGITIMATE EMAIL** (Confidence: {ensemble_confidence:.1f}%)")
        
        # Detailed breakdown
        with st.expander("📊 View Detailed Breakdown"):
            st.markdown("#### Model Votes:")
            phishing_count = sum(1 for r in results.values() if r['label'] == "PHISHING")
            legitimate_count = len(results) - phishing_count
            
            vote_col1, vote_col2 = st.columns(2)
            with vote_col1:
                st.metric("🔴 Phishing Votes", phishing_count)
            with vote_col2:
                st.metric("🟢 Legitimate Votes", legitimate_count)
            
            st.markdown("#### Individual Probabilities:")
            for model_name, result in results.items():
                st.write(f"**{model_name.upper()}:** {result['probability']*100:.2f}% phishing")
            
            st.markdown(f"**Ensemble Probability:** {ensemble_prob*100:.2f}% phishing")

# ============================================================================
# TAB 2: EXPLAINABLE AI
# ============================================================================

def xai_tab():
    """Explainable AI Interface (LIME explanations only)"""
    
    st.title("🔍 Explainable AI for Phishing Detection")
    st.markdown("*Understand why models classify emails as phishing or legitimate using LIME*")
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 XAI Configuration")
        
        st.markdown("### 📊 Select Models")
        model_selection = st.multiselect(
            "Choose models to explain:",
            ["baseline", "cnn", "lstm", "bert", "hybrid"],
            default=["baseline"],
            help="Select one or more models to generate explanations"
        )
        
        st.markdown("---")
        st.markdown("### 🔬 Explanation Methods")
        
        use_lime = st.checkbox("LIME (Local Interpretable Model-agnostic Explanations)", value=True, help="10-30 seconds per model")
        
        st.markdown("---")
        st.markdown("### ⚙️ LIME Parameters")
        num_features = st.slider("Number of features", 5, 20, 10, help="Top N words to show")
        num_samples = st.slider("Number of samples", 100, 1000, 500, step=100, help="More = slower but more accurate")
        
        st.markdown("---")
        st.markdown("### 💾 Save Options")
        save_html = st.checkbox("Save LIME HTML", value=False, help="Save to artifacts/explanations/")
    
    # Main area
    st.markdown("### 📧 Enter Email Content")
    email_text = st.text_area(
        "Paste your email text below:",
        height=200,
        placeholder="Enter the email content you want to explain...",
        help="The model's decision will be explained with word importance"
    )
    
    explain_button = st.button("🔍 Generate Explanations", type="primary")
    
    if explain_button:
        if not email_text.strip():
            st.error("❌ Please enter some email text to analyze")
            return
        
        if not model_selection:
            st.error("❌ Please select at least one model")
            return
        
        st.markdown("---")
        
        # Process each model
        for model_name in model_selection:
            st.markdown(f"## 🎯 {model_name.upper()} Explanations")
            
            # LIME Explanations
            
            
            with st.spinner(f"Generating LIME explanation for {model_name}..."):
                try:
                    # Initialize appropriate explainer
                    if model_name == "baseline":
                        explainer = LimeBaseline()
                    elif model_name in ["cnn", "lstm"]:
                        explainer = LimeKeras(model_type=model_name)
                    elif model_name == "bert":
                        explainer = LimeBERT()
                    elif model_name == "hybrid":
                        explainer = LimeHybrid()
                    
                    # Get prediction first
                    pred_proba = explainer.predict_proba([email_text])[0]
                    pred_class = 1 if pred_proba[1] >= 0.5 else 0
                    pred_label = "PHISHING" if pred_class == 1 else "LEGITIMATE"
                    confidence = max(pred_proba) * 100
                    
                    # Display prediction with better visibility
                    if pred_label == "PHISHING":
                        st.markdown(f"""
                        <div style="background-color: #fff5f5; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4444; margin: 15px 0;">
                            <h3 style="color: #ffffff; background-color: #c62828; padding: 10px; border-radius: 5px; margin: 0;">🔴 Prediction: PHISHING</h3>
                            <p style="font-size: 18px; color: #ffffff; background-color: #c62828; padding: 8px; border-radius: 5px; margin: 10px 0 0 0;">Confidence: <strong>{confidence:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #f1f8f4; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50; margin: 15px 0;">
                            <h3 style="color: #ffffff; background-color: #2e7d32; padding: 10px; border-radius: 5px; margin: 0;">🟢 Prediction: LEGITIMATE</h3>
                            <p style="font-size: 18px; color: #ffffff; background-color: #2e7d32; padding: 8px; border-radius: 5px; margin: 10px 0 0 0;">Confidence: <strong>{confidence:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display explanation with white background
        
                    
                    # Generate explanation (don't save automatically)
                    html_exp = explainer.explain_html(
                        email_text,
                        num_features=num_features,
                        num_samples=num_samples,
                        save_artifacts=False
                    )
                    
                    # Inject custom styling to add white background to LIME content
                    html_exp_styled = f"""
                    <style>
                        body {{
                            background-color: white !important;
                            padding: 15px;
                        }}
                        .lime {{
                            background-color: white !important;
                        }}
                    </style>
                    {html_exp}
                    """
                    
                    # Display explanation
                    components.html(html_exp_styled, height=400, scrolling=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Save if requested
                    if save_html:
                        output_dir = project_root / "artifacts" / "explanations"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_path = output_dir / f"lime_{model_name}.html"
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(html_exp)
                        
                        st.success(f"✅ Saved to: {output_path}")
                    
                    # Show feature importance list
                    with st.expander("📊 View Feature Importance List"):
                        st.markdown("""
                        <div style="background-color: #424242; padding: 15px; border-radius: 8px;">
                        """, unsafe_allow_html=True)
                        
                        exp_list, pred, prob = explainer.explain_to_list(
                            email_text,
                            num_features=num_features,
                            num_samples=num_samples
                        )
                        
                        for word, score in exp_list:
                            direction = "🔴 Phishing" if score > 0 else "🟢 Legitimate"
                            if score > 0:
                                bg_color = "#d32f2f"
                                text_color = "#ffffff"
                            else:
                                bg_color = "#388e3c"
                                text_color = "#ffffff"
                            st.markdown(f"""
                            <div style="background-color: {bg_color}; padding: 8px; margin: 5px 0; border-radius: 5px;">
                                <span style="color: {text_color};"><strong>{word}</strong>: {score:.4f} ({direction})</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.success(f"✅ LIME explanation generated for {model_name}")
                    
                except Exception as e:
                    st.error(f"❌ Error generating LIME explanation: {str(e)}")
            
            st.markdown("---")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application with tabs"""
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; text-align: center; margin: 0;">
            🎯 Phishing Email Detection System
        </h1>
        <p style="color: white; text-align: center; margin: 5px 0 0 0;">
            Unified Interface: Quick Predictions + Explainable AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["⚡ Quick Prediction", "🔍 Explainable AI"])
    
    with tab1:
        quick_prediction_tab()
    
    with tab2:
        xai_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 14px;">
        <p>
            💡 <b>Tip:</b> Use Quick Prediction for fast analysis, switch to XAI tab for detailed explanations
        </p>
        <p>
            📚 Documentation: <code>STREAMLIT_APPS_GUIDE.md</code> | 
            🔧 XAI Docs: <code>src/xai/README.md</code>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
