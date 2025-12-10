"""
Unified Streamlit App: Quick Prediction + Explainable AI
Combines both streamlit_multimodel.py and streamlit_xai.py into a single interface with tabs.
"""

import streamlit as st
import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file in project root
from dotenv import load_dotenv
load_dotenv(project_root / ".env", override=True)  # Force override system variables

# Import for Quick Prediction tab
from src.app.model_loader import ModelLoader

# Import for XAI tab
from src.xai.lime_baseline import LimeBaseline
from src.xai.lime_keras import LimeKeras
from src.xai.lime_bert import LimeBERT
from src.xai.lime_hybrid import LimeHybrid
import streamlit.components.v1 as components

# Import for VirusTotal integration
from src.app.urls import extract_urls, check_virustotal, check_phishtank

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
# HELPER FUNCTIONS
# ============================================================================

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
        st.markdown("### 📊 Model Performance Metrics")
        
        # Performance metrics table
        st.markdown("""
        <div style="background-color: #1e1e1e; padding: 15px; border-radius: 8px;">
        <table style="width: 100%; color: white; font-size: 12px;">
        <tr style="border-bottom: 2px solid #444;">
            <th style="text-align: left; padding: 8px;">Model</th>
            <th style="text-align: center; padding: 8px;">Acc</th>
            <th style="text-align: center; padding: 8px;">Prec</th>
            <th style="text-align: center; padding: 8px;">Rec</th>
            <th style="text-align: center; padding: 8px;">F1</th>
        </tr>
        <tr style="border-bottom: 1px solid #333;">
            <td style="padding: 8px;"><b>Baseline</b></td>
            <td style="text-align: center; padding: 8px;">96.2%</td>
            <td style="text-align: center; padding: 8px;">95.8%</td>
            <td style="text-align: center; padding: 8px;">96.5%</td>
            <td style="text-align: center; padding: 8px;">96.1%</td>
        </tr>
        <tr style="border-bottom: 1px solid #333;">
            <td style="padding: 8px;"><b>CNN</b></td>
            <td style="text-align: center; padding: 8px;">97.4%</td>
            <td style="text-align: center; padding: 8px;">97.1%</td>
            <td style="text-align: center; padding: 8px;">97.8%</td>
            <td style="text-align: center; padding: 8px;">97.4%</td>
        </tr>
        <tr style="border-bottom: 1px solid #333;">
            <td style="padding: 8px;"><b>LSTM</b></td>
            <td style="text-align: center; padding: 8px;">98.1%</td>
            <td style="text-align: center; padding: 8px;">97.9%</td>
            <td style="text-align: center; padding: 8px;">98.3%</td>
            <td style="text-align: center; padding: 8px;">98.1%</td>
        </tr>
        <tr style="border-bottom: 1px solid #333;">
            <td style="padding: 8px;"><b>BERT</b></td>
            <td style="text-align: center; padding: 8px; color: #4caf50;">99.4%</td>
            <td style="text-align: center; padding: 8px; color: #4caf50;">99.3%</td>
            <td style="text-align: center; padding: 8px; color: #4caf50;">99.5%</td>
            <td style="text-align: center; padding: 8px; color: #4caf50;">99.4%</td>
        </tr>
        <tr>
            <td style="padding: 8px;"><b>Hybrid</b></td>
            <td style="text-align: center; padding: 8px;">98.5%</td>
            <td style="text-align: center; padding: 8px;">98.3%</td>
            <td style="text-align: center; padding: 8px;">98.7%</td>
            <td style="text-align: center; padding: 8px;">98.5%</td>
        </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)
      
        
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
        
        # Enhanced Ensemble - HIDDEN
        # st.markdown("### 🎯 Enhanced Ensemble Prediction")
        # st.markdown("*Weighted average across all selected models*")
        
        # Calculate weighted ensemble (still calculated, just not displayed)
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
        
        # Display ensemble result - HIDDEN
        # if ensemble_label == "PHISHING":
        #     st.error(f"🔴 **PHISHING EMAIL DETECTED** (Confidence: {ensemble_confidence:.1f}%)")
        # else:
        #     st.success(f"🟢 **LEGITIMATE EMAIL** (Confidence: {ensemble_confidence:.1f}%)")
        
        # Detailed breakdown - HIDDEN
        # with st.expander("📊 View Detailed Breakdown"):
        #     st.markdown("#### Model Votes:")
        #     phishing_count = sum(1 for r in results.values() if r['label'] == "PHISHING")
        #     legitimate_count = len(results) - phishing_count
        #     
        #     vote_col1, vote_col2 = st.columns(2)
        #     with vote_col1:
        #         st.metric("🔴 Phishing Votes", phishing_count)
        #     with vote_col2:
        #         st.metric("🟢 Legitimate Votes", legitimate_count)
        #     
        #     st.markdown("#### Individual Probabilities:")
        #     for model_name, result in results.items():
        #         st.write(f"**{model_name.upper()}:** {result['probability']*100:.2f}% phishing")
        #     
        #     st.markdown(f"**Ensemble Probability:** {ensemble_prob*100:.2f}% phishing")
        
        # VirusTotal URL Analysis
        st.markdown("### 🔗 URL Threat Analysis")
        
        # Extract URLs from email (also get raw URLs to check for filtered ones)
        import re
        raw_url_pattern = re.compile(r'(https?://\S+|www\.\S+)', re.IGNORECASE)
        raw_urls = raw_url_pattern.findall(email_text)
        urls = extract_urls(email_text)
        
        # Check if some URLs were filtered
        if raw_urls and len(raw_urls) > len(urls):
            filtered_count = len(raw_urls) - len(urls)
            st.warning(f"ℹ️ {filtered_count} URL(s) skipped (reserved/test domains like .example, .test, localhost)")
        
        if urls:
            st.info(f"Found {len(urls)} URL(s) in email")
            
            # Check if VirusTotal API key is set
            vt_api_key = os.getenv("VT_API_KEY")
            
            if vt_api_key:
                with st.spinner("Checking URLs with VirusTotal..."):
                    for url_idx, url in enumerate(urls, 1):
                        with st.expander(f"🔗 URL {url_idx}: {url[:50]}..." if len(url) > 50 else f"🔗 URL {url_idx}: {url}", expanded=True):
                            vt_result = check_virustotal(url, get_report=True)
                            
                            if vt_result["ok"]:
                                if "report_pending" in vt_result:
                                    # Analysis submitted but report not ready
                                    st.info("✅ URL submitted to VirusTotal for analysis")
                                    st.markdown(f"**Analysis ID:** `{vt_result.get('analysis_id', 'N/A')}`")
                                    st.markdown("[Check results on VirusTotal](https://www.virustotal.com/)")
                                    
                                elif "stats" in vt_result:
                                    # Full report available
                                    stats = vt_result["stats"]
                                    malicious = vt_result.get("malicious", 0)
                                    suspicious = vt_result.get("suspicious", 0)
                                    harmless = vt_result.get("harmless", 0)
                                    undetected = vt_result.get("undetected", 0)
                                    total = vt_result.get("total_engines", 0)
                                    
                                    # Determine threat level
                                    if malicious > 0:
                                        st.error(f"🚨 **MALICIOUS URL DETECTED**")
                                        st.markdown(f"**{malicious}** out of **{total}** security vendors flagged this URL as malicious")
                                    elif suspicious > 0:
                                        st.warning(f"⚠️ **SUSPICIOUS URL**")
                                        st.markdown(f"**{suspicious}** out of **{total}** security vendors flagged this URL as suspicious")
                                    else:
                                        st.success(f"✅ **URL appears safe**")
                                        st.markdown(f"**{harmless}** out of **{total}** security vendors marked this URL as harmless")
                                    
                                    # Display statistics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("🔴 Malicious", malicious)
                                    with col2:
                                        st.metric("🟡 Suspicious", suspicious)
                                    with col3:
                                        st.metric("🟢 Harmless", harmless)
                                    with col4:
                                        st.metric("⚪ Undetected", undetected)
                                    
                                    # Add link to full report
                                    url_id = vt_result.get("analysis_id", "")
                                    if url_id:
                                        st.markdown(f"[🔍 View full report on VirusTotal](https://www.virustotal.com/gui/url/{url_id})")
                            else:
                                st.error(f"❌ VirusTotal check failed: {vt_result.get('reason', 'Unknown error')}")
            else:
                st.warning("⚠️ VirusTotal API key not set. Set `VT_API_KEY` environment variable to enable URL threat checking.")
                st.markdown("**To get a free VirusTotal API key:**")
                st.markdown("1. Sign up at [https://www.virustotal.com/](https://www.virustotal.com/)")
                st.markdown("2. Go to your profile → API Key")
                st.markdown("3. Set the environment variable: `VT_API_KEY=your_api_key`")
                
                # Show the URLs anyway
                st.markdown("**URLs found in email:**")
                for url_idx, url in enumerate(urls, 1):
                    st.code(f"{url_idx}. {url}")
        else:
            st.info("ℹ️ No URLs found in this email.")

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
        
        # LIME Parameters - HIDDEN
        # st.markdown("---")
        # st.markdown("### ⚙️ LIME Parameters")
        # num_features = st.slider("Number of features", 5, 20, 10, help="Top N words to show")
        # num_samples = st.slider("Number of samples", 100, 1000, 500, step=100, help="More = slower but more accurate")
        
        # Set default values since UI is hidden
        num_features = 10
        num_samples = 500
        
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
                    
                    # Display feature importance list
                    with st.expander("📊 View Feature Importance List", expanded=True):
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
        
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
