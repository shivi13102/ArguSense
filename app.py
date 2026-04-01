import streamlit as st
import os
import pandas as pd
import config
from src.predict import ArguSensePredictor
from src.utils import load_sample_texts, get_metrics_summary

# --- Styling & Layout ---
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Predictor ---
@st.cache_resource
def get_predictor():
    return ArguSensePredictor()

predictor = get_predictor()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title(f"ArguSense v3.0")
    st.markdown("Sarcasm-Aware Argument Analysis Framework")
    st.divider()
    page = st.radio("Navigation", ["Prediction Dashboard", "Model Benchmarks", "Methodology"])
    st.divider()
    st.info("Classical NLP Pipeline (Local Models)")

# --- Helper UI Components ---
def result_metric(title, value, conf, color="#1E88E5"):
    st.markdown(f"""
    <div style="background-color:#f8f9fa; border-left: 5px solid {color}; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <p style="color:#666; margin:0; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px;">{title}</p>
        <h2 style="color:{color}; margin:10px 0;">{value}</h2>
        <div style="display:flex; align-items:center;">
            <div style="background-color:#ddd; height:8px; width:100px; border-radius:4px; margin-right:10px;">
                <div style="background-color:{color}; height:8px; width:{conf*100}px; border-radius:4px;"></div>
            </div>
            <span style="font-size:0.8rem; color:#888;">{conf:.1%} confidence</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Prediction Page ---
if page == "Prediction Dashboard":
    st.title("🏹 Analysis Dashboard")
    st.markdown("Detect sarcasm and evaluate argument effectiveness simultaneously.")
    
    col_in, col_out = st.columns([1, 1])
    
    with col_in:
        st.subheader("Input Text")
        user_input = st.text_area("Your text to analyze:", height=200, placeholder="Type or paste your text here...")
        
        sample_data = load_sample_texts(os.path.join(config.BASE_DIR, "assets", "sample_texts.json"))
        if sample_data:
            selected_sample = st.selectbox("Or choose a sample:", ["Custom..."] + [s['name'] for s in sample_data])
            if selected_sample != "Custom...":
                user_input = next(s['text'] for s in sample_data if s['name'] == selected_sample)
        
        analyze_btn = st.button("🚀 Run Arguasense Pipeline", type="primary", use_container_width=True)

    with col_out:
        if analyze_btn:
            if not user_input.strip():
                st.warning("Please enter text for analysis.")
            elif not predictor.is_loaded:
                st.error("⚠️ Models Not Found!")
                st.markdown("Please run the training scripts in your VS Code terminal first:")
                st.code("python src/train_sarcasm.py\npython src/train_argument.py", language="bash")
            else:
                with st.spinner("Processing NLP Pipeline..."):
                    res = predictor.predict(user_input)
                
                st.subheader("Results")
                r_col1, r_col2 = st.columns(2)
                
                # Sarcasm Metrics
                with r_col1:
                    s_label = res['sarcasm']['label']
                    s_color = "#d32f2f" if res['sarcasm']['id'] == 1 else "#388e3c"
                    result_metric("Sarcasm Detector", s_label, res['sarcasm']['confidence'], s_color)
                
                # Argument Metrics
                with r_col2:
                    a_label = res['argument']['label']
                    a_colors = ["#d32f2f", "#fbc02d", "#388e3c"]
                    a_color = a_colors[res['argument']['id']]
                    result_metric("Argument Quality", a_label, res['argument']['confidence'], a_color)
                
                # Fusion Interpretation
                st.divider()
                f_label = res['fusion']['label']
                f_desc = res['fusion']['explanation']
                f_color = res['fusion']['color']
                
                st.markdown(f"""
                <div style="background-color:{f_color}; padding:30px; border-radius:15px; color:white; text-align:center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h1 style="margin:0; font-size:2.5rem;">{f_label}</h1>
                    <p style="font-size:1.1rem; margin-top:15px; font-weight:300;">{f_desc}</p>
                </div>
                """, unsafe_allow_html=True)

# --- Benchmarks Page ---
elif page == "Model Benchmarks":
    st.title("📉 Performance Benchmarks")
    st.markdown("Comparison of multi-model pipelines and classical ML performance.")
    
    tab1, tab2 = st.tabs(["Sarcasm Detection", "Argument Effectiveness"])
    
    with tab1:
        metrics = get_metrics_summary("sarcasm")
        if metrics is not None:
            st.subheader("Model Comparison (Sorted by F1-Score)")
            st.dataframe(metrics.style.highlight_max(subset=['f1', 'accuracy'], color='lightgreen'), use_container_width=True)
            
            # Show Plot
            best_model = metrics.iloc[0]['model']
            st.success(f"Best Sarcasm Model: **{best_model}**")
            plot_path = os.path.join(config.PLOTS_DIR, f"sarcasm_{best_model}_cm.png")
            if os.path.exists(plot_path):
                st.image(plot_path, caption=f"Confusion Matrix: {best_model} (Sarcasm)")
        else:
            st.warning("No sarcasm metrics found. Train the models first.")

    with tab2:
        metrics = get_metrics_summary("argument")
        if metrics is not None:
            st.subheader("Model Comparison (Sorted by F1-Score)")
            st.dataframe(metrics.style.highlight_max(subset=['f1', 'accuracy'], color='lightgreen'), use_container_width=True)
            
            best_model = metrics.iloc[0]['model']
            st.success(f"Best Argument Model: **{best_model}**")
            plot_path = os.path.join(config.PLOTS_DIR, f"argument_{best_model}_cm.png")
            if os.path.exists(plot_path):
                st.image(plot_path, caption=f"Confusion Matrix: {best_model} (Argument)")
        else:
            st.warning("No argument metrics found. Train the models first.")

# --- Methodology Page ---
elif page == "Methodology":
    st.title("🔬 Methodology & Features")
    
    st.markdown("""
    ### 1. Improved Feature Engineering
    - **Strong TF-IDF**: Unigrams, Bigrams, and Trigrams enabled.
    - **Smart Representation**: Argument labels are improved by combining `Discourse Type` labels (e.g., *Lead, Claim, Evidence*) with the text content.
    - **Sublinear Scaling**: Logarithmic scaling for TF frequencies to reduce the impact of word repetition.
    
    ### 2. Model Suite
    We compare a wide range of classical models to find the optimal speed/accuracy trade-off:
    - **Linear Models**: Logistic Regression, SVM (LinearSVC).
    - **Probabilistic**: Multinomial Naive Bayes.
    - **Ensembles**: Random Forest.
    - **Linear Classifiers with Online Learning**: SGD (Stochastic Gradient Descent), PassiveAggressive.
    
    ### 3. Sarcasm-Aware Fusion
    The fusion layer acts as a heuristic interpreter that weights the argument quality based on the presence of sarcasm, helping identify *Effective Irony* vs. *Poor Sarcastic Effort*.
    """)
    
    st.divider()
    st.markdown("**ArguSense v2.0** | Built for NLP Project Simulation")
