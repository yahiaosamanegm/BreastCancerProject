import streamlit as st
from backend import CancerPredictionModel

# Page Config
st.set_page_config(
    page_title="Cellular Health AI Lab 🔬",
    page_icon="🌸",
    layout="wide"
)

# Custom Styling & Colors
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #E91E63;
        font-weight: 700;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555555;
    }
    </style>
""", unsafe_allow_html=True)

# Cache model instance for speed
@st.cache_resource
def get_model():
    return CancerPredictionModel()

model = get_model()
stats = model.get_feature_stats()

# Header Section
st.markdown('<div class="main-title">🌸 Cellular Health Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by AI & Scikit-Learn • Instant Breast Cancer Cell Diagnostics</div>', unsafe_allow_html=True)
st.write("---")

# Main Content Layout
col_input, col_display = st.columns([2, 1])

with col_input:
    st.subheader("📋 Input Cell Measurement Features")
    st.info("Adjust the sliders to match sample measurements, then click **Run Diagnostic Test**.")
    
    with st.form("prediction_form"):
        user_inputs = {}
        
        # Tabbed Organization for Features
        tab1, tab2, tab3 = st.tabs(["📏 Mean Values", "🔍 Error Metrics", "📉 Worst Metrics"])
        
        feature_list = list(model.feature_names)
        
        with tab1:
            st.caption("Average dimensions of the cell nuclei")
            for feat in feature_list[:10]:
                user_inputs[feat] = st.slider(
                    feat.replace("mean ", "").title(),
                    min_value=stats[feat]["min"],
                    max_value=stats[feat]["max"],
                    value=stats[feat]["mean"],
                    key=feat
                )
                
        with tab2:
            st.caption("Standard error of measurement values")
            for feat in feature_list[10:20]:
                user_inputs[feat] = st.slider(
                    feat.replace("error", "").title(),
                    min_value=stats[feat]["min"],
                    max_value=stats[feat]["max"],
                    value=stats[feat]["mean"],
                    key=feat
                )
                
        with tab3:
            st.caption("Largest or worst dimensions observed")
            for feat in feature_list[20:]:
                user_inputs[feat] = st.slider(
                    feat.replace("worst ", "").title(),
                    min_value=stats[feat]["min"],
                    max_value=stats[feat]["max"],
                    value=stats[feat]["mean"],
                    key=feat
                )
        
        st.write("")
        submit_btn = st.form_submit_button("🚀 Run Diagnostic Test", use_container_width=True)

with col_display:
    st.subheader("🎯 Result Dashboard")
    
    if submit_btn:
        result = model.predict(user_inputs)
        
        st.markdown("### Diagnosis Summary")
        if result["is_benign"]:
            st.balloons()  # Celebration animation! ✨
            st.success("🟢 **BENIGN CELL PATTERN**")
            st.metric(label="Assessment Result", value="No Cancer Detected")
            st.metric(label="Model Confidence", value=f"{result['confidence']}%")
            st.markdown("Everything looks standard based on provided measurements!")
        else:
            st.error("🔴 **MALIGNANT CELL PATTERN**")
            st.metric(label="Assessment Result", value="Action Advised")
            st.metric(label="Model Confidence", value=f"{result['confidence']}%")
            st.warning("Sample indicators fall into malignant ranges. Further clinical evaluation is recommended.")
    else:
        st.info("💡 Adjust values on the left panel and hit **Run Diagnostic Test** to generate an AI evaluation.")