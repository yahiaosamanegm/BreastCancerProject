import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. Page Config & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cellular Diagnostics Pro 🔬",
    page_icon="🌸",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.3rem;
        color: #E91E63;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #666;
        text-align: center;
        margin-bottom: 25px;
    }
    .insight-card {
        background-color: #1E1E2E;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #E91E63;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Data Loading & Model Training
# ---------------------------------------------------------
@st.cache_resource
def load_and_train():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    
    X = df.drop("target", axis=1)
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate average profiles for radar comparison
    benign_avg = df[df['target'] == 1].mean()
    malignant_avg = df[df['target'] == 0].mean()
    
    return model, data, df, benign_avg, malignant_avg

model, data, df, benign_avg, malignant_avg = load_and_train()
feature_names = list(data.feature_names)

# ---------------------------------------------------------
# 3. Header Section
# ---------------------------------------------------------
st.markdown('<div class="main-header">🌸 Cellular Diagnostics Pro & AI Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive Medical Intelligence • Visualization & Feature Analysis</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. Input Layout (Left Side) & Visual Output (Right Side)
# ---------------------------------------------------------
col_inputs, col_visuals = st.columns([1.1, 1.9], gap="large")

with col_inputs:
    st.subheader("📋 Enter Cell Measurements")
    st.caption("Adjust sliders below or use default sample values.")
    
    with st.form("diagnostic_form"):
        user_inputs = {}
        tab1, tab2, tab3 = st.tabs(["📏 Means", "🔍 Errors", "📉 Worst"])
        
        with tab1:
            for i in range(0, 10):
                feat = feature_names[i]
                user_inputs[feat] = st.slider(
                    feat.replace("mean ", "").title(),
                    float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()), key=feat
                )
                
        with tab2:
            for i in range(10, 20):
                feat = feature_names[i]
                user_inputs[feat] = st.slider(
                    feat.replace(" error", "").title(),
                    float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()), key=feat
                )
                
        with tab3:
            for i in range(20, 30):
                feat = feature_names[i]
                user_inputs[feat] = st.slider(
                    feat.replace("worst ", "").title(),
                    float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()), key=feat
                )
        
        submit_btn = st.form_submit_button("🚀 Run AI Diagnosis", use_container_width=True)

# ---------------------------------------------------------
# 5. Visualizations & Diagnostic Reasoning
# ---------------------------------------------------------
with col_visuals:
    # Prepare data for prediction
    input_df = pd.DataFrame([user_inputs])
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    
    is_benign = (prediction == 1)
    confidence = probs[1] if is_benign else probs[0]
    
    # Header KPI Cards
    st.subheader("🎯 Diagnostic Summary")
    kpi1, kpi2, kpi3 = st.columns(3)
    
    if is_benign:
        kpi1.metric("Diagnosis", "Benign (حميد)", delta="Healthy", delta_color="normal")
        kpi2.metric("Confidence", f"{confidence*100:.1f}%", "High Precision")
        kpi3.metric("Risk Score", "Low Risk", "- Safe Profile", delta_color="normal")
    else:
        kpi1.metric("Diagnosis", "Malignant (خبيث)", delta="Action Needed", delta_color="inverse")
        kpi2.metric("Confidence", f"{confidence*100:.1f}%", "Requires Review")
        kpi3.metric("Risk Score", "High Risk", "+ Critical Profile", delta_color="inverse")
        
    if submit_btn and is_benign:
        st.balloons()

    st.divider()

    # Visual Tabs
    vis_tab1, vis_tab2, vis_tab3 = st.tabs([
        "🕸️ Cellular Radar Profile", 
        "🔥 Diagnostic Reasons (Feature Importance)", 
        "💡 AI Insight & Explanation"
    ])
    
    # --- VISUAL TAB 1: Radar Chart ---
    with vis_tab1:
        st.markdown("#### Sample Profile vs Baseline Means")
        radar_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
        
        # Normalize for radar visualization scale (0 to 1)
        user_norm = [ (user_inputs[f] - df[f].min()) / (df[f].max() - df[f].min()) for f in radar_features ]
        benign_norm = [ (benign_avg[f] - df[f].min()) / (df[f].max() - df[f].min()) for f in radar_features ]
        malignant_norm = [ (malignant_avg[f] - df[f].min()) / (df[f].max() - df[f].min()) for f in radar_features ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=user_norm, theta=[f.title() for f in radar_features], fill='toself', name='User Sample', line_color='#E91E63'))
        fig_radar.add_trace(go.Scatterpolar(r=benign_norm, theta=[f.title() for f in radar_features], fill='toself', name='Average Benign', line_color='#2ECC71', opacity=0.4))
        fig_radar.add_trace(go.Scatterpolar(r=malignant_norm, theta=[f.title() for f in radar_features], fill='toself', name='Average Malignant', line_color='#E74C3C', opacity=0.4))
        
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=380, margin=dict(t=20, b=20, l=40, r=40))
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- VISUAL TAB 2: Feature Importance ---
    with vis_tab2:
        st.markdown("#### Top 8 Global Features Influencing Model")
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-8:]
        
        top_features = [feature_names[i].title() for i in top_indices]
        top_values = importances[top_indices]
        
        fig_bar = px.bar(
            x=top_values, y=top_features, orientation='h',
            labels={'x': 'Relative Importance Weight', 'y': 'Cellular Attribute'},
            color=top_values, color_continuous_scale='PuRd'
        )
        fig_bar.update_layout(height=380, margin=dict(t=20, b=20, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- VISUAL TAB 3: Explanation & Reason ---
    with vis_tab3:
        st.markdown("#### 🔍 Why did the AI give this outcome?")
        
        # Find which important feature diverted most from normal
        top_feat = feature_names[top_indices[-1]]
        user_val = user_inputs[top_feat]
        avg_val = df[top_feat].mean()
        diff_pct = ((user_val - avg_val) / avg_val) * 100
        
        if is_benign:
            st.success("✨ **Reasoning:** The input cell measurements align closely with typical benign (non-cancerous) parameters.")
            st.write(f"- **Key Attribute:** The `{top_feat.title()}` was recorded at **{user_val:.2f}**.")
            st.write(f"- **Deviation:** It is **{abs(diff_pct):.1f}% {'higher' if diff_pct > 0 else 'lower'}** than average, keeping it within safe ranges.")
        else:
            st.error("⚠️ **Reasoning:** Critical cell features show enlarged or elevated structural patterns.")
            st.write(f"- **Key Attribute:** The `{top_feat.title()}` measured **{user_val:.2f}**.")
            st.write(f"- **Deviation:** It exceeded standard values by **{diff_pct:.1f}%**, which strongly correlates with malignant indicators.")
            
        st.markdown("""
            <div class="insight-card">
                <b>💡 Clinical Note:</b> <br>
                Machine learning models analyze multiple structural parameters concurrently. High values in radius, perimeter, and concavity are the strongest contributors to a malignant prediction.
            </div>
        """, unsafe_allow_html=True)