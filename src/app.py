import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
import io
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import run_cleaning_pipeline

# --- SETUP AND PREMIUM STYLING ---
st.set_page_config(page_title="Data Insight Agent", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Enterprise Deep Slate Background */
    .stApp { 
        background-color: #0f172a; 
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Elegant Gradients for Headers */
    h1 {
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        font-size: 2.8rem !important;
        padding-bottom: 0.5rem;
    }
    h2, h3 {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    
    /* Professional Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    
    /* Enterprise Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.5);
    }
    
    /* Premium Dropdowns */
    div[data-baseweb="select"] > div {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        color: #f8fafc;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #94a3b8;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h1>🧬 Data Insight Agent</h1>", unsafe_allow_html=True)
    st.markdown("## Navigation")
    menu = st.radio("", [
        "1. Upload & Auto-Clean", 
        "2. Feature Engineering", 
        "3. AI Cohort Report (K-Means)",
        "4. Drag & Drop BI Dashboard"
    ])
    st.markdown("---")
    st.markdown("**Engineered from scratch:**")
    st.markdown("""
    - **Union-Find Deduplication**
    - **Neural Network Imputation**
    - **Statistical Outlier Engine**
    - **K-Means Clustering**
    """)

# --- STATE MANAGEMENT ---
if 'raw_df' not in st.session_state:
    st.session_state['raw_df'] = None
if 'clean_df' not in st.session_state:
    st.session_state['clean_df'] = None

# --- PAGE ROUTING ---
if menu == "1. Upload & Auto-Clean":
    st.title("Data Pipeline: Upload & Auto-Clean")
    st.markdown("Upload a messy dataset and watch our from-scratch mathematical engines clean it in real-time.")
    
    # 1. FILE UPLOADER
    uploaded_file = st.file_uploader("Upload a CSV file (e.g., messy_data.csv)", type=["csv"])
    
    if uploaded_file is not None:
        # Load into memory
        st.session_state['raw_df'] = pd.read_csv(uploaded_file)
            
    if st.session_state['raw_df'] is not None:
        st.markdown("### 🔍 Raw Data Preview")
        st.dataframe(st.session_state['raw_df'].head(50), use_container_width=True)
        
        # 2. RUN PIPELINE BUTTON
        if st.button("🚀 Run Auto-Clean Pipeline"):
            with st.spinner("Running IQR Outlier Clipping, Mean Imputation, and Deduplication..."):
                cleaned = run_cleaning_pipeline(st.session_state['raw_df'])
                st.session_state['clean_df'] = cleaned
                st.success("Data Cleaning Pipeline Complete! Outliers clipped and Missing Values Imputed.")
                
    if st.session_state['clean_df'] is not None:
        st.markdown("---")
        st.markdown("### ✨ Cleaned Data")
        st.dataframe(st.session_state['clean_df'].head(50), use_container_width=True)
        
        # 3. DOWNLOAD BUTTON
        csv_buffer = io.StringIO()
        st.session_state['clean_df'].to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Cleaned Data CSV",
            data=csv_buffer.getvalue(),
            file_name="cleaned_insight_data.csv",
            mime="text/csv",
        )

elif menu == "2. Feature Engineering":
    st.title("Feature Engineering & Analytics")
    st.markdown("We isolate each feature's mathematical relationship with the target to discover predictive signals.")
    
    if st.session_state['clean_df'] is None:
        st.warning("⚠️ Please go to '1. Upload & Auto-Clean' and run the pipeline first!")
    else:
        df_clean = st.session_state['clean_df'].copy()
        
        # Auto-encode categorical columns dynamically for the math engines!
        for col in df_clean.columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = pd.factorize(df_clean[col])[0]
                
        df = df_clean
        numeric_cols = df.columns.tolist()
        
        if not numeric_cols:
            st.error("No numeric columns found in the dataset for feature engineering.")
        else:
            # Dynamic Target Selection
            target_col = st.selectbox("🎯 Select Target Variable:", numeric_cols, index=0)
            
            st.markdown("### 1. Individual Feature Correlations")
            st.markdown(f"Mathematical relationships between individual features and **{target_col}**.")
            
            with st.spinner("Calculating Pearson Correlation..."):
                from pipeline import calculate_correlation_matrix
                corr_df = calculate_correlation_matrix(df)
                
                if target_col in corr_df.columns:
                    correlations = corr_df[target_col].drop(target_col).sort_values(key=abs, ascending=True)
                    
                    # Bar Chart
                    fig_bar = px.bar(
                        x=correlations.values, y=correlations.index, orientation='h',
                        title=f"Pearson Correlation Strength against {target_col}",
                        color=correlations.values, color_continuous_scale="RdBu",
                        labels={'x': 'Correlation Coefficient (r)', 'y': 'Feature'}
                    )
                    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Scatter Plots Grid
                    st.markdown("#### Feature vs Target Scatter Plots")
                    cols = st.columns(3)
                    for idx, feature in enumerate(correlations.index):
                        col = cols[idx % 3]
                        fig_scatter = px.scatter(
                            df, x=feature, y=target_col, 
                            opacity=0.6, color_discrete_sequence=['#38bdf8']
                        )
                        fig_scatter.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=20, r=20, t=30, b=20),
                            height=250, title=f"{feature}"
                        )
                        col.plotly_chart(fig_scatter, use_container_width=True)
                        
            st.markdown("---")
            st.markdown("### 2. Deep Learning Feature Importance")
            st.markdown(f"We train our from-scratch Neural Network live to predict **{target_col}**.")
            
            with st.spinner("Training Neural Network to extract feature weights..."):
                # Exclude target from features
                feature_cols = [c for c in numeric_cols if c != target_col]
                # Cap at 15 to prevent UI freezing
                feature_cols = feature_cols[:15]
                
                if len(feature_cols) > 0:
                    X = df[feature_cols].values
                    y = df[target_col].values.reshape(-1, 1)
                    
                    # Normalize exactly like we did in Phase 2
                    X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-8)
                    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
                    
                    from nn_from_scratch import NeuralNetwork
                    # Initialize our custom NN
                    nn = NeuralNetwork(input_size=len(feature_cols), hidden_size=8, output_size=1, learning_rate=0.1)
                    
                    for _ in range(1000):
                        y_pred = nn.forward(X_norm)
                        nn.backward(X_norm, y_norm)
                        
                    importance = np.mean(np.abs(nn.W1), axis=1)
                    
                    imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
                    imp_df = imp_df.sort_values(by='Importance', ascending=True)
                    
                    fig2 = px.bar(
                        imp_df, x='Importance', y='Feature', orientation='h',
                        title=f"Neural Network Weight Analysis (Target: {target_col})",
                        color='Importance', color_continuous_scale="viridis"
                    )
                    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig2, use_container_width=True)
                    
            st.markdown("---")
            st.markdown("### 3. AI-Powered Feature Engineering")
            st.markdown("We use **Groq** to analyze the raw correlations and suggest novel Composite Features to improve predictive power.")
            
            if st.button("Generate AI Feature Recommendations"):
                with st.spinner("Connecting to Groq API (Llama 3)..."):
                    groq_key = os.environ.get("GROQ_API_KEY")
                    
                    if not groq_key:
                        st.error("Error: GROQ_API_KEY not found in .env file.")
                    else:
                        try:
                            from openai import OpenAI
                            client = OpenAI(
                                api_key=groq_key,
                                base_url="https://api.groq.com/openai/v1",
                            )
                            
                            prompt = f"""
                            You are a Principal Data Scientist. Here are the Pearson correlation coefficients of various features against the Target Variable ({target_col}):
                            
                            {correlations.to_string()}
                            
                            Suggest exactly 3 'Composite Features' to engineer that will improve the model's predictive power for {target_col}.
                            
                            You MUST output your response as a strict JSON object containing a "recommendations" array, exactly like this:
                            {{
                                "recommendations": [
                                    {{
                                        "feature_name": "Wellness Index",
                                        "raw_features_combined": ["Feature A", "Feature B"],
                                        "mathematical_reasoning": "Both show strong positive correlation...",
                                        "expected_correlation_improvement_pct": 15
                                    }}
                                ]
                            }}
                            """
                            
                            response = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[
                                    {"role": "system", "content": "You are a clinical data scientist. You only output valid JSON."},
                                    {"role": "user", "content": prompt},
                                ],
                                response_format={"type": "json_object"}
                            )
                            
                            import json
                            result_json = json.loads(response.choices[0].message.content)
                            recs = result_json.get("recommendations", [])
                            
                            if recs:
                                st.success("✅ The AI has mathematically designed new composite features!")
                                
                                # Draw beautiful Plotly Bar Chart from the AI's JSON output
                                ai_df = pd.DataFrame(recs)
                                fig_ai = px.bar(
                                    ai_df, x="expected_correlation_improvement_pct", y="feature_name",
                                    orientation="h", title=f"Expected Predictive Improvement for {target_col}",
                                    color="expected_correlation_improvement_pct", color_continuous_scale="viridis",
                                    text="expected_correlation_improvement_pct"
                                )
                                fig_ai.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                                st.plotly_chart(fig_ai, use_container_width=True)
                                
                                # Display the mathematical reasoning cleanly
                                for r in recs:
                                    st.markdown(f"#### {r['feature_name']}")
                                    st.markdown(f"**Combines:** *{', '.join(r['raw_features_combined'])}*")
                                    st.info(r['mathematical_reasoning'])
                                    
                        except Exception as e:
                            st.error(f"Groq API Error: {str(e)}")

elif menu == "3. AI Cohort Report (K-Means)":
    st.title("Generative AI Cohort Report")
    st.markdown("We run our K-Means engine to mathematically cluster rows, then feed the results into **Groq** to generate insights.")
    
    if st.session_state['clean_df'] is None:
        st.warning("⚠️ Please go to '1. Upload & Auto-Clean' and run the pipeline first!")
    else:
        st.info("Using secure backend API Key.")
        
        df_clean = st.session_state['clean_df'].copy()
        for col in df_clean.columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = pd.factorize(df_clean[col])[0]
                
        df = df_clean
        numeric_cols = df.columns.tolist()
        
        # Dynamic multiselect for clustering
        st.markdown("### Mathematical Clustering Setup")
        cluster_cols = st.multiselect("Select Features for K-Means Clustering:", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
        
        if st.button("Run Clustering & Generate Report"):
            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                st.error("Error: GROQ_API_KEY not found in .env file.")
            elif len(cluster_cols) < 2:
                st.error("Please select at least 2 features to cluster.")
            else:
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from kmeans_from_scratch import KMeansFromScratch
                from openai import OpenAI
                
                with st.spinner("Running Mathematical Clustering Engine..."):
                    X = df[cluster_cols].values
                    
                    X_norm = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-8)
                    
                    kmeans = KMeansFromScratch(k=4, n_init=5)
                    labels = kmeans.fit(X_norm)
                    df['Cluster'] = labels
                    
                    st.markdown("### 1. Pattern Discovery (K=4)")
                    
                    # Ensure we have 3 columns for a 3D scatter plot if possible
                    x_col = cluster_cols[0]
                    y_col = cluster_cols[1]
                    z_col = cluster_cols[2] if len(cluster_cols) > 2 else cluster_cols[0]
                    
                    fig = px.scatter_3d(
                        df, x=x_col, y=y_col, z=z_col,
                        color=labels.astype(str),
                        title="Interactive 3D Data Profiles",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, b=0, t=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                st.markdown("### 2. Groq Pattern Analysis")
                with st.spinner("Connecting to Groq API (Llama 3)..."):
                    try:
                        summary = df.groupby('Cluster')[cluster_cols].mean().round(2).to_string()
                        
                        client = OpenAI(
                            api_key=groq_key,
                            base_url="https://api.groq.com/openai/v1",
                        )
                        
                        prompt = f"""
                        You are an expert Data Scientist. I have clustered a dataset into 4 distinct groups using K-Means.
                        Here are the average metrics for each cluster:
                        
                        {summary}
                        
                        Write a brief, professional report analyzing these 4 cohorts. 
                        - What uniquely defines each cohort based on the numbers? 
                        - What high-level recommendations would you make for each group?
                        Format in beautiful Markdown using bullet points and headers. Be concise.
                        """
                        
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": "You are an expert data scientist."},
                                {"role": "user", "content": prompt},
                            ],
                        )
                        
                        report = response.choices[0].message.content
                        st.markdown(report)
                        
                    except Exception as e:
                        st.error(f"Failed to connect to Groq API: {str(e)}")

elif menu == "4. Drag & Drop BI Dashboard":
    st.title("Business Intelligence Studio")
    st.markdown("We have embedded an enterprise-grade Tableau/Power BI drag-and-drop experience directly into the app natively.")
    
    if st.session_state['clean_df'] is None:
        st.warning("⚠️ Please go to '1. Upload & Auto-Clean' and run the pipeline first!")
    else:
        import pygwalker as pyg
        import streamlit.components.v1 as components
        
        with st.spinner("Rendering Business Intelligence Studio..."):
            # Generate pure HTML payload instead of hacking Streamlit's server
            html = pyg.to_html(st.session_state['clean_df'], dark="dark")
            
            # Embed the HTML directly into Streamlit
            components.html(html, height=900, scrolling=True)
