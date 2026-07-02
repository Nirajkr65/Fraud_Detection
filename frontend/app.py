import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
from eda import render_eda

# Load API URL from Environment Variable (Fallback to localhost for dev)
API_URL = os.getenv("API_URL", "https://fraud-detection-1-tutc.onrender.com")

st.set_page_config(page_title="Fraud Guard - AI Fraud Detection", layout="wide")

# Custom styling for a premium look
st.markdown("""
<style>
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
        border-left: 5px solid #3498db;
    }
    .metric-fraud {
        border-left: 5px solid #e74c3c;
    }
    .metric-normal {
        border-left: 5px solid #2ecc71;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None

# Helper functions for API calls
def get_auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}

def handle_api_error(e):
    if isinstance(e, requests.exceptions.ConnectionError):
        st.error(f"🔌 **Connection Error**: Unable to connect to the backend server at `{API_URL}`. "
                 f"Please ensure the FastAPI service is running. Run: `uvicorn backend.app.main:app --reload --port 8000`")
    elif isinstance(e, requests.exceptions.Timeout):
        st.error("⏳ **Timeout Error**: The request timed out. Please try again.")
    else:
        st.error(f"❌ **API Error**: An unexpected error occurred: {e}")

def make_login(username, password):
    with st.spinner("Authenticating credentials..."):
        try:
            res = requests.post(f"{API_URL}/token", data={"username": username, "password": password}, timeout=10)
            if res.status_code == 200:
                data = res.json()
                st.session_state.token = data["access_token"]
                st.session_state.username = username
                st.rerun()
            elif res.status_code == 401:
                st.error("🔑 **Authentication Failed**: Incorrect username or password.")
            else:
                st.error(f"⚠️ Error {res.status_code}: {res.json().get('detail', 'Unknown error')}")
        except Exception as e:
            handle_api_error(e)

def make_signup(username, password):
    with st.spinner("Creating user account..."):
        try:
            res = requests.post(f"{API_URL}/signup", json={"username": username, "password": password}, timeout=10)
            if res.status_code == 201:
                st.success("📝 **Registration Successful!** You can now login using the credentials above.")
            else:
                st.error(f"⚠️ Registration failed: {res.json().get('detail', 'Unknown error')}")
        except Exception as e:
            handle_api_error(e)

# --- AUTH PORTAL ---
if not st.session_state.token:
    st.title("🛡️ Fraud Guard - Security Portal")
    st.markdown("Please log in or register to access the AI Fraud Detection System.")
    
    auth_tab1, auth_tab2 = st.tabs(["🔑 Login", "📝 Register"])
    
    with auth_tab1:
        with st.form("login_form"):
            user_in = st.text_input("Username")
            pass_in = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            if submitted:
                if user_in and pass_in:
                    make_login(user_in, pass_in)
                else:
                    st.warning("Please fill out all fields.")
                    
    with auth_tab2:
        with st.form("register_form"):
            reg_user = st.text_input("Choose Username")
            reg_pass = st.text_input("Choose Password", type="password")
            submitted_reg = st.form_submit_button("Sign Up", use_container_width=True)
            if submitted_reg:
                if reg_user and reg_pass:
                    make_signup(reg_user, reg_pass)
                else:
                    st.warning("Please fill out all fields.")

else:
    # --- LOGGED IN DASHBOARD ---
    
    # Sidebar
    st.sidebar.title("🛡️ Fraud Guard")
    st.sidebar.caption(f"Logged in as: **{st.session_state.username}**")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Navigation", [
        "🏠 Dashboard",
        "🔍 Single Predict",
        "📁 Upload CSV",
        "📊 Analytics",
        "📈 Model Comparison",
        "📜 Prediction History",
        "⚙️ Settings"
    ])
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.token = None
        st.session_state.username = None
        st.rerun()

    # --- 1. DASHBOARD PAGE ---
    if page == "🏠 Dashboard":
        st.title("🏠 Security Overview Dashboard")
        st.markdown("High-level overview of transaction monitoring logs.")
        
        # Load user prediction history
        history = []
        with st.spinner("Fetching dashboard statistics..."):
            try:
                res = requests.get(f"{API_URL}/transactions", headers=get_auth_headers(), timeout=10)
                if res.status_code == 200:
                    history = res.json()
                elif res.status_code == 401:
                    st.session_state.token = None
                    st.session_state.username = None
                    st.rerun()
                else:
                    st.error("Failed to load transactions history.")
            except Exception as e:
                handle_api_error(e)
            
        if history:
            df = pd.DataFrame(history)
            
            # Key statistics
            total_tx = len(df)
            fraud_tx = df["is_fraud"].sum()
            clear_tx = total_tx - fraud_tx
            fraud_pct = (fraud_tx / total_tx) * 100 if total_tx > 0 else 0.0
            
            total_fraud_amt = df[df["is_fraud"] == True]["amount"].sum() if fraud_tx > 0 else 0.0
            avg_fraud_amt = df[df["is_fraud"] == True]["amount"].mean() if fraud_tx > 0 else 0.0
            avg_normal_amt = df[df["is_fraud"] == False]["amount"].mean() if clear_tx > 0 else 0.0

            # Row 1: KPI Cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-box"><h4>Total Monitored</h4><h2>{total_tx:,}</h2></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-box metric-normal"><h4>Cleared Cases</h4><h2>{clear_tx:,}</h2></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-box metric-fraud"><h4>Fraud Flags</h4><h2>{fraud_tx:,}</h2></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-box metric-fraud"><h4>Fraud Rate</h4><h2>{fraud_pct:.2f}%</h2></div>', unsafe_allow_html=True)
            
            st.markdown(" ")
            
            # Row 2: Secondary Stats
            s1, s2, s3 = st.columns(3)
            s1.metric("High-Risk Exposure Amount", f"${total_fraud_amt:,.2f}")
            s2.metric("Avg Fraudulent Ticket", f"${avg_fraud_amt:,.2f}")
            s3.metric("Avg Normal Ticket", f"${avg_normal_amt:,.2f}")

            st.markdown("---")

            # Row 3: Interactive Charts
            st.subheader("📊 Fraud Statistics & Trends")
            col1, col2 = st.columns(2)
            with col1:
                cat_df = df.groupby(["merchant_category", "is_fraud"]).size().reset_index(name="count")
                cat_df["Status"] = cat_df["is_fraud"].map({True: "Fraud", False: "Normal"})
                fig_cat = px.bar(cat_df, x="merchant_category", y="count", color="Status", 
                                 color_discrete_map={"Normal": "#2ecc71", "Fraud": "#e74c3c"},
                                 barmode="group",
                                 title="Flagged Transactions by Merchant Category")
                st.plotly_chart(fig_cat, use_container_width=True)
                
            with col2:
                hr_df = df.groupby(["transaction_hour", "is_fraud"]).size().reset_index(name="count")
                hr_df["Status"] = hr_df["is_fraud"].map({True: "Fraud", False: "Normal"})
                fig_hr = px.line(hr_df, x="transaction_hour", y="count", color="Status",
                                 color_discrete_map={"Normal": "#2ecc71", "Fraud": "#e74c3c"},
                                 markers=True,
                                 title="Hourly Transaction Flow & Fraud Density")
                st.plotly_chart(fig_hr, use_container_width=True)

            st.markdown("---")

            col3, col4 = st.columns(2)
            with col3:
                fig_scatter = px.scatter(df, x="cardholder_age", y="amount", color="is_fraud",
                                         color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
                                         title="Transaction Values vs. Cardholder Age",
                                         labels={"cardholder_age": "Cardholder Age", "amount": "Amount ($)", "is_fraud": "Is Fraud"})
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            with col4:
                st.subheader("📜 Recent Predictions")
                df_sorted = df.sort_values(by="timestamp", ascending=False).head(5)
                show_cols = ["transaction_id", "amount", "merchant_category", "prediction_score", "is_fraud"]
                df_show = df_sorted[show_cols].copy()
                df_show.columns = ["Tx ID", "Amount ($)", "Category", "Fraud Prob", "Flagged"]
                df_show["Amount ($)"] = df_show["Amount ($)"].map(lambda x: f"${x:.2f}")
                df_show["Fraud Prob"] = df_show["Fraud Prob"].map(lambda x: f"{x*100:.1f}%")
                df_show["Flagged"] = df_show["Flagged"].map(lambda x: "🔴 Yes" if x else "🟢 No")
                st.dataframe(df_show, use_container_width=True)
        else:
            st.info("No transaction logs available. Go to **Single Predict** or **Upload CSV** to get started.")

    # --- 2. SINGLE PREDICT PAGE ---
    elif page == "🔍 Single Predict":
        st.title("🔍 Single Transaction Analyzer")
        st.markdown("Analyze transaction attributes using the active machine learning classifier.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.form("single_predict_form"):
                tx_id = st.text_input("Transaction ID", f"TX_{int(datetime.utcnow().timestamp())}")
                amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=125.50, step=0.01)
                tx_hour = st.slider("Transaction Hour (0-23)", 0, 23, 12)
                merchant_category = st.selectbox("Merchant Category", ["Electronics", "Travel", "Grocery", "Food", "Clothing"])
                
                c_a, c_b = st.columns(2)
                with c_a:
                    foreign_tx = st.selectbox("Foreign Transaction?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
                    location_mismatch = st.selectbox("Location Mismatch?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
                with c_b:
                    device_trust = st.slider("Device Trust Score (0-100)", 0, 100, 75)
                    velocity_24h = st.number_input("Transactions in Last 24h", min_value=0, value=2, step=1)
                
                age = st.slider("Cardholder Age", 18, 90, 42)
                
                submitted = st.form_submit_button("Run Analysis", use_container_width=True)
                
                if submitted:
                    payload = {
                        "transaction_id": tx_id,
                        "amount": amount,
                        "transaction_hour": tx_hour,
                        "merchant_category": merchant_category,
                        "foreign_transaction": foreign_tx,
                        "location_mismatch": location_mismatch,
                        "device_trust_score": device_trust,
                        "velocity_last_24h": velocity_24h,
                        "cardholder_age": age
                    }
                    
                    with st.spinner("Submitting transaction attributes to model..."):
                        try:
                            res = requests.post(f"{API_URL}/predict", json=payload, headers=get_auth_headers(), timeout=10)
                            if res.status_code == 200:
                                data = res.json()
                                st.session_state.last_prediction = data
                            else:
                                st.error(f"⚠️ Error {res.status_code}: {res.json().get('detail', 'Failed prediction')}")
                        except Exception as e:
                            handle_api_error(e)
                        
        with col2:
            st.subheader("Analysis Results")
            if "last_prediction" in st.session_state:
                res_data = st.session_state.last_prediction
                is_f = res_data["is_fraud"]
                prob = res_data["prediction_score"]
                
                confidence = prob if is_f else (1.0 - prob)
                
                if is_f:
                    st.error("🚨 **RESULT: FRAUDULENT**")
                else:
                    st.success("🟢 **RESULT: NORMAL**")
                
                st.markdown(f"**Fraud Probability**: `{prob * 100:.2f}%`")
                if prob > 0.8:
                    st.progress(prob)
                    st.caption("🔴 High Risk Profile Detected")
                elif prob >= 0.5:
                    st.progress(prob)
                    st.caption("🟡 Medium Risk Profile Detected")
                else:
                    st.progress(prob)
                    st.caption("🟢 Low Risk Profile Detected")

                st.metric("Model Confidence Score", f"{confidence * 100:.2f}%")
                
                st.markdown("---")
                st.markdown("**Transaction Details**:")
                st.write({
                    "Tx ID": res_data["transaction_id"],
                    "Amount": f"${res_data['amount']:.2f}",
                    "Category": res_data["merchant_category"],
                    "Device Trust": res_data["device_trust_score"],
                    "Velocity": res_data["velocity_last_24h"]
                })
            else:
                st.info("Submit the transaction form to see predictions.")

    # --- 3. UPLOAD CSV PAGE ---
    elif page == "📁 Upload CSV":
        st.title("📁 Batch Predictions Upload")
        st.markdown("Upload a CSV file containing transactions to evaluate them in batch.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            if st.button("Evaluate Batch CSV", use_container_width=True):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                with st.spinner("Processing batch predictions (this may take a few seconds)..."):
                    try:
                        res = requests.post(f"{API_URL}/predict/batch", files=files, headers=get_auth_headers(), timeout=30)
                        if res.status_code == 200:
                            batch_res = res.json()
                            st.success("Batch evaluation complete!")
                            
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Processed Records", batch_res["total_processed"])
                            c2.metric("Fraud Flagged", batch_res["fraud_detected"])
                            c3.metric("Normal Transactions", batch_res["normal_detected"])
                            c4.metric("Fraud Ratio", f"{batch_res['fraud_ratio']*100:.2f}%")
                            
                            df_res = pd.DataFrame(batch_res["predictions"])
                            st.dataframe(df_res, use_container_width=True)
                            
                            csv_data = df_res.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Predicted Transactions CSV",
                                data=csv_data,
                                file_name=f"predicted_{uploaded_file.name}",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            st.error(f"⚠️ Error {res.status_code}: {res.json().get('detail', 'Batch prediction failed')}")
                    except Exception as e:
                        handle_api_error(e)

    # --- 4. ANALYTICS (EDA) PAGE ---
    elif page == "📊 Analytics":
        render_eda()

    # --- 5. MODEL COMPARISON PAGE ---
    elif page == "📈 Model Comparison":
        st.title("📈 Model Evaluation & Comparison")
        st.markdown("Compare the performance of the Baseline classifier vs. the model retrained with SMOTE oversampling.")
        
        metrics = None
        with st.spinner("Retrieving evaluation scores..."):
            try:
                res = requests.get(f"{API_URL}/metrics", timeout=10)
                if res.status_code == 200:
                    metrics = res.json()
                else:
                    st.error("Failed to load metrics from API.")
            except Exception as e:
                handle_api_error(e)
                
        if metrics:
            base = metrics["baseline_model"]
            smote = metrics["smote_model"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Baseline RF Classifier")
                st.write({
                    "Precision": base["precision"],
                    "Recall": base["recall"],
                    "F1-Score": base["f1_score"],
                    "ROC-AUC": base["roc_auc"]
                })
                cm_base = base["confusion_matrix"]
                st.markdown("**Confusion Matrix**:")
                st.write(pd.DataFrame([
                    [cm_base["tn"], cm_base["fp"]],
                    [cm_base["fn"], cm_base["tp"]]
                ], columns=["Predicted Normal", "Predicted Fraud"], index=["Actual Normal", "Actual Fraud"]))
                
            with col2:
                st.subheader("SMOTE Retrained Classifier")
                st.write({
                    "Precision": smote["precision"],
                    "Recall": smote["recall"],
                    "F1-Score": smote["f1_score"],
                    "ROC-AUC": smote["roc_auc"]
                })
                cm_smote = smote["confusion_matrix"]
                st.markdown("**Confusion Matrix**:")
                st.write(pd.DataFrame([
                    [cm_smote["tn"], cm_smote["fp"]],
                    [cm_smote["fn"], cm_smote["tp"]]
                ], columns=["Predicted Normal", "Predicted Fraud"], index=["Actual Normal", "Actual Fraud"]))
                
            chart_data = pd.DataFrame([
                {"Model": "Baseline RF", "Metric": "Precision", "Value": base["precision"]},
                {"Model": "Baseline RF", "Metric": "Recall", "Value": base["recall"]},
                {"Model": "Baseline RF", "Metric": "F1-Score", "Value": base["f1_score"]},
                {"Model": "Baseline RF", "Metric": "ROC-AUC", "Value": base["roc_auc"]},
                {"Model": "SMOTE RF", "Metric": "Precision", "Value": smote["precision"]},
                {"Model": "SMOTE RF", "Metric": "Recall", "Value": smote["recall"]},
                {"Model": "SMOTE RF", "Metric": "F1-Score", "Value": smote["f1_score"]},
                {"Model": "SMOTE RF", "Metric": "ROC-AUC", "Value": smote["roc_auc"]},
            ])
            
            fig_comp = px.bar(chart_data, x="Metric", y="Value", color="Model", barmode="group",
                               color_discrete_map={"Baseline RF": "#95a5a6", "SMOTE RF": "#3498db"},
                               title="Model Metrics Comparison Chart")
            st.plotly_chart(fig_comp, use_container_width=True)

    # --- 6. HISTORY PAGE ---
    elif page == "📜 Prediction History":
        st.title("📜 Prediction Log History")
        st.markdown("View history logs of individual and batch transactions analyzed under your account.")
        
        history = []
        with st.spinner("Fetching transaction logs..."):
            try:
                res = requests.get(f"{API_URL}/transactions", headers=get_auth_headers(), timeout=10)
                if res.status_code == 200:
                    history = res.json()
                else:
                    st.error("Failed to load prediction history.")
            except Exception as e:
                handle_api_error(e)
                
        if history:
            df = pd.DataFrame(history)
            
            search_id = st.text_input("🔍 Search by Transaction ID")
            filter_fraud = st.selectbox("Filter by Status", ["All", "Only Fraud Flags", "Only Cleared Transactions"])
            
            if search_id:
                df = df[df["transaction_id"].str.contains(search_id, case=False)]
                
            if filter_fraud == "Only Fraud Flags":
                df = df[df["is_fraud"] == True]
            elif filter_fraud == "Only Cleared Transactions":
                df = df[df["is_fraud"] == False]
                
            st.dataframe(df, use_container_width=True)
            
            csv_hist = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Log",
                data=csv_hist,
                file_name="transaction_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No transaction prediction records logged yet.")

    # --- 7. SETTINGS PAGE ---
    elif page == "⚙️ Settings":
        st.title("⚙️ System Registry & Settings")
        st.markdown("Inspect active model registry version metadata and server configuration.")
        
        st.subheader("User Session Info")
        st.write({
            "Authorized User": st.session_state.username,
            "API URL": API_URL
        })
        
        st.markdown("---")
        
        # Model Registry info
        st.subheader("Model Registry Directory")
        models_info = []
        with st.spinner("Querying registry..."):
            try:
                res = requests.get(f"{API_URL}/models", headers=get_auth_headers(), timeout=10)
                if res.status_code == 200:
                    models_info = res.json()
                else:
                    st.error("Could not fetch model registry information.")
            except Exception as e:
                handle_api_error(e)
                
        if models_info:
            st.dataframe(pd.DataFrame(models_info), use_container_width=True)
        else:
            st.info("No models registered.")
            
        st.markdown("---")
        
        # Uploaded files history
        st.subheader("Uploaded Files History Log")
        files_info = []
        with st.spinner("Fetching uploads logs..."):
            try:
                res = requests.get(f"{API_URL}/uploaded-files", headers=get_auth_headers(), timeout=10)
                if res.status_code == 200:
                    files_info = res.json()
                else:
                    st.error("Could not fetch uploaded files information.")
            except Exception as e:
                handle_api_error(e)
                
        if files_info:
            st.dataframe(pd.DataFrame(files_info), use_container_width=True)
        else:
            st.info("No file uploads logged.")
