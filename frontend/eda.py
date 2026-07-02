import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

DATA_PATH = "/Users/NIRAJKUMAR/Desktop/Fraud_Detection/fraud_dataset.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def render_eda():
    st.header("📊 Exploratory Data Analysis Dashboard")
    st.markdown("Dive deeper into the properties, distributions, and correlation patterns of the transaction dataset.")

    try:
        df = load_data()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return

    # Metric summary row
    st.markdown("### Key Dataset Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{len(df):,}")
    m2.metric("Total Fraud Cases", f"{df['is_fraud'].sum():,}")
    fraud_pct = (df['is_fraud'].sum() / len(df)) * 100
    m3.metric("Fraud Rate", f"{fraud_pct:.2f}%")
    m4.metric("Average Transaction", f"${df['amount'].mean():.2f}")

    st.markdown("---")

    # Layout for charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution (Imbalance)")
        fraud_counts = df['is_fraud'].value_counts().reset_index()
        fraud_counts.columns = ['Status', 'Count']
        fraud_counts['Status'] = fraud_counts['Status'].map({0: 'Normal', 1: 'Fraud'})
        
        fig_pie = px.pie(
            fraud_counts, 
            values='Count', 
            names='Status', 
            color='Status',
            color_discrete_map={'Normal': '#2ECC71', 'Fraud': '#E74C3C'},
            hole=0.4,
            title="Normal vs Fraud Transaction Distribution"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Fraud by Merchant Category")
        category_fraud = df.groupby(['merchant_category', 'is_fraud']).size().reset_index(name='count')
        category_fraud['Status'] = category_fraud['is_fraud'].map({0: 'Normal', 1: 'Fraud'})
        
        fig_bar = px.bar(
            category_fraud,
            x='merchant_category',
            y='count',
            color='Status',
            color_discrete_map={'Normal': '#95A5A6', 'Fraud': '#E74C3C'},
            barmode='stack',
            title="Transaction Volume by Category (Stacked)",
            labels={'merchant_category': 'Category', 'count': 'Transaction Count'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Transaction Amounts Distribution")
        fig_hist = px.histogram(
            df,
            x='amount',
            color='is_fraud',
            color_discrete_map={0: '#34495E', 1: '#E74C3C'},
            marginal='box',
            nbins=50,
            title="Distribution of Amount ($) by Fraud Status",
            labels={'amount': 'Amount ($)', 'is_fraud': 'Is Fraud'},
            log_y=True
        )
        fig_hist.update_layout(
            legend_title_text='Is Fraud',
            xaxis_title='Transaction Amount ($)',
            yaxis_title='Log Frequency'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col4:
        st.subheader("Feature Distributions Stratified by Fraud")
        feature_to_plot = st.selectbox(
            "Select feature to plot distribution:",
            ["device_trust_score", "cardholder_age", "velocity_last_24h", "transaction_hour"]
        )
        
        fig_violin = px.violin(
            df,
            y=feature_to_plot,
            x='is_fraud',
            color='is_fraud',
            color_discrete_map={0: '#3498DB', 1: '#E74C3C'},
            box=True,
            points="all",
            title=f"{feature_to_plot.replace('_', ' ').title()} by Fraud Label",
            labels={'is_fraud': 'Is Fraud (0 = No, 1 = Yes)'}
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    st.markdown("---")
    
    st.subheader("Feature Correlation Heatmap")
    numerical_cols = df.drop(columns=['transaction_id', 'merchant_category']).corr()
    
    z = np.round(numerical_cols.values, 2)
    x = list(numerical_cols.columns)
    y = list(numerical_cols.index)
    
    fig_corr = ff.create_annotated_heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        showscale=True
    )
    fig_corr.update_layout(title_text="Correlation Matrix Heatmap", margin=dict(t=50, b=50))
    st.plotly_chart(fig_corr, use_container_width=True)
