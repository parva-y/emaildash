import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Email Campaign Projections", layout="wide")

st.title("📩 Email Campaigns – Historical Data & Projections")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload your monthly performance CSV (with columns: Month, Emails Sent, Visits, Orders, CR%, Revenue, AOV, Cost, CPO)", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Clean & format
    df.columns = df.columns.str.strip()
    df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
    df = df.sort_values('Month')

    st.subheader("📊 Historical Data Preview")
    st.dataframe(df.tail(12))

    # --- YoY Growth Calculation ---
    df['Year'] = df['Month'].dt.year
    df['Month_Num'] = df['Month'].dt.month

    # Pivot for YoY comparison
    pivot = df.pivot(index='Month_Num', columns='Year', values='Emails Sent')
    st.write("### Year-on-Year Sends Trend")
    st.line_chart(pivot)

    # --- Projection Logic ---
    st.subheader("📈 Projection Settings")
    growth_type = st.radio("Projection method:", ["Simple Avg Growth", "CAGR", "Last Year Copy"])

    years_available = df['Year'].unique()
    base_year = st.selectbox("Base Year for projection", sorted(years_available, reverse=True))

    future_year = base_year + 1
    st.markdown(f"**Projecting for {future_year}**")

    proj_df = df[df['Year'] == base_year].copy()
    proj_df['Year'] = future_year

    if growth_type == "Simple Avg Growth":
        avg_growth = df.groupby('Month_Num')['Emails Sent'].pct_change().mean()
        proj_df['Emails Sent'] = proj_df['Emails Sent'] * (1 + avg_growth)
    elif growth_type == "CAGR":
        first = df.groupby('Month_Num').first()['Emails Sent']
        last = df.groupby('Month_Num').last()['Emails Sent']
        n = len(years_available) - 1
        cagr = (last / first) ** (1/n) - 1
        proj_df['Emails Sent'] = proj_df['Month_Num'].map(lambda m: proj_df.loc[proj_df['Month_Num']==m, 'Emails Sent'] * (1 + cagr[m]))
    else:
        # Just copy last year
        proj_df['Emails Sent'] = proj_df['Emails Sent']

    # Project dependent metrics using simple ratios
    order_rate = (df['Orders'].sum() / df['Emails Sent'].sum()) if 'Orders' in df else 0.002
    revenue_per_order = (df['Revenue'].sum() / df['Orders'].sum()) if 'Revenue' in df else 1500

    proj_df['Orders'] = (proj_df['Emails Sent'] * order_rate).astype(int)
    proj_df['Revenue'] = (proj_df['Orders'] * revenue_per_order).round()
    proj_df['CR%'] = (proj_df['Orders'] / proj_df['Emails Sent'] * 100).round(2)

    st.subheader("🔮 Projected Data")
    st.dataframe(proj_df[['Month','Emails Sent','Orders','Revenue','CR%']])

    st.line_chart(proj_df.set_index('Month')[['Emails Sent','Orders','Revenue']])

    # --- Export option ---
    csv = proj_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Projection CSV", csv, f"projection_{future_year}.csv", "text/csv")

else:
    st.info("👆 Upload a CSV/Excel file to start.")
