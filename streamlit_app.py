import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="CLM Budget Projections", layout="wide")

st.title("📩 CLM Budget Projections – Email, SMS, WA")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload your Excel file with WA, SMS, Email sheets", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)

    # Load sheets
    sheets = {s: pd.read_excel(uploaded_file, sheet_name=s) for s in xls.sheet_names if s in ["WA","SMS","Email"]}

    st.sidebar.header("Projection Settings")
    growth_targets = [43, 51, 75]

    channel_outputs = {}

    for channel, df in sheets.items():
        st.subheader(f"📊 {channel} Historical Data Preview")
        st.dataframe(df.head())

        # --- Normalize metrics per channel ---
        if channel == "Email":
            df.columns = df.iloc[0]
            df = df.drop(0)
            df = df.rename(columns={"MONTH":"Month","Emails Sent":"Sends","Revenue(INR)":"Revenue","COST":"Cost"})
            df = df[["Month","Sends","Revenue","Cost"]]

        elif channel == "SMS":
            df = df.rename(columns={"Month":"Month","Submitted":"Sends","Revenue":"Revenue","Cost":"Cost"})
            df = df[["Month","Sends","Revenue","Cost"]]

        elif channel == "WA":
            df = df.rename(columns={"Month":"Month","Total Delivered":"Sends","Total Revenue":"Revenue","Cost":"Cost"})
            df = df[["Month","Sends","Revenue","Cost"]]

        # Convert numeric
        for c in ["Sends","Revenue","Cost"]:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # --- Compute ROI & revenue per send ---
        roi = (df["Revenue"].sum() / df["Cost"].sum()) if df["Cost"].sum() > 0 else 0
        rev_per_send = (df["Revenue"].sum() / df["Sends"].sum()) if df["Sends"].sum() > 0 else 0

        # --- Last FY Sep–Mar seasonality ---
        fy_df = df.tail(18)  # last 1.5 years approx
        fy_df = fy_df[fy_df["Month"].str.contains("Sep|Oct|Nov|Dec|Jan|Feb|Mar", case=False, regex=True)]
        seasonality = fy_df[["Month","Sends"]].copy()
        seasonality["Weight"] = seasonality["Sends"] / seasonality["Sends"].sum()

        # --- Projection base (Aug actuals + growth target) ---
        total_ytd_growth = 0.43  # baseline achieved till Aug
        base_revenue = df["Revenue"].sum()

        scenarios = {}
        for g in growth_targets:
            target_growth = g/100
            required_rev = base_revenue * (target_growth/total_ytd_growth)
            extra_rev = required_rev - base_revenue
            extra_cost = extra_rev / roi if roi > 0 else 0
            extra_sends = extra_rev / rev_per_send if rev_per_send > 0 else 0

            # Allocate by seasonality
            alloc = seasonality.copy()
            alloc["Projected_Sends"] = alloc["Weight"] * extra_sends
            alloc["Projected_Cost"] = alloc["Weight"] * extra_cost
            alloc["Projected_Revenue"] = alloc["Projected_Cost"] * roi
            alloc["ROI"] = roi

            scenarios[g] = alloc

        channel_outputs[channel] = scenarios

        # --- Show tables ---
        for g, alloc in scenarios.items():
            st.write(f"### {channel} – {g}% Growth Projection")
            st.dataframe(alloc[["Month","Projected_Sends","Projected_Cost","Projected_Revenue","ROI"]])

    # --- Combine into CSV ---
    all_records = []
    for ch, scenarios in channel_outputs.items():
        for g, alloc in scenarios.items():
            temp = alloc.copy()
            temp["Channel"] = ch
            temp["GrowthScenario"] = g
            all_records.append(temp)
    result = pd.concat(all_records)

    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download All Projections", csv, "CLM_Projections.csv", "text/csv")

else:
    st.info("👆 Upload the projections Excel to start.")
