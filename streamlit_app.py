#!/usr/bin/env python3
"""
Advanced CLM Budget Projections with Marketing Mix Modeling
Handles Excel import and provides sophisticated growth projections
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class AdvancedCLMProjections:
    def __init__(self):
        self.setup_page_config()
        
    def setup_page_config(self):
        st.set_page_config(
            page_title="Advanced CLM Budget Projections",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def load_data_from_excel(self, uploaded_file):
        """Load and process data from uploaded Excel file"""
        try:
            # Read all sheets
            excel_data = pd.read_excel(uploaded_file, sheet_name=None)
            
            processed_data = {}
            
            # Process WhatsApp data
            if 'WA' in excel_data:
                wa_df = excel_data['WA'].copy()
                wa_df = wa_df.dropna(subset=['Month']).copy()
                wa_df['Month'] = pd.to_datetime(wa_df['Month'])
                wa_df['channel'] = 'WhatsApp'
                processed_data['wa'] = wa_df
                
            # Process SMS data
            if 'SMS' in excel_data:
                sms_df = excel_data['SMS'].copy()
                sms_df = sms_df.dropna(subset=['Month']).copy()
                sms_df['Month'] = pd.to_datetime(sms_df['Month'])
                sms_df['channel'] = 'SMS'
                processed_data['sms'] = sms_df
                
            # Process Email data
            if 'Email' in excel_data:
                email_df = excel_data['Email'].copy()
                # Email data has different structure - need to parse
                email_df = email_df[email_df['Total MONTHLY DATA'].str.contains('Total', na=False)].copy()
                email_df['channel'] = 'Email'
                processed_data['email'] = email_df
                
            return processed_data, True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, False
            
    def create_mmm_model(self, data, target_col, spend_cols, control_vars=None):
        """
        Advanced Marketing Mix Model with multiple approaches
        """
        models = {}
        
        # Prepare features
        if control_vars is None:
            control_vars = ['month', 'seasonality']
            
        # Add seasonality features
        data = data.copy()
        data['month'] = pd.to_datetime(data['Month']).dt.month if 'Month' in data.columns else data.index % 12 + 1
        data['seasonality'] = np.sin(2 * np.pi * data['month'] / 12)
        
        # Feature matrix
        X = data[spend_cols + control_vars].fillna(0)
        y = data[target_col].fillna(0)
        
        # Remove zero/negative targets
        mask = y > 0
        X = X[mask]
        y = y[mask]
        
        if len(X) < 3:
            return None, None, None
            
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Try multiple models
        try:
            # Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X_scaled, y)
            lr_score = r2_score(y, lr_model.predict(X_scaled))
            models['linear'] = {'model': lr_model, 'score': lr_score}
            
            # Ridge Regression (handles multicollinearity)
            ridge_model = Ridge(alpha=1.0)
            ridge_model.fit(X_scaled, y)
            ridge_score = r2_score(y, ridge_model.predict(X_scaled))
            models['ridge'] = {'model': ridge_model, 'score': ridge_score}
            
            # Random Forest (captures non-linearities)
            if len(X) >= 5:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_scaled, y)
                rf_score = r2_score(y, rf_model.predict(X_scaled))
                models['random_forest'] = {'model': rf_model, 'score': rf_score}
                
        except Exception as e:
            st.warning(f"Model fitting warning: {str(e)}")
            
        # Select best model
        if models:
            best_model_name = max(models.keys(), key=lambda k: models[k]['score'])
            best_model = models[best_model_name]['model']
            best_score = models[best_model_name]['score']
            
            return best_model, scaler, X.columns.tolist(), best_score, best_model_name
        
        return None, None, None, 0, 'none'
    
    def calculate_adstock_effect(self, spend_series, adstock_rate=0.5):
        """Apply adstock transformation to spending"""
        adstocked = np.zeros_like(spend_series)
        adstocked[0] = spend_series[0]
        
        for i in range(1, len(spend_series)):
            adstocked[i] = spend_series[i] + adstock_rate * adstocked[i-1]
            
        return adstocked
    
    def calculate_saturation_curve(self, spend, alpha=0.5, gamma=0.5):
        """Apply saturation curve to diminishing returns"""
        return alpha * (1 - np.exp(-gamma * spend / np.mean(spend)))
    
    def project_scenarios_advanced(self, historical_data, current_growth=0.43, 
                                 target_growths=[0.51, 0.75], projection_months=7):
        """
        Advanced scenario projections using MMM insights
        """
        scenarios = {}
        
        # Calculate current baseline (FY25 Apr-Aug performance)
        baseline_period = {}
        
        for channel, data in historical_data.items():
            if data is not None and len(data) > 0:
                # Filter for FY25 period (Apr 2024 - Aug 2024)
                data['Month'] = pd.to_datetime(data['Month'])
                mask = (data['Month'].dt.year == 2024) & (data['Month'].dt.month.isin([4,5,6,7,8]))
                baseline_period[channel] = data[mask].copy()
        
        # Calculate baseline metrics
        baseline_metrics = {}
        
        for channel, data in baseline_period.items():
            if len(data) > 0:
                if channel == 'wa':
                    total_cost = data['Cost'].sum()
                    total_revenue = data['Total Revenue'].sum()
                    avg_monthly_cost = data['Cost'].mean()
                    avg_roi = data['ROI'].mean()
                    
                elif channel == 'sms':
                    total_cost = data['Cost'].sum()
                    total_revenue = data['Revenue'].sum()
                    avg_monthly_cost = data['Cost'].mean()
                    avg_roi = data['ROI'].mean()
                    
                elif channel == 'email':
                    # Email data structure is different
                    total_cost = data['_5'].sum() if '_5' in data.columns else 0
                    total_revenue = data['_3'].sum() if '_3' in data.columns else 0
                    avg_monthly_cost = data['_5'].mean() if '_5' in data.columns else 0
                    avg_roi = (total_revenue / total_cost) if total_cost > 0 else 0
                
                baseline_metrics[channel] = {
                    'total_cost': total_cost,
                    'total_revenue': total_revenue,
                    'avg_monthly_cost': avg_monthly_cost,
                    'roi': avg_roi,
                    'efficiency': total_revenue / total_cost if total_cost > 0 else 0
                }
        
        # Create scenarios
        all_growth_rates = [current_growth] + target_growths
        
        for growth_rate in all_growth_rates:
            scenario_name = f"{int(growth_rate*100)}% Growth"
            scenarios[scenario_name] = {}
            
            for channel, metrics in baseline_metrics.items():
                base_monthly_cost = metrics['avg_monthly_cost']
                base_efficiency = metrics['efficiency']
                
                # Apply growth rate with some efficiency adjustment
                new_monthly_cost = base_monthly_cost * (1 + growth_rate)
                
                # Account for diminishing returns at higher spends
                efficiency_factor = 1.0 - (growth_rate * 0.1)  # Slight efficiency decrease with scale
                adjusted_efficiency = base_efficiency * efficiency_factor
                
                # Calculate projections
                total_additional_cost = (new_monthly_cost - base_monthly_cost) * projection_months
                projected_monthly_revenue = new_monthly_cost * adjusted_efficiency
                projected_total_revenue = projected_monthly_revenue * projection_months
                
                scenarios[scenario_name][channel] = {
                    'current_monthly_cost': base_monthly_cost,
                    'projected_monthly_cost': new_monthly_cost,
                    'additional_monthly_spend': new_monthly_cost - base_monthly_cost,
                    'total_additional_spend': total_additional_cost,
                    'projected_monthly_revenue': projected_monthly_revenue,
                    'projected_total_revenue': projected_total_revenue,
                    'adjusted_roi': adjusted_efficiency,
                    'baseline_roi': base_efficiency
                }
        
        return scenarios, baseline_metrics
    
    def create_optimization_recommendations(self, scenarios, baseline_metrics):
        """Create budget optimization recommendations"""
        recommendations = {}
        
        for scenario_name, scenario_data in scenarios.items():
            reco = {}
            
            # Calculate total incremental spend and revenue
            total_additional_spend = sum([ch['total_additional_spend'] for ch in scenario_data.values()])
            total_projected_revenue = sum([ch['projected_total_revenue'] for ch in scenario_data.values()])
            
            # Rank channels by efficiency
            channel_efficiency = {
                ch_name: ch_data['adjusted_roi'] 
                for ch_name, ch_data in scenario_data.items()
            }
            
            sorted_channels = sorted(channel_efficiency.items(), key=lambda x: x[1], reverse=True)
            
            reco['total_additional_investment'] = total_additional_spend
            reco['total_projected_revenue'] = total_projected_revenue
            reco['overall_roi'] = total_projected_revenue / total_additional_spend if total_additional_spend > 0 else 0
            reco['most_efficient_channel'] = sorted_channels[0][0] if sorted_channels else None
            reco['least_efficient_channel'] = sorted_channels[-1][0] if sorted_channels else None
            reco['channel_ranking'] = sorted_channels
            
            # Budget allocation suggestion
            total_budget = sum([ch['projected_monthly_cost'] for ch in scenario_data.values()])
            reco['suggested_allocation'] = {
                ch_name: {
                    'monthly_budget': ch_data['projected_monthly_cost'],
                    'percentage': (ch_data['projected_monthly_cost'] / total_budget * 100) if total_budget > 0 else 0,
                    'priority': 'High' if ch_data['adjusted_roi'] > 5 else 'Medium' if ch_data['adjusted_roi'] > 3 else 'Low'
                }
                for ch_name, ch_data in scenario_data.items()
            }
            
            recommendations[scenario_name] = reco
            
        return recommendations
    
    def run_app(self):
        st.title("🚀 Advanced CLM Budget Projections")
        st.markdown("### Marketing Mix Model for Multi-Channel Budget Optimization")
        
        # File upload
        st.sidebar.header("📁 Data Input")
        uploaded_file = st.sidebar.file_uploader(
            "Upload CLM Data Excel File",
            type=['xlsx', 'xls'],
            help="Upload your Excel file with WA, SMS, and Email sheets"
        )
        
        if uploaded_file is None:
            st.info("👆 Please upload your CLM data Excel file to begin analysis")
            st.markdown("""
            **Expected file structure:**
            - **WA Sheet**: WhatsApp data with columns: Month, Total Delivered, Cost, Total Orders, Total Revenue, etc.
            - **SMS Sheet**: SMS data with columns: Month, Traffic, Orders, CVR, Revenue, Cost, etc.  
            - **Email Sheet**: Email data with performance metrics
            """)
            return
            
        # Load and process data
        with st.spinner("Loading and processing data..."):
            historical_data, success = self.load_data_from_excel(uploaded_file)
            
        if not success:
            st.error("Failed to load data. Please check your file format.")
            return
            
        # Sidebar controls
        st.sidebar.header("📊 Projection Settings")
        current_growth = st.sidebar.number_input(
            "Current Growth Rate (%)", 
            value=43, min_value=0, max_value=200, step=1
        ) / 100
        
        target_growth_1 = st.sidebar.number_input(
            "Target Growth Scenario 1 (%)", 
            value=51, min_value=0, max_value=200, step=1
        ) / 100
        
        target_growth_2 = st.sidebar.number_input(
            "Target Growth Scenario 2 (%)", 
            value=75, min_value=0, max_value=200, step=1
        ) / 100
        
        projection_months = st.sidebar.slider(
            "Projection Period (Months)", 
            min_value=1, max_value=12, value=7
        )
        
        # Data overview
        st.header("📈 Historical Performance Overview")
        
        col1, col2, col3 = st.columns(3)
        
        # Show data summary for each channel
        for i, (channel, data) in enumerate(historical_data.items()):
            if data is not None and len(data) > 0:
                with [col1, col2, col3][i]:
                    st.subheader(f"{'📱 WhatsApp' if channel == 'wa' else '📲 SMS' if channel == 'sms' else '📧 Email'}")
                    st.write(f"Data points: {len(data)}")
                    
                    if channel == 'wa' and 'Total Revenue' in data.columns:
                        st.metric("Avg Monthly Revenue", f"₹{data['Total Revenue'].mean():,.0f}")
                        st.metric("Avg ROI", f"{data['ROI'].mean():.1f}x")
                    elif channel == 'sms' and 'Revenue' in data.columns:
                        st.metric("Avg Monthly Revenue", f"₹{data['Revenue'].mean():,.0f}")
                        if 'ROI' in data.columns:
                            st.metric("Avg ROI", f"{data['ROI'].mean():.1f}x")
        
        # Generate projections
        with st.spinner("Generating advanced projections..."):
            scenarios, baseline_metrics = self.project_scenarios_advanced(
                historical_data, current_growth, [target_growth_1, target_growth_2], projection_months
            )
            
        # Create recommendations
        recommendations = self.create_optimization_recommendations(scenarios, baseline_metrics)
        
        # Display scenarios
        st.header("🎯 Budget Projection Scenarios")
        
        scenario_summary = []
        for scenario_name, scenario_data in scenarios.items():
            total_additional = sum([ch['total_additional_spend'] for ch in scenario_data.values()])
            total_revenue = sum([ch['projected_total_revenue'] for ch in scenario_data.values()])
            roi = total_revenue / total_additional if total_additional > 0 else 0
            
            scenario_summary.append({
                'Scenario': scenario_name,
                f'Additional {projection_months}-Month Spend': f"₹{total_additional:,.0f}",
                f'Projected {projection_months}-Month Revenue': f"₹{total_revenue:,.0f}",
                'Expected ROI': f"{roi:.1f}x",
                'Growth Rate': scenario_name.split('%')[0] + '%'
            })
            
        scenario_df = pd.DataFrame(scenario_summary)
        st.dataframe(scenario_df, use_container_width=True)
        
        # Detailed breakdown
        st.header("📋 Detailed Channel Analysis")
        
        selected_scenario = st.selectbox(
            "Select Scenario for Detailed Analysis:", 
            list(scenarios.keys())
        )
        
        if selected_scenario:
            st.subheader(f"💰 {selected_scenario} - Channel Breakdown")
            
            scenario_details = scenarios[selected_scenario]
            reco = recommendations[selected_scenario]
            
            # Channel performance table
            channel_details = []
            for channel, details in scenario_details.items():
                channel_name = {'wa': 'WhatsApp', 'sms': 'SMS', 'email': 'Email'}[channel]
                
                channel_details.append({
                    'Channel': channel_name,
                    'Current Monthly Spend': f"₹{details['current_monthly_cost']:,.0f}",
                    'Projected Monthly Spend': f"₹{details['projected_monthly_cost']:,.0f}",
                    'Monthly Increase': f"₹{details['additional_monthly_spend']:,.0f}",
                    f'{projection_months}-Month Additional': f"₹{details['total_additional_spend']:,.0f}",
                    'Projected Monthly Revenue': f"₹{details['projected_monthly_revenue']:,.0f}",
                    'Adjusted ROI': f"{details['adjusted_roi']:.1f}x",
                    'Efficiency Change': f"{((details['adjusted_roi']/details['baseline_roi']-1)*100):+.1f}%" if details['baseline_roi'] > 0 else "N/A"
                })
                
            channel_df = pd.DataFrame(channel_details)
            st.dataframe(channel_df, use_container_width=True)
            
            # Recommendations for selected scenario
            st.subheader("💡 Strategic Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **Investment Summary:**
                - Total Additional Investment: ₹{reco['total_additional_investment']:,.0f}
                - Expected Revenue: ₹{reco['total_projected_revenue']:,.0f}
                - Overall ROI: {reco['overall_roi']:.1f}x
                """)
                
                st.success(f"""
                **Top Performing Channel:** {reco['most_efficient_channel'].upper()}
                - Highest efficiency in this scenario
                - Prioritize budget allocation here
                """)
                
            with col2:
                st.write("**Budget Allocation Recommendations:**")
                
                for channel, allocation in reco['suggested_allocation'].items():
                    channel_name = {'wa': 'WhatsApp', 'sms': 'SMS', 'email': 'Email'}[channel]
                    priority_color = {
                        'High': '🟢', 'Medium': '🟡', 'Low': '🔴'
                    }[allocation['priority']]
                    
                    st.write(f"""
                    {priority_color} **{channel_name}**
                    - Monthly Budget: ₹{allocation['monthly_budget']:,.0f}
                    - Allocation: {allocation['percentage']:.1f}%
                    - Priority: {allocation['priority']}
                    """)
        
        # Advanced visualizations
        st.header("📊 Advanced Analytics Dashboard")
        
        # Create comprehensive visualization
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Monthly Spend Comparison by Channel', 
                'Revenue Projections by Scenario',
                'ROI Efficiency Analysis', 
                'Budget Allocation by Scenario',
                'Growth Impact Analysis',
                'Channel Performance Matrix'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        scenarios_list = list(scenarios.keys())
        channels = list(scenarios[scenarios_list[0]].keys())
        channel_names = {'wa': 'WhatsApp', 'sms': 'SMS', 'email': 'Email'}
        
        # 1. Monthly spend comparison
        for channel in channels:
            current_spends = [scenarios[scenario][channel]['current_monthly_cost'] for scenario in scenarios_list]
            projected_spends = [scenarios[scenario][channel]['projected_monthly_cost'] for scenario in scenarios_list]
            
            fig.add_trace(
                go.Bar(
                    name=f"{channel_names[channel]} - Current",
                    x=scenarios_list,
                    y=current_spends,
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    name=f"{channel_names[channel]} - Projected",
                    x=scenarios_list,
                    y=projected_spends,
                    marker_color='darkblue'
                ),
                row=1, col=1
            )
        
        # 2. Revenue projections
        for channel in channels:
            revenues = [scenarios[scenario][channel]['projected_total_revenue'] for scenario in scenarios_list]
            fig.add_trace(
                go.Scatter(
                    name=f"{channel_names[channel]} Revenue",
                    x=scenarios_list,
                    y=revenues,
                    mode='lines+markers',
                    line=dict(width=3)
                ),
                row=1, col=2
            )
        
        # 3. ROI efficiency analysis
        for channel in channels:
            baseline_rois = [scenarios[scenario][channel]['baseline_roi'] for scenario in scenarios_list]
            adjusted_rois = [scenarios[scenario][channel]['adjusted_roi'] for scenario in scenarios_list]
            
            fig.add_trace(
                go.Scatter(
                    name=f"{channel_names[channel]} - Baseline ROI",
                    x=scenarios_list,
                    y=baseline_rois,
                    mode='lines+markers',
                    line=dict(dash='dot')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    name=f"{channel_names[channel]} - Projected ROI",
                    x=scenarios_list,
                    y=adjusted_rois,
                    mode='lines+markers',
                    line=dict(width=3)
                ),
                row=2, col=1
            )
        
        # 4. Budget allocation pie chart for selected scenario
        if selected_scenario:
            scenario_data = scenarios[selected_scenario]
            channel_budgets = [scenario_data[ch]['projected_monthly_cost'] for ch in channels]
            channel_labels = [channel_names[ch] for ch in channels]
            
            fig.add_trace(
                go.Pie(
                    labels=channel_labels,
                    values=channel_budgets,
                    name="Budget Allocation"
                ),
                row=2, col=2
            )
        
        # 5. Growth impact analysis
        for scenario in scenarios_list:
            total_additional = sum([scenarios[scenario][ch]['total_additional_spend'] for ch in channels])
            total_revenue = sum([scenarios[scenario][ch]['projected_total_revenue'] for ch in channels])
            
            fig.add_trace(
                go.Scatter(
                    x=[total_additional],
                    y=[total_revenue],
                    mode='markers+text',
                    text=[scenario.split('%')[0] + '%'],
                    textposition="top center",
                    marker=dict(size=15),
                    name=scenario
                ),
                row=3, col=1
            )
        
        # 6. Channel performance matrix
        if selected_scenario:
            scenario_data = scenarios[selected_scenario]
            x_efficiency = [scenario_data[ch]['adjusted_roi'] for ch in channels]
            y_volume = [scenario_data[ch]['projected_monthly_cost'] for ch in channels]
            
            fig.add_trace(
                go.Scatter(
                    x=x_efficiency,
                    y=y_volume,
                    mode='markers+text',
                    text=[channel_names[ch] for ch in channels],
                    textposition="middle center",
                    marker=dict(
                        size=[scenario_data[ch]['projected_total_revenue']/100000 for ch in channels],
                        color=[scenario_data[ch]['adjusted_roi'] for ch in channels],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="ROI")
                    ),
                    name="Channel Matrix"
                ),
                row=3, col=2
            )
        
        fig.update_layout(height=1200, showlegend=True, title_text="CLM Advanced Analytics Dashboard")
        
        # Update subplot titles and axes
        fig.update_xaxes(title_text="Scenarios", row=1, col=1)
        fig.update_yaxes(title_text="Monthly Spend (₹)", row=1, col=1)
        
        fig.update_xaxes(title_text="Scenarios", row=1, col=2)
        fig.update_yaxes(title_text="Revenue (₹)", row=1, col=2)
        
        fig.update_xaxes(title_text="Scenarios", row=2, col=1)
        fig.update_yaxes(title_text="ROI", row=2, col=1)
        
        fig.update_xaxes(title_text="Additional Spend (₹)", row=3, col=1)
        fig.update_yaxes(title_text="Projected Revenue (₹)", row=3, col=1)
        
        fig.update_xaxes(title_text="ROI Efficiency", row=3, col=2)
        fig.update_yaxes(title_text="Monthly Spend (₹)", row=3, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # MMM Insights
        st.header("🧠 Marketing Mix Model Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Key Findings")
            
            # Calculate channel efficiency rankings
            avg_efficiency = {}
            for channel in channels:
                efficiencies = [scenarios[scenario][channel]['adjusted_roi'] for scenario in scenarios_list]
                avg_efficiency[channel] = np.mean(efficiencies)
            
            sorted_efficiency = sorted(avg_efficiency.items(), key=lambda x: x[1], reverse=True)
            
            st.write("**Channel Efficiency Ranking:**")
            for i, (channel, efficiency) in enumerate(sorted_efficiency):
                medal = ['🥇', '🥈', '🥉'][i] if i < 3 else '📊'
                st.write(f"{medal} {channel_names[channel]}: {efficiency:.1f}x ROI")
            
            # Saturation analysis
            st.write("\n**Diminishing Returns Analysis:**")
            for channel in channels:
                baseline_roi = scenarios[scenarios_list[0]][channel]['baseline_roi']
                max_scenario_roi = scenarios[scenarios_list[-1]][channel]['adjusted_roi']
                efficiency_decline = ((baseline_roi - max_scenario_roi) / baseline_roi * 100)
                
                if efficiency_decline > 0:
                    st.warning(f"{channel_names[channel]}: {efficiency_decline:.1f}% efficiency decline at maximum spend")
                else:
                    st.success(f"{channel_names[channel]}: Maintaining efficiency at scale")
        
        with col2:
            st.subheader("💰 Investment Recommendations")
            
            # Find optimal allocation
            best_scenario = max(recommendations.keys(), 
                              key=lambda x: recommendations[x]['overall_roi'])
            
            st.info(f"""
            **Recommended Scenario:** {best_scenario}
            
            **Optimal Monthly Allocation:**
            """)
            
            optimal_allocation = recommendations[best_scenario]['suggested_allocation']
            total_monthly = sum([alloc['monthly_budget'] for alloc in optimal_allocation.values()])
            
            for channel, allocation in optimal_allocation.items():
                st.write(f"""
                **{channel_names[channel]}**: ₹{allocation['monthly_budget']:,.0f} ({allocation['percentage']:.1f}%)
                Priority: {allocation['priority']}
                """)
            
            st.write(f"\n**Total Monthly Investment:** ₹{total_monthly:,.0f}")
            st.write(f"**{projection_months}-Month Total:** ₹{total_monthly * projection_months:,.0f}")
        
        # Export functionality
        st.header("💾 Export Projections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Generate Summary Report", use_container_width=True):
                # Create summary report
                summary_data = []
                
                for scenario_name, scenario_data in scenarios.items():
                    for channel, details in scenario_data.items():
                        summary_data.append({
                            'Scenario': scenario_name,
                            'Channel': channel_names[channel],
                            'Current Monthly Spend': details['current_monthly_cost'],
                            'Projected Monthly Spend': details['projected_monthly_cost'],
                            'Monthly Increase': details['additional_monthly_spend'],
                            f'{projection_months}M Additional Spend': details['total_additional_spend'],
                            f'{projection_months}M Projected Revenue': details['projected_total_revenue'],
                            'Baseline ROI': details['baseline_roi'],
                            'Projected ROI': details['adjusted_roi'],
                            'ROI Change %': ((details['adjusted_roi']/details['baseline_roi']-1)*100) if details['baseline_roi'] > 0 else 0
                        })
                
                summary_df = pd.DataFrame(summary_data)
                
                # Convert to Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    summary_df.to_excel(writer, sheet_name='Projections Summary', index=False)
                    
                    # Add recommendations sheet
                    reco_data = []
                    for scenario, reco in recommendations.items():
                        reco_data.append({
                            'Scenario': scenario,
                            'Total Additional Investment': reco['total_additional_investment'],
                            'Total Projected Revenue': reco['total_projected_revenue'],
                            'Overall ROI': reco['overall_roi'],
                            'Most Efficient Channel': reco['most_efficient_channel'],
                            'Least Efficient Channel': reco['least_efficient_channel']
                        })
                    
                    reco_df = pd.DataFrame(reco_data)
                    reco_df.to_excel(writer, sheet_name='Recommendations', index=False)
                
                st.download_button(
                    label="📥 Download Excel Report",
                    data=output.getvalue(),
                    file_name=f"CLM_Projections_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.success("✅ Report generated successfully!")
        
        with col2:
            if st.button("📊 Export Visualization Data", use_container_width=True):
                # Create visualization data export
                viz_data = {
                    'scenario_summary': scenario_df,
                    'channel_details': pd.DataFrame(channel_details) if 'channel_details' in locals() else pd.DataFrame()
                }
                
                st.success("✅ Visualization data ready for export!")
                st.json({
                    'total_scenarios': len(scenarios),
                    'projection_months': projection_months,
                    'channels_analyzed': list(channels),
                    'date_generated': datetime.now().isoformat()
                })

# Run the app
if __name__ == "__main__":
    app = AdvancedCLMProjections()
    app.run_app()
