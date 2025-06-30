# SolarSmart - AI-Driven Solar Panel Performance Forecasting Tool
# Updated with Enhanced Performance Analyzer

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import requests
import json
import datetime
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="SolarSmart - AI Solar Forecasting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .performance-good {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 5px 0;
    }
    .performance-warning {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 5px 0;
    }
    .performance-danger {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

class WeatherAPI:
    """Mock Weather API for demonstration"""
    @staticmethod
    def get_weather_data(location="New York", days=7):
        # Simulating weather API response
        dates = [datetime.datetime.now() + timedelta(days=i) for i in range(days)]
        weather_data = []
        
        for date in dates:
            weather_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'temperature': np.random.normal(25, 5),  # Celsius
                'humidity': np.random.normal(60, 15),    # %
                'irradiance': np.random.normal(800, 200), # W/m²
                'wind_speed': np.random.normal(10, 5),   # km/h
                'cloud_cover': np.random.normal(30, 20)  # %
            })
        
        return pd.DataFrame(weather_data)

class SolarDataGenerator:
    """Generate synthetic solar panel data for demonstration"""
    
    @staticmethod
    def generate_historical_data(days=365):
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        data = []
        for i, date in enumerate(dates):
            # Seasonal patterns
            season_factor = 0.8 + 0.4 * np.sin(2 * np.pi * i / 365)
            
            # Daily patterns (peak at noon)
            for hour in range(24):
                hour_factor = max(0, np.sin(np.pi * (hour - 6) / 12))
                
                base_irradiance = 1000 * season_factor * hour_factor
                noise = np.random.normal(0, 50)
                irradiance = max(0, base_irradiance + noise)
                
                # Panel efficiency (degradation over time)
                panel_efficiency = 0.2 - (i * 0.0001)  # Slight degradation
                
                energy_output = irradiance * panel_efficiency * 10  # 10 m² panel
                
                data.append({
                    'datetime': date + pd.Timedelta(hours=hour),
                    'irradiance': irradiance,
                    'temperature': 20 + 15 * hour_factor + np.random.normal(0, 3),
                    'humidity': 50 + np.random.normal(0, 10),
                    'energy_output': max(0, energy_output),
                    'panel_voltage': 24 + np.random.normal(0, 1),
                    'panel_current': max(0, energy_output / 24 + np.random.normal(0, 0.5)),
                    'panel_id': f'Panel_0{i%5 + 1}'
                })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_realistic_data(num_panels=10, days=30):
        """Generate more realistic data with varied panel performance"""
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        data = []
        panel_base_efficiency = np.random.normal(0.2, 0.02, num_panels)  # Different efficiency per panel
        panel_degradation = np.random.normal(0.5, 0.2, num_panels)  # Different degradation rates
        
        for i, date in enumerate(dates):
            # Seasonal patterns
            season_factor = 0.8 + 0.4 * np.sin(2 * np.pi * i / 365)
            
            # Daily patterns (peak at noon)
            for hour in range(6, 19):  # Only daylight hours
                hour_factor = max(0, np.sin(np.pi * (hour - 6) / 12))
                
                base_irradiance = 1000 * season_factor * hour_factor
                noise = np.random.normal(0, 50)
                irradiance = max(0, base_irradiance + noise)
                
                temperature = 20 + 15 * hour_factor + np.random.normal(0, 3)
                humidity = 50 + np.random.normal(0, 10)
                
                for panel_idx in range(num_panels):
                    panel_id = f'Panel_{panel_idx+1:02d}'
                    
                    # Panel-specific efficiency
                    efficiency = panel_base_efficiency[panel_idx] * (1 - panel_degradation[panel_idx] * i / 10000)
                    
                    # Random faults for some panels
                    fault_probability = 0.005  # 0.5% chance of fault per reading
                    if np.random.random() < fault_probability:
                        efficiency *= np.random.uniform(0.1, 0.6)  # Severe performance drop
                    
                    energy_output = irradiance * efficiency * 10  # 10 m² panel
                    voltage = 24 + np.random.normal(0, 1)
                    current = max(0, energy_output / 24 + np.random.normal(0, 0.5))
                    
                    data.append({
                        'datetime': date + pd.Timedelta(hours=hour, minutes=np.random.randint(0, 60)),
                        'panel_id': panel_id,
                        'irradiance': irradiance,
                        'temperature': temperature,
                        'humidity': humidity,
                        'energy_output': max(0, energy_output),
                        'panel_voltage': voltage,
                        'panel_current': current,
                        'panel_power': max(0, voltage * current),
                        'ambient_temp': temperature + np.random.normal(0, 2),
                        'wind_speed': np.random.exponential(5)
                    })
        
        return pd.DataFrame(data)

class EnhancedAnomalyDetector:
    """Enhanced anomaly detection with multiple methods"""
    
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.detector = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        
    def detect_anomalies(self, data):
        """Detect anomalies using multiple features"""
        # Features for anomaly detection
        features = ['energy_output', 'panel_voltage', 'panel_current', 'panel_power']
        
        # Handle missing features
        available_features = [f for f in features if f in data.columns]
        
        if len(available_features) < 2:
            st.warning("Insufficient features for anomaly detection. Need at least energy_output and one other metric.")
            return data
        
        # Scale features
        feature_data = self.scaler.fit_transform(data[available_features].fillna(0))
        
        # Fit the detector
        self.detector.fit(feature_data)
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        anomalies = self.detector.predict(feature_data)
        anomaly_scores = self.detector.score_samples(feature_data)
        
        # Add results to dataframe
        data_copy = data.copy()
        data_copy['anomaly'] = anomalies
        data_copy['anomaly_score'] = anomaly_scores
        data_copy['is_anomaly'] = anomalies == -1
        
        return data_copy
    
    def analyze_panel_health(self, data):
        """Analyze individual panel health"""
        panel_health = {}
        
        for panel_id in data['panel_id'].unique():
            panel_data = data[data['panel_id'] == panel_id]
            
            # Calculate health metrics
            anomaly_rate = (panel_data['is_anomaly'].sum() / len(panel_data)) * 100
            avg_output = panel_data['energy_output'].mean()
            output_std = panel_data['energy_output'].std()
            voltage_stability = panel_data['panel_voltage'].std()
            
            # Determine health status
            if anomaly_rate > 15:
                health_status = "Critical"
                priority = 1
            elif anomaly_rate > 8:
                health_status = "Poor"
                priority = 2
            elif anomaly_rate > 3:
                health_status = "Fair"
                priority = 3
            else:
                health_status = "Good"
                priority = 4
            
            panel_health[panel_id] = {
                'health_status': health_status,
                'anomaly_rate': anomaly_rate,
                'avg_output': avg_output,
                'output_stability': output_std,
                'voltage_stability': voltage_stability,
                'priority': priority,
                'total_readings': len(panel_data),
                'anomaly_count': panel_data['is_anomaly'].sum()
            }
        
        return panel_health

class SimpleSolarPredictor:
    """Simplified solar energy prediction using Random Forest"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, data):
        """Prepare data for training"""
        features = ['irradiance', 'temperature', 'humidity']
        target = 'energy_output'
        
        # Check if all features are available
        available_features = [f for f in features if f in data.columns]
        
        if len(available_features) < len(features):
            st.warning(f"Missing features: {set(features) - set(available_features)}")
        
        # Scale features
        X = self.scaler.fit_transform(data[available_features])
        y = data[target].values
        
        return X, y
    
    def train(self, data):
        """Train the Random Forest model"""
        X, y = self.prepare_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {'mae': mae, 'r2': r2}, X_test, y_test
    
    def predict(self, weather_data):
        """Make predictions based on weather forecast"""
        if self.model is None:
            return None
        
        features = ['irradiance', 'temperature', 'humidity']
        available_features = [f for f in features if f in weather_data.columns]
        X = self.scaler.transform(weather_data[available_features])
        
        predictions = self.model.predict(X)
        return predictions

def main():
    # Header
    st.markdown('<h1 class="main-header">☀️ SolarSmart - AI Solar Forecasting</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🏠 Dashboard",
        "📈 Performance Forecasting",
        "🔍 Enhanced Performance Analyzer",
        "🎯 Scenario Simulator",
        "📊 Data Upload"
    ])
    
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "📈 Performance Forecasting":
        forecasting_page()
    elif page == "🔍 Enhanced Performance Analyzer":
        enhanced_efficiency_page()
    elif page == "🎯 Scenario Simulator":
        simulator_page()
    elif page == "📊 Data Upload":
        data_upload_page()

def dashboard_page():
    st.header("Solar Performance Dashboard")
    
    # Generate sample data
    data = SolarDataGenerator.generate_historical_data(30)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_energy = data['energy_output'].sum()
        st.metric("Total Energy (kWh)", f"{total_energy:.1f}", delta="12.5%")
    
    with col2:
        avg_efficiency = data['energy_output'].mean()
        st.metric("Avg Daily Output", f"{avg_efficiency:.1f} kWh", delta="5.2%")
    
    with col3:
        max_output = data['energy_output'].max()
        st.metric("Peak Output", f"{max_output:.1f} kWh", delta="8.1%")
    
    with col4:
        uptime = 98.5
        st.metric("System Uptime", f"{uptime:.1f}%", delta="0.5%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Energy Output Over Time")
        daily_output = data.groupby(data['datetime'].dt.date)['energy_output'].sum().reset_index()
        fig = px.line(daily_output, x='datetime', y='energy_output', 
                     title="Daily Energy Production")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Panel Performance Heatmap")
        panel_performance = data.groupby(['panel_id', data['datetime'].dt.date])['energy_output'].sum().reset_index()
        pivot_data = panel_performance.pivot(index='panel_id', columns='datetime', values='energy_output')
        fig = px.imshow(pivot_data, aspect="auto", title="Panel Performance Heatmap")
        st.plotly_chart(fig, use_container_width=True)

def forecasting_page():
    st.header("Solar Energy Forecasting")
    
    # User inputs
    col1, col2 = st.columns(2)
    
    with col1:
        location = st.text_input("Location", value="New York")
        forecast_days = st.slider("Forecast Days", 1, 14, 7)
    
    with col2:
        panel_capacity = st.number_input("Panel Capacity (kW)", value=10.0, step=0.5)
        panel_efficiency = st.slider("Panel Efficiency (%)", 15, 25, 20)
    
    if st.button("Generate Forecast"):
        # Get weather data
        weather_data = WeatherAPI.get_weather_data(location, forecast_days)
        
        # Generate historical data for training
        historical_data = SolarDataGenerator.generate_historical_data(100)
        
        # Train predictor
        predictor = SimpleSolarPredictor()
        with st.spinner("Training prediction model..."):
            metrics, X_test, y_test = predictor.train(historical_data)
        
        st.success(f"Model trained! R² Score: {metrics['r2']:.3f}, MAE: {metrics['mae']:.2f}")
        
        # Make predictions
        predictions = predictor.predict(weather_data)
        weather_data['predicted_output'] = predictions
        
        # Adjust predictions based on panel specifications
        weather_data['predicted_output'] = (weather_data['predicted_output'] * 
                                          panel_capacity / 10 * panel_efficiency / 20)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Weather Forecast")
            fig = make_subplots(rows=2, cols=2, 
                              subplot_titles=['Temperature', 'Irradiance', 'Humidity', 'Cloud Cover'])
            
            fig.add_trace(go.Scatter(x=weather_data['date'], y=weather_data['temperature'],
                                   name='Temperature'), row=1, col=1)
            fig.add_trace(go.Scatter(x=weather_data['date'], y=weather_data['irradiance'],
                                   name='Irradiance'), row=1, col=2)
            fig.add_trace(go.Scatter(x=weather_data['date'], y=weather_data['humidity'],
                                   name='Humidity'), row=2, col=1)
            fig.add_trace(go.Scatter(x=weather_data['date'], y=weather_data['cloud_cover'],
                                   name='Cloud Cover'), row=2, col=2)
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Energy Output Forecast")
            fig = px.line(weather_data, x='date', y='predicted_output',
                         title=f"Predicted Energy Output - {location}")
            fig.update_layout(yaxis_title="Energy Output (kWh)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            total_predicted = weather_data['predicted_output'].sum()
            avg_daily = weather_data['predicted_output'].mean()
            
            st.metric("Total Forecast Energy", f"{total_predicted:.1f} kWh")
            st.metric("Average Daily Output", f"{avg_daily:.1f} kWh")

def enhanced_efficiency_page():
    st.header("Enhanced Panel Performance Analyzer")
    
    # Data source selection
    data_source = st.radio("Select Data Source:", 
                          ["Generate Sample Data", "Manual Input", "Upload CSV"])
    
    data = None
    
    if data_source == "Generate Sample Data":
        st.subheader("Sample Data Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            num_panels = st.slider("Number of Panels", 5, 20, 10)
            days = st.slider("Days of Data", 7, 60, 30)
        
        with col2:
            fault_probability = st.slider("Fault Probability (%)", 0, 10, 2) / 100
            
        if st.button("Generate Data"):
            data = SolarDataGenerator.generate_realistic_data(num_panels, days)
            st.session_state['analysis_data'] = data
            st.success(f"Generated data for {num_panels} panels over {days} days")
    
    elif data_source == "Manual Input":
        st.subheader("Manual Data Entry")
        
        with st.expander("Add Manual Data Points"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                panel_id = st.text_input("Panel ID", value="Panel_01")
                energy_output = st.number_input("Energy Output (kWh)", value=5.0, step=0.1)
                panel_voltage = st.number_input("Panel Voltage (V)", value=24.0, step=0.1)
            
            with col2:
                panel_current = st.number_input("Panel Current (A)", value=2.5, step=0.1)
                temperature = st.number_input("Temperature (°C)", value=25.0, step=0.5)
                irradiance = st.number_input("Irradiance (W/m²)", value=800, step=10)
            
            with col3:
                humidity = st.number_input("Humidity (%)", value=60.0, step=1.0)
                wind_speed = st.number_input("Wind Speed (km/h)", value=10.0, step=0.5)
                
            if st.button("Add Data Point"):
                new_data = {
                    'datetime': [datetime.datetime.now()],
                    'panel_id': [panel_id],
                    'energy_output': [energy_output],
                    'panel_voltage': [panel_voltage],
                    'panel_current': [panel_current],
                    'panel_power': [panel_voltage * panel_current],
                    'temperature': [temperature],
                    'irradiance': [irradiance],
                    'humidity': [humidity],
                    'wind_speed': [wind_speed],
                    'ambient_temp': [temperature + np.random.normal(0, 1)]
                }
                
                if 'manual_data' not in st.session_state:
                    st.session_state['manual_data'] = pd.DataFrame(new_data)
                else:
                    st.session_state['manual_data'] = pd.concat([
                        st.session_state['manual_data'], 
                        pd.DataFrame(new_data)
                    ], ignore_index=True)
                
                st.success("Data point added!")
        
        if 'manual_data' in st.session_state and not st.session_state['manual_data'].empty:
            st.subheader("Current Manual Data")
            st.dataframe(st.session_state['manual_data'])
            
            if st.button("Analyze Manual Data"):
                data = st.session_state['manual_data']
                st.session_state['analysis_data'] = data
    
    elif data_source == "Upload CSV":
        st.subheader("CSV File Upload")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="performance_csv")
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                
                st.subheader("Data Preview")
                st.dataframe(data.head())
                
                # Column mapping
                st.subheader("Column Mapping")
                col1, col2, col3 = st.columns(3)
                
                required_columns = ['panel_id', 'energy_output', 'panel_voltage', 'panel_current']
                column_mapping = {}
                
                with col1:
                    column_mapping['panel_id'] = st.selectbox("Panel ID Column", data.columns)
                    column_mapping['energy_output'] = st.selectbox("Energy Output Column", data.columns)
                
                with col2:
                    column_mapping['panel_voltage'] = st.selectbox("Panel Voltage Column", data.columns)
                    column_mapping['panel_current'] = st.selectbox("Panel Current Column", data.columns)
                
                with col3:
                    column_mapping['datetime'] = st.selectbox("DateTime Column (optional)", 
                                                            ['None'] + list(data.columns))
                    column_mapping['temperature'] = st.selectbox("Temperature Column (optional)", 
                                                               ['None'] + list(data.columns))
                
                if st.button("Process CSV Data"):
                    # Rename columns according to mapping
                    processed_data = data.copy()
                    for new_col, old_col in column_mapping.items():
                        if old_col != 'None' and old_col in data.columns:
                            processed_data[new_col] = data[old_col]
                    
                    # Add calculated power if not present
                    if 'panel_power' not in processed_data.columns:
                        processed_data['panel_power'] = (processed_data['panel_voltage'] * 
                                                       processed_data['panel_current'])
                    
                    # Add datetime if not present
                    if 'datetime' not in processed_data.columns or column_mapping['datetime'] == 'None':
                        processed_data['datetime'] = pd.date_range(start='2024-01-01', 
                                                                  periods=len(processed_data), 
                                                                  freq='H')
                    
                    data = processed_data
                    st.session_state['analysis_data'] = data
                    st.success("CSV data processed successfully!")
                    
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
    
    # Perform analysis if data is available
    if 'analysis_data' in st.session_state:
        data = st.session_state['analysis_data']
        
        if st.button("🔍 Analyze Performance", type="primary"):
            analyze_performance(data)

def analyze_performance(data):
    """Enhanced performance analysis function"""
    
    # Initialize anomaly detector
    detector = EnhancedAnomalyDetector(contamination=0.1)
    
    with st.spinner("Analyzing panel performance..."):
        # Detect anomalies
        analyzed_data = detector.detect_anomalies(data)
        
        # Analyze panel health
        panel_health = detector.analyze_panel_health(analyzed_data)
    
    # Display results
    st.success("Analysis completed!")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_panels = len(data['panel_id'].unique())
    total_anomalies = analyzed_data['is_anomaly'].sum()
    critical_panels = sum(1 for p in panel_health.values() if p['health_status'] == 'Critical')
    avg_output = data['energy_output'].mean()
    
    with col1:
        st.metric("Total Panels", total_panels)
    with col2:
        st.metric("Anomalies Detected", total_anomalies)
    with col3:
        st.metric("Critical Panels", critical_panels, delta=f"-{critical_panels}" if critical_panels > 0 else "0")
    with col4:
        st.metric("Avg Output (kWh)", f"{avg_output:.2f}")
    
    # Detailed Analysis Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🚨 Panel Health", "📈 Performance Trends", "💡 Recommendations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Anomaly Detection Results")
            fig = px.scatter(analyzed_data, x='datetime', y='energy_output',
                           color='is_anomaly', 
                           hover_data=['panel_id', 'panel_voltage', 'panel_current'],
                           title="Energy Output with Anomalies",
                           color_discrete_map={True: 'red', False: 'blue'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Panel Performance Distribution")
            panel_avg = analyzed_data.groupby('panel_id')['energy_output'].mean().reset_index()
            fig = px.box(analyzed_data, x='panel_id', y='energy_output',
                        title="Energy Output Distribution by Panel")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Panel Health Status")
        
        # Create health status dataframe
        health_df = pd.DataFrame.from_dict(panel_health, orient='index').reset_index()
        health_df = health_df.rename(columns={'index': 'panel_id'})
        health_df = health_df.sort_values('priority')
        
        # Color coding based on health status
        def color_health_status(val):
            if val == 'Critical':
                return 'background-color: #f8d7da'
            elif val == 'Poor':
                return 'background-color: #fff3cd'
            elif val == 'Fair':
                return 'background-color: #d1ecf1'
            else:
                return 'background-color: #d4edda'
        
        styled_df = health_df.style.applymap(color_health_status, subset=['health_status'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Health status distribution
        col1, col2 = st.columns(2)
        
        with col1:
            health_counts = health_df['health_status'].value_counts()
            fig = px.pie(values=health_counts.values, names=health_counts.index,
                        title="Panel Health Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(health_df, x='panel_id', y='anomaly_rate',
                        color='health_status', title="Anomaly Rate by Panel")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Performance Trends")
        
        # Time series analysis
        if 'datetime' in analyzed_data.columns:
            daily_performance = analyzed_data.groupby([
                analyzed_data['datetime'].dt.date, 'panel_id'
            ])['energy_output'].sum().reset_index()
            
            daily_performance['datetime'] = pd.to_datetime(daily_performance['datetime'])

            fig = px.line(daily_performance, x='datetime', y='energy_output',
                          color='panel_id', title="Daily Energy Output per Panel")
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Recommendations")

        st.markdown("""
        <div class="performance-good">
            ✅ <strong>Good Panels:</strong> Maintain current performance. Schedule periodic cleaning and basic maintenance.
        </div>
        <div class="performance-warning">
            ⚠️ <strong>Fair/Poor Panels:</strong> Review wiring, inspect for dirt/shading, and monitor more closely for potential degradation.
        </div>
        <div class="performance-danger">
            ❌ <strong>Critical Panels:</strong> Immediate technical inspection recommended. Consider replacing or repairing components.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Panel-Specific Suggestions")
        for panel_id, info in panel_health.items():
            if info['health_status'] == "Critical":
                st.error(f"🔧 **{panel_id}** (Critical) – High anomaly rate ({info['anomaly_rate']:.2f}%). Urgent inspection needed.")
            elif info['health_status'] == "Poor":
                st.warning(f"⚠️ **{panel_id}** (Poor) – Moderate anomaly rate ({info['anomaly_rate']:.2f}%). Check for partial faults or dirt.")
            elif info['health_status'] == "Fair":
                st.info(f"ℹ️ **{panel_id}** (Fair) – Slight anomaly rate. Recommend routine inspection.")
            else:
                st.success(f"✅ **{panel_id}** (Good) – Operating normally. No action needed.")


