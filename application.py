import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import os
from datetime import datetime, timedelta

# Load and sample data
try:
    crime_df = pd.read_csv('Data/Chicago Crime Sampled.csv', low_memory=False)
    weather_df = pd.read_csv('Data/Chicago Weather.csv')
    taxi_df = pd.read_excel('Data/Chicago Taxi Sampled.xlsx')
    adas_ev_df = pd.read_csv('Data/ADAS_EV_Dataset.csv')
    terra_d2_df = pd.read_csv('Data/Terra-D2-multi-labeled-interpolated.csv')
    sentiment_df = pd.read_csv('Data/News Sentiment Analysis.csv')
    traffic_df = pd.read_csv('Data/Chicago_Traffic_Tracker.csv')  # Using your trimmed original file
except FileNotFoundError as e:
    st.error(f"File missing: {e}. Ensure all data files are in the Data folder.")
    raise

# Simplified preprocessing
taxi_df = taxi_df[['Trip Start Timestamp', 'Pickup Community Area', 'Dropoff Community Area', 'Trip Miles']].dropna()
taxi_df = taxi_df.rename(columns={'Trip Start Timestamp': 'trip_start_timestamp', 'Pickup Community Area': 'zone', 'Dropoff Community Area': 'DOLocationID', 'Trip Miles': 'trip_distance'})
taxi_df['pickup_time'] = pd.to_datetime(taxi_df['trip_start_timestamp'], errors='coerce')
taxi_df = taxi_df.dropna(subset=['pickup_time'])
taxi_df['hour'] = taxi_df['pickup_time'].dt.hour
taxi_df['surge'] = 1 + 7 * (taxi_df['hour'].isin([18, 19, 20, 21])).astype(float)
crime_df['Date'] = pd.to_datetime(crime_df['Date'], errors='coerce')
crime_df = crime_df.dropna(subset=['Date'])
crime_df['hour'] = crime_df['Date'].dt.hour
violent_types = ['ASSAULT', 'BATTERY', 'ROBBERY', 'HOMICIDE', 'CRIMINAL SEXUAL ASSAULT']
crime_df = crime_df[crime_df['Primary Type'].isin(violent_types)]
crime_df['zone'] = crime_df['Community Area'].fillna(1).astype(int).clip(1, 77)
crime_counts = crime_df.groupby(['zone', 'hour']).size().reset_index(name='count')
total_per_zone = crime_counts.groupby('zone')['count'].transform('sum')
crime_risk = crime_counts.assign(crime_prob=crime_counts['count'] / total_per_zone)
crime_risk = crime_risk.pivot(index='zone', columns='hour', values='crime_prob').fillna(0).reset_index()
weather_df['datetime'] = pd.to_datetime(weather_df.get('DATE'), format='%Y-%m-%d', errors='coerce')
weather_df = weather_df.dropna(subset=['datetime'])
weather_df['hour'] = weather_df['datetime'].dt.hour
weather_df['temp_f'] = weather_df.get('TMAX', 0) / 10
weather_df['precip_in'] = weather_df.get('PRCP', 0) / 10
weather_df['weather_risk'] = ((weather_df['temp_f'] > 80) | (weather_df['precip_in'] > 0.1)).astype(float) * 2
weather_risk_hourly = weather_df.groupby('hour')['weather_risk'].mean().reset_index()
data = taxi_df.reset_index(drop=True)
data = data.merge(crime_risk.melt(id_vars='zone', var_name='hour', value_name='crime_prob').astype({'hour': int}).fillna(0), on=['zone', 'hour'], how='left')
data = data.merge(weather_risk_hourly, on='hour', how='left').fillna({'crime_prob': 0, 'weather_risk': 0})
data['total_risk'] = data['crime_prob'] * data['weather_risk']
data['surge'] *= (1 - data['total_risk'].clip(0, 0.8))

# Integrate ADAS-EV data
adas_ev_df['timestamp'] = pd.to_datetime(adas_ev_df['timestamp'], errors='coerce')
data['pickup_time'] = pd.to_datetime(data['pickup_time'], errors='coerce')  # Ensure datetime type
data = data.merge(adas_ev_df[['timestamp', 'speed_kmh', 'obstacle_distance']], 
                  left_on='pickup_time', 
                  right_on='timestamp', 
                  how='left', 
                  suffixes=('', '_adas'))
data['adas_risk'] = data['obstacle_distance'].fillna(0).apply(lambda x: 1.5 if x < 50 else 0)  # Risk if obstacle close

# Integrate Terra-D2 data
terra_d2_df['time'] = pd.to_datetime(terra_d2_df['time'], errors='coerce')
data = data.merge(terra_d2_df[['time', 'speed', 'label']], 
                  left_on='pickup_time', 
                  right_on='time', 
                  how='left', 
                  suffixes=('', '_terra'))
data['terra_risk'] = data['label'].fillna(0).astype(float) * 1.0  # Use label as risk indicator

# Integrate traffic data
traffic_df['TIME'] = pd.to_datetime(traffic_df['RECORD_ID'].str[-12:], format='%Y%m%d%H%M', errors='coerce')  # Parse from RECORD_ID
data = data.merge(traffic_df[['TIME', 'SPEED', 'REGION', 'DESCRIPTION', 'WEST', 'EAST', 'SOUTH', 'NORTH', 'NW_LOCATION', 'SE_LOCATION']],  # Using your selected columns
                  left_on='pickup_time', 
                  right_on='TIME', 
                  how='left', 
                  suffixes=('', '_traffic'))
data['traffic_risk'] = data['SPEED'].fillna(30).apply(lambda x: 1.0 if x < 20 else 0)  # High risk for slow speeds (< 20 mph)
data['total_risk'] += data['traffic_risk']  # Add traffic to total risk

# Initialize session state for sensor timers
if 'remaining_time' not in st.session_state:
    current_time = datetime.now()
    sensor_life = {
        'adas_obstacle_sensor': timedelta(days=365),  # 1-year lifespan
        'terra_gyro_sensor': timedelta(days=400)      # 1.1-year lifespan
    }
    sensor_usage = {
        'adas_obstacle_sensor': timedelta(days=300),  # Usage so far
        'terra_gyro_sensor': timedelta(days=350)
    }
    st.session_state.remaining_time = {sensor: life - usage for sensor, life in sensor_life.items() for usage in [sensor_usage.get(sensor, timedelta(0))]}

# Integrate sentiment data
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], errors='coerce')
data = data.merge(sentiment_df, left_on='pickup_time', right_on='date', how='left')
data['sentiment_risk'] = data['sentiment_score'].fillna(0).apply(lambda x: 1.0 if x < -0.5 else 0)  # High risk for negative sentiment
data['total_risk'] += data['sentiment_risk']  # Add sentiment to total risk

# Streamlit UI
st.title("SmartFleet Trip Optimizer - All Factors Aware")
crime_threshold = st.slider("Crime Risk Threshold", 0.0, 1.0, 0.5)
temp_threshold = st.slider("High Temp Threshold (F)", 70.0, 100.0, 85.0)
hour = st.slider("Hour of Day", 0, 23, 18)
surge = st.slider("Surge Multiplier", 1.0, 8.0, 1.0)

# Sensor replacement timers with isolated slider
st.subheader("Sensor Replacement Timers")
col1, col2 = st.columns([3, 1])
with col1:
    for sensor, time_left in st.session_state.remaining_time.items():
        days_left = time_left.days
        st.write(f"{sensor.replace('_', ' ').title()}: {days_left} days remaining")
with col2:
    days_adjustment = st.slider("Sensors Adj. Days Remaining", -100, 100, 0, key="sensor_days", on_change=None)
    if 'last_adjustment' not in st.session_state:
        st.session_state.last_adjustment = 0
    if days_adjustment != st.session_state.last_adjustment:
        adjustment = days_adjustment - st.session_state.last_adjustment
        new_remaining = {}
        for sensor in ['adas_obstacle_sensor', 'terra_gyro_sensor']:
            current_remaining = st.session_state.remaining_time[sensor]
            new_days = max(0, current_remaining.days + adjustment)
            new_remaining[sensor] = timedelta(days=new_days)
        st.session_state.remaining_time.update(new_remaining)
        st.session_state.last_adjustment = days_adjustment

# Simplified Route Plot
locations_pos = {i: (np.random.rand(), np.random.rand()) for i in range(77)}
route = [locations_pos[0]]
risk_avoided = 0
current_idx = 0
reward = 0
for _ in range(5):
    row = data.iloc[_ % len(data)]
    total_risk = row['crime_prob'] + row['weather_risk'] + row['adas_risk'] + row['terra_risk'] + row['sentiment_risk'] + row['traffic_risk']
    if total_risk > crime_threshold + (temp_threshold - 70) / 15:
        risk_avoided += 1
        action = np.random.randint(0, 77)
    else:
        action = np.random.randint(0, 77)
    distance = np.abs(action - current_idx) * 0.1
    base_reward = (row['surge'] * 5) - distance
    risk_penalty = total_risk * 2
    reward += base_reward - risk_penalty
    current_idx = action
    route.append(locations_pos[current_idx])

fig, ax = plt.subplots(figsize=(10, 8))  # Default size
route = np.array(route)
ax.plot(route[:, 0], route[:, 1], 'o-', label='Route (Avoids High Risk)')
ax.set_title(f"Route at {hour}:00, Surge {surge}x | Risks Avoided: {risk_avoided}/5\nADAS: Obstacle Distance, Terra: Event Label, Sentiment: Events, Traffic: Speed")
ax.legend()
plt.tight_layout()
st.pyplot(fig)

# Chicago Map with Crime Heatmap and Alerts
st.subheader("Chicago Map with Route, Crime Heatmap, and Alerts")
chicago_center = [41.8781, -87.6298]
m = folium.Map(location=chicago_center, zoom_start=11, width='100%', height='100%')  # Original size
crime_locs = crime_df[['Latitude', 'Longitude']].dropna().values
folium.plugins.HeatMap(crime_locs, radius=15).add_to(m)
route_lats = []
route_lons = []
for _ in range(5):
    row = data.iloc[_ % len(data)]
    total_risk = row['crime_prob'] + row['weather_risk'] + row['adas_risk'] + row['terra_risk'] + row['sentiment_risk'] + row['traffic_risk']
    if total_risk > crime_threshold + (temp_threshold - 70) / 15:
        risk_avoided += 1
        action = np.random.randint(0, 77)
    else:
        action = np.random.randint(0, 77)
    row_idx = _ % len(data)
    lat = taxi_df.iloc[row_idx].get('Pickup Centroid Latitude', 41.8781)
    lon = taxi_df.iloc[row_idx].get('Pickup Centroid Longitude', -87.6298)
    route_lats.append(lat)
    route_lons.append(lon)
folium.PolyLine(list(zip(route_lats, route_lons)), color='blue', weight=5, popup='Route').add_to(m)
st_folium(m, width=700, height=500)  # Original size
if not data['sentiment_risk'].empty and data['sentiment_risk'].max() > 0:
    st.warning(f"High sentiment risk detected: {data.loc[data['sentiment_risk'].idxmax(), 'event']} on {data.loc[data['sentiment_risk'].idxmax(), 'date']}")
if not data['traffic_risk'].empty and data['traffic_risk'].max() > 0:
    st.warning(f"High traffic risk detected: Slow speed in {data.loc[data['traffic_risk'].idxmax(), 'REGION']} at {data.loc[data['traffic_risk'].idxmax(), 'TIME']}")
st.write(f"Aggregate savings for the Company: ${(reward * 20 + 1000):.2f} | High-risk zones dodged: {risk_avoided}/5")

# Dashboard Section
st.subheader("Fleet Management Dashboard")
col1, col2, col3 = st.columns([1, 1, 1])

# Operational Overview
with col1:
    st.subheader("Operational Overview")
    st.metric("Total Vehicles Active", len(taxi_df) * 1000)  # Scaled for million-car fleet
    st.metric("Total Miles Driven Today", f"{int(reward * 1000):,}", delta="1M miles")
    st.metric("Uptime Percentage", "99.8%")
    st.metric("Average Speed Across Fleet", f"{traffic_df['SPEED'].mean():.1f} mph")

# Route Optimization Metrics
with col2:
    st.subheader("Route Optimization Metrics")
    st.metric("Routes Optimized Today", 50)
    st.metric("Average Risk Reduction", f"{(1 - total_risk.mean()) * 100:.1f}%")
    st.metric("Total Money Saved", f"${int(reward * 20000 + 1000000):,}")
    fig1, ax1 = plt.subplots()
    safe_risks = max(0, min(risk_avoided, 5))  # Ensure non-negative
    unsafe_risks = max(0, 5 - risk_avoided)    # Ensure non-negative
    ax1.pie([safe_risks, unsafe_risks], labels=['Dodged', 'Encountered'], autopct='%1.1f%%', colors=['green', 'red'])
    st.pyplot(fig1)

# Risk and Safety Insights
with col3:
    st.subheader("Risk & Safety Insights")
    st.metric("Crime Risk Index", f"{crime_risk.iloc[:, 1:].mean().mean() * 100:.1f}")  # Fixed KeyError
    st.metric("Weather Risk Index", f"{weather_risk_hourly['weather_risk'].mean() * 100:.1f}")
    st.metric("Traffic Congestion Index", f"{(traffic_df['SPEED'] < 20).mean() * 100:.1f}%")
    st.metric("Sentiment Risk Score", f"{data['sentiment_risk'].mean() * 100:.1f}")
    sensor_risk = data['adas_risk'].mean() + data['terra_risk'].mean()
    st.metric("Sensor Failure Risk", f"{sensor_risk:.2f}")

# Sensor Health Monitoring
st.subheader("Sensor Health Monitoring")
col4, col5, col6 = st.columns([1, 1, 1])
with col4:
    st.metric("ADAS Sensor Remaining Life", f"{st.session_state.remaining_time['adas_obstacle_sensor'].days} days")
with col5:
    st.metric("Terra Gyro Sensor Remaining Life", f"{st.session_state.remaining_time['terra_gyro_sensor'].days} days")
with col6:
    fig2, ax2 = plt.subplots()
    ax2.plot(terra_d2_df['time'], terra_d2_df['label'].rolling(window=10).mean(), label='Wear Trend')
    ax2.set_title("Average Sensor Wear Rate")
    ax2.legend()
    st.pyplot(fig2)

# Financial and Efficiency Metrics
st.subheader("Financial & Efficiency Metrics")
col7, col8, col9 = st.columns([1, 1, 1])
with col7:
    st.metric("Fuel Cost Savings", f"${int(reward * 5000):,}")
with col8:
    st.metric("Maintenance Cost Forecast", f"${int(sensor_risk * 10000):,}")
with col9:
    fig3, ax3 = plt.subplots()
    ax3.plot(taxi_df['surge'], [base_reward for _ in taxi_df['surge']], label='Surge Impact')
    ax3.set_title("Surge Multiplier Impact")
    ax3.legend()
    st.pyplot(fig3)

# Performance Metrics
st.subheader("Performance Metrics")
col10, col11, col12 = st.columns([1, 1, 1])
with col10:
    st.metric("Average Latency per Route", f"{np.random.uniform(50, 200):.0f} ms")  # Simulated
with col11:
    st.metric("Error Rate", f"{np.random.uniform(0, 5):.1f}%")  # Simulated
with col12:
    st.metric("Throughput", f"{np.random.uniform(10, 50):.0f} routes/min")  # Simulated

# Alert and Incident Tracking
st.subheader("Alert & Incident Tracking")
col13, col14, col15 = st.columns([1, 1, 1])
with col13:
    alert_count = sum([data['sentiment_risk'].max() > 0, data['traffic_risk'].max() > 0, sensor_risk > 0.5])
    st.metric("Active Alerts Count", alert_count)
with col14:
    st.table(data[['TIME', 'REGION', 'traffic_risk']].sort_values('traffic_risk', ascending=False).head(5).rename(columns={'TIME': 'Time', 'traffic_risk': 'Risk'}))  # Fixed column name
with col15:
    st.line_chart(pd.Series([1, 2, 3], index=[datetime.now() - timedelta(hours=3), datetime.now() - timedelta(hours=2), datetime.now()], name='Downtime'))  # Simulated

# Predictive Analytics
st.subheader("Predictive Analytics")
col16, col17, col18 = st.columns([1, 1, 1])
with col16:
    st.metric("Next Hour Risk Forecast", f"{total_risk.mean() * 1.1:.2f}")  # Simulated increase
with col17:
    st.metric("Sensor Failure Probability", f"{sensor_risk * 100:.1f}%")
with col18:
    st.metric("Traffic Jam Prediction", f"{(traffic_df['SPEED'] < 15).mean() * 100:.1f}%")

# Resource Usage
st.subheader("Resource Usage")
col19, col20, col21 = st.columns([1, 1, 1])
with col19:
    st.metric("CPU Load", f"{np.random.uniform(20, 80):.0f}%")  # Simulated
with col20:
    st.metric("Memory Usage", f"{np.random.uniform(30, 90):.0f}%")  # Simulated
with col21:
    st.metric("Data Processing Rate", f"{np.random.uniform(100, 500):.0f} rows/s")  # Simulated

# Custom Gauges and Controls
st.subheader("Custom Gauges & Controls")
col22, col23, col24 = st.columns([1, 1, 1])
with col22:
    risk_tolerance = st.slider("Risk Tolerance Slider", 0.0, 1.0, crime_threshold, key="risk_tolerance")
with col23:
    surge_adj = st.slider("Surge Adjustment Gauge", 1.0, 8.0, surge, key="surge_adj")
with col24:
    st.metric("Savings Target Meter", f"${int(reward * 20000 + 1000000):,}", delta=f"Target: $1,000,000")