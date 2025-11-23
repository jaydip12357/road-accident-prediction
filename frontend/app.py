import streamlit as st
import requests
import json

# Configure page
st.set_page_config(
    page_title="Road Accident Severity Predictor",
    page_icon="🚗",
    layout="wide"
)

# API URL
API_URL = "https://road-accident-api-1028014290034.us-east1.run.app"

# Title and description
st.title("🚗 Road Accident Severity Predictor")
st.markdown("""
This application predicts the severity of road accidents based on various conditions.
Enter the details below to get a prediction.
""")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Road Conditions")
    speed_limit = st.selectbox(
        "Speed Limit (mph)",
        [20, 30, 40, 50, 60, 70],
        index=1
    )
    
    road_surface = st.selectbox(
        "Road Surface Conditions",
        ["Dry", "Wet or damp", "Snow", "Frost or ice", "Flood over 3cm. deep"]
    )
    
    urban_rural = st.selectbox(
        "Area Type",
        ["Urban", "Rural"]
    )

with col2:
    st.subheader("🌤️ Environment")
    light_conditions = st.selectbox(
        "Light Conditions",
        ["Daylight", "Darkness - lights lit", "Darkness - lights unlit", 
         "Darkness - no lighting", "Darkness - lighting unknown"]
    )
    
    weather = st.selectbox(
        "Weather Conditions",
        ["Fine no high winds", "Raining no high winds", "Snowing no high winds",
         "Fine + high winds", "Raining + high winds", "Snowing + high winds", "Fog or mist"]
    )

st.subheader("🚙 Accident Details")
col3, col4, col5 = st.columns(3)

with col3:
    num_vehicles = st.number_input(
        "Number of Vehicles",
        min_value=1,
        max_value=10,
        value=2
    )

with col4:
    num_casualties = st.number_input(
        "Number of Casualties",
        min_value=1,
        max_value=20,
        value=1
    )

with col5:
    hour = st.slider(
        "Hour of Day",
        min_value=0,
        max_value=23,
        value=17
    )

# Predict button
if st.button("🔮 Predict Severity", type="primary"):
    with st.spinner("Analyzing accident conditions..."):
        # Prepare data
        data = {
            "speed_limit": speed_limit,
            "number_of_vehicles": num_vehicles,
            "number_of_casualties": num_casualties,
            "hour": hour,
            "light_conditions": light_conditions,
            "weather_conditions": weather,
            "road_surface_conditions": road_surface,
            "urban_or_rural_area": urban_rural
        }
        
        try:
            # Make prediction
            response = requests.post(f"{API_URL}/predict", json=data)
            result = response.json()
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            # Show prediction with color coding
            prediction = result["prediction"]
            confidence = result["confidence"]
            
            if prediction == "Slight":
                st.success(f"### 🟢 Predicted Severity: {prediction}")
            elif prediction == "Serious":
                st.warning(f"### 🟡 Predicted Severity: {prediction}")
            else:
                st.error(f"### 🔴 Predicted Severity: {prediction}")
            
            # Show confidence
            st.metric("Confidence", f"{confidence*100:.1f}%")
            
            # Show probabilities
            st.subheader("📊 Probability Distribution")
            probs = result["probabilities"]
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Slight", f"{probs['Slight']*100:.1f}%")
            with col_b:
                st.metric("Serious", f"{probs['Serious']*100:.1f}%")
            with col_c:
                st.metric("Fatal", f"{probs['Fatal']*100:.1f}%")
            
            # Visualization
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(probs.keys()),
                    y=list(probs.values()),
                    marker_color=['green', 'orange', 'red']
                )
            ])
            fig.update_layout(
                title="Severity Probabilities",
                yaxis_title="Probability",
                xaxis_title="Severity Level",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**About:** This model uses logistic regression trained on UK Road Safety data (2005-2023).  
**Note:** Predictions are for educational purposes only.  
**GitHub:** [View Project](https://github.com/jaydip12357/road-accident-prediction)
""")
