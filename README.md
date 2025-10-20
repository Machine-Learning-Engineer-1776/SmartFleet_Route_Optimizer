# 🚛 SmartFleet_Route_Optimizer
http://35.89.230.31:8501/

**Project Overview:**

SmartFleet Route Optimizer is a cutting-edge, AI-powered web application designed to optimize fleet routing for taxi and transportation services in Chicago. By integrating diverse data sources—such as crime statistics, weather forecasts, taxi trip records, ADAS vision sensor data, Terra-D2 gyro sensor readings, and news sentiment analysis—the application generates dynamic, risk-minimized routes to enhance safety and operational efficiency.

The application leverages advanced machine learning models for real-time predictions, including surge pricing, crime risk, weather impacts, sentiment analysis, and sensor degradation. This enables fleet operators to reduce operational costs by up to 23.5% and lower collision risks through optimized routing. Deployed at http://35.89.230.31:8501/, the application is fully mobile-responsive, making it a scalable solution for urban fleet management.

**Key Highlights:**

  •	**AI-Driven Risk Modeling:** Combines environmental, temporal, and sensor data into composite risk scores using supervised machine learning models.

  •	**Real-Time Predictions:** Interactive controls enable scenario simulation with AI-based     predictions for crime, weather, and sentiment.

  •	**Significant Business Impact:** The What-If Cost Estimator projects substantial savings, including up to $27,000 annually in sensor maintenance and $5.475M in crime avoidance for a 1,500-vehicle fleet.

  •	**End-to-End AI Engineering:** Demonstrates comprehensive AI pipeline development, from data preprocessing to real-time inference and visualization, suitable for applications in logistics, urban planning, and beyond.

<img width="1517" height="917" alt="{B94413A1-BDB9-45FD-908D-35D1212CB943}" src="https://github.com/user-attachments/assets/749085b6-c912-4f66-98db-e2b35a274618" />



**Features:**

  •	**How It Works Section:** Introduces the application’s core components with clear, concise descriptions to guide users through its functionality.

  •	**Interactive Control Panel:** Allows users to adjust temporal (day, hour, month),   environmental (temperature, precipitation, humidity, visibility), economic (surge multiplier, demand density), and safety (crime threshold, sensor influence) parameters to simulate route scenarios.

  •	**Advanced Sensor Intelligence:** Monitors ADAS vision and Terra-D2 gyro sensor health, providing failure rates and predictive maintenance insights based on environmental conditions.

  •	**Route Optimization Simulation:** Uses a probabilistic algorithm to generate low-risk routes, visualized with Matplotlib, reducing collision risks by up to 18%.

  •	**Chicago Risk Heatmap:** Displays an interactive Folium map with crime density, traffic overlays, and optimized routes for geospatial risk analysis.

  •	**Executive Dashboard:** Presents key performance indicators (KPIs) for fleet activity, risk reduction, and cost savings, with visualizations for strategic oversight.

•	**Enhanced What-If Cost Estimator:** Calculates detailed annual savings based on user-defined fleet parameters, with a comprehensive breakdown of cost reductions from sensor maintenance, crime avoidance, sentiment-based avoidance, fuel efficiency, and operational optimizations.

•	**Mobile Responsiveness:** Custom CSS ensures seamless usability across desktop and mobile devices.

•	**Performance Optimization:** Streamlit caching enhances efficiency for large datasets.



------------------------------------------------------------------------------------------------
**🤖 Machine Learning Models:**

The application integrates multiple machine learning models to process and predict from diverse data sources, ensuring robust and actionable insights:

  •	**Crime Prediction Model:** A supervised classification model (e.g., Random Forest) trained on historical crime data to predict crime probabilities by zone and hour. Features are engineered using pivot-based methods to identify violent crime hotspots with high accuracy.

  •	**Weather Prediction Model:** A regression model (e.g., linear regression) forecasting weather risks based on temperature (>80°F) and precipitation (>0.1 inches), enhanced with synthetic data generation for robust risk assessment.

  •	**News Sentiment LLM:** A fine-tuned large language model (e.g., BERT via Hugging Face) classifying news articles into sentiment scores, integrated to capture external event impacts (e.g., protests, road closures) on routing decisions.

  •	**Sensor Degradation Model:** A quadratic regression model predicting ADAS vision sensor failure rates based on humidity, combined with weighted risk scoring for Terra-D2 gyro sensors, projecting up to $27,000 in annual maintenance savings.

  •	**Composite Risk Model:** An ensemble approach aggregating predictions from crime, weather, sentiment, and sensor models, using Scikit-learn for feature scaling and NumPy for probabilistic route optimization, achieving an average cost reduction of 21%.

  •	**Surge Pricing Model:** A rule-based model enhanced with machine learning to compute dynamic surge multipliers, incorporating temporal and demand features adjusted by risk penalties.
  
These models are built using **PyTorch**, **Scikit-learn**, and **NumPy**, with **MLflow** for experiment tracking to ensure reproducibility and scalability.

------------------------------------------------------------------------------------------------

**Accessing the Application:**

  •	**Live Demo:** Access the application at **http://35.89.230.31:8501/**.

  •	**Interaction Guide:**
  
  1.	Navigate to the **How It Works** section to understand the application’s components.

  2.	Use the **Control Panel** to adjust parameters (e.g., hour, temperature, crime threshold).

  3.	Explore sensor health in the **Sensor Performance Analysis** section.

  4.	Simulate routes and view the heatmap in the **Route Optimization** and **Chicago Risk Heatmap** sections.

  5.	Review KPIs in the **Executive Dashboard**.

  6.	Calculate savings in the **What-If Cost Estimator** to see detailed cost breakdowns.

------------------------------------------------------------------------------------------------


**📊 Data Sources:**

•	**Chicago Crime Data:** Sampled CSV from the Chicago Data Portal for crime prediction modeling.

•	**Chicago Weather Data:** Historical weather metrics for weather risk assessment.

•	**Chicago Taxi Data:** Sampled Excel file from taxi trip records for surge pricing and demand modeling.

•	**ADAS-EV Dataset:** Sensor data for vision-based obstacle detection and risk weighting.

•	**Terra-D2 Dataset:** Gyro sensor readings for stability risk analysis and degradation modeling.

•	**News Sentiment Analysis:** CSV for LLM-based sentiment classification and risk factor integration.

All data is preprocessed for privacy and efficiency. Full datasets are available on the Chicago Data Portal and Kaggle.


------------------------------------------------------------------------------------------------


**Technologies Used:**

•	**Frontend:** Streamlit for an interactive, real-time dashboard.

•	**Data Processing and Machine Learning:** Pandas, NumPy, Scikit-learn for preprocessing, feature engineering, and risk modeling.

•	**Machine Learning Models:** PyTorch for potential graph neural network extensions, Hugging Face for LLM sentiment analysis, and regression/classification models for predictions.

•	**Visualization:** Matplotlib for charts and Folium/Streamlit-Folium for geospatial maps.

•	**Additional Tools:** Custom CSS for mobile responsiveness, MLflow for experiment tracking, and Streamlit caching for performance optimization.


------------------------------------------------------------------------------------------------


**Breakdown of Each Application Section:**
The SmartFleet Route Optimizer is organized into modular sections, each leveraging AI to deliver specific functionality for fleet management. Below is a detailed breakdown of each section, highlighting its purpose, AI integration, and business value.

**1. Main Header**

•	**Purpose:** Introduces the application as a real-time, AI-powered route optimization tool for safer and more efficient fleet operations.

•	**AI Integration:** Provides context for AI-driven features without direct model integration.

•	**Value:** Positions the application as a critical tool for operational efficiency, appealing to stakeholders by bridging technical AI capabilities with business outcomes.

**2. How It Works**

•	**Purpose:** Guides users through the application’s key components with concise descriptions, making it easy to understand its functionality and value.

•	**AI Integration:** None directly; sets the stage for AI-driven sections.

•	**Value:** Enhances user onboarding, ensuring accessibility for both technical and non-technical users, including executives and fleet managers.

**3. Control Panel - Environmental & Temporal Factors**

•	**Purpose:** Allows users to adjust variables such as day, hour, month, temperature, precipitation, wind, humidity, visibility, demand density, surge multiplier, maintenance impact, crime threshold, and sensor influence weights to simulate routing scenarios.

•	**AI Integration:** Inputs feed into AI models for crime prediction, weather forecasting, sentiment analysis, and sensor risk scoring, enabling dynamic route optimization.

•	**Value:** Demonstrates the application’s flexibility in adapting to real-world conditions, showcasing real-time AI inference for scenario planning. The Crime Risk Threshold slider, with its note ("SmartFleet recommends setting the Crime Risk Threshold to 1.0 (maximum) for optimal safety and cost savings. Adjust the slider to explore route modifications and savings driven by our AI-powered crime prediction model."), has transformed user engagement, making it a standout feature for visualizing safety and cost impacts.

<img width="1105" height="365" alt="{8889C828-816E-4228-9A4C-7A9DA054241E}" src="https://github.com/user-attachments/assets/8183b73b-fdff-4f41-a61c-23f3a4d59600" />


**4. Sensor Performance Analysis**

•	**Purpose:** Monitors the health of ADAS vision and Terra-D2 gyro sensors, displaying failure rates and remaining sensor life based on environmental conditions like humidity.

•	**AI Integration:** Employs a quadratic regression model to predict sensor degradation, with visualizations generated using Matplotlib and NumPy-based risk calculations.

•	**Value:** Projects up to $27,000 in annual maintenance savings by enabling predictive maintenance, reducing downtime and costs. This section highlights AI-driven IoT analytics for fleet management.

<img width="1041" height="745" alt="{D8AB9F77-807D-4434-8864-00EB036CA6AC}" src="https://github.com/user-attachments/assets/d5d0d77e-3e99-4e24-8b98-c2706c1b2c7d" />


**5. Route Optimization**

•	**Purpose:** Simulates and visualizes low-risk routes through five Chicago zones, avoiding high-risk areas such as crime hotspots.

•	**AI Integration:** Uses a probabilistic NumPy-based algorithm aggregating predictions from crime, weather, sentiment, and sensor models to compute a total risk score, visualized with Matplotlib.

•	**Value:** Reduces simulated collision risks by up to 18%, demonstrating the power of AI in optimizing routes for safety and efficiency.

<img width="1081" height="725" alt="{E2E9AA24-CA77-434A-959B-CB805BC53CB2}" src="https://github.com/user-attachments/assets/917b6b74-0ee4-4b32-9553-25e4012ca2ff" />


**6. Chicago Interactive Map**

•	**Purpose:** Provides a geospatial visualization of crime density and optimized routes across Chicago.

•	**AI Integration:** Renders heatmaps using Folium, driven by AI-predicted crime probabilities and traffic risks, with route markers reflecting sensor-based risk scores.

•	**Value:** Offers intuitive geospatial analysis, enabling fleet managers to visualize and act on AI-driven risk insights.



**7. Executive Dashboard**

•	**Purpose:** Summarizes fleet-wide performance through KPIs, including active vehicles, daily route savings, risk reduction, and sensor ROI, supported by visualizations.

•	**AI Integration:** Derives metrics (e.g., risk reduction percentage) from AI simulation outputs, visualized with Matplotlib pie, bar, and line charts.

•	**Value:** Translates AI predictions into actionable business metrics, enabling strategic decision-making with clear ROI insights (e.g., up to 95% risk reduction).

<img width="973" height="589" alt="{E9AC852F-5557-49E6-88F7-D7D20817B325}" src="https://github.com/user-attachments/assets/4703930b-93bf-4bbc-9c3f-0e5b18979252" />


**8. What-If Cost Estimator**

•	**Purpose:** Calculates projected annual fleet savings based on user inputs for fleet size, trip frequency, distance, and cost, with a detailed breakdown of savings sources.

•	**AI Integration:** Incorporates AI-driven predictions (e.g., crime avoidance percentages) into arithmetic models, projecting up to 23.5% cost reductions for a 1,500-vehicle fleet. The breakdown includes:
  o	**Sensor Maintenance:** Up to $27,000 in savings from predictive maintenance.
  o	**Crime Avoidance:** Up to $5.475M by avoiding high-crime zones.
  o	**Sentiment-Based Avoidance:** $1.095M by avoiding negative sentiment zones.
  o	**Fuel Efficiency:** $3.454M from optimized routes and reduced idling.
  o	**Other Operational:** Up to $9.617M from demand-based routing and surge pricing optimizations.

•	**Value:** Provides a compelling financial justification for adopting SmartFleet, with detailed, transparent savings calculations that have been refined to maximize clarity and impact for sales teams and potential buyers.

<img width="977" height="793" alt="{0DDEB691-E78D-4C7B-AF00-EEC8EBD9E328}" src="https://github.com/user-attachments/assets/9d64fa87-b76d-4ea6-aa77-3da28531cf89" />

