AI-Driven Road Accident Prediction and Prevention

This project leverages machine learning to predict the severity of road accidents using large-scale datasets on accidents, vehicles, and casualties. The goal is to assist traffic authorities and city planners in identifying high-risk areas, improving road safety infrastructure, and deploying preventive measures to reduce severe accidents.

1. Project Structure
AI_Road_Accident_Prediction/
│
├── AccidentsBig.csv
├── CasualtiesBig.csv
├── VehiclesBig.csv
├── road_accident_prediction.ipynb
├── requirements.txt


2. Objective

To build an AI-driven predictive model that:

Classifies the severity of road accidents (slight, serious, fatal)

Identifies key factors influencing accident outcomes (weather, road type, lighting, etc.)

Provides insights for proactive prevention and policy making

3. Approach

Data Integration

Combined three datasets (AccidentsBig.csv, CasualtiesBig.csv, and VehiclesBig.csv) using the Accident_Index key.

Aggregated vehicle and casualty counts per accident to enrich the main accident dataset.

Data Cleaning & Preprocessing

Handled missing values and categorical encoding.

Removed irrelevant or high-cardinality columns.

Scaled numerical features for model training.

Exploratory Data Analysis (EDA)

Analyzed severity distribution.

Explored correlations between features such as weather, lighting, and road type.

Modeling

Built multiple ML models:

Random Forest Classifier (baseline)

LightGBM Classifier (optimized gradient boosting)

Used accuracy, F1-score, and confusion matrix for evaluation.

Feature Importance

Extracted and visualized key predictors influencing accident severity.

Prevention Insights

Highlighted actionable insights for improving road safety and accident prevention strategies.

4. Key Features
Feature	Description
Speed_limit	Speed limit at accident location
Road_Type	Type of road (single carriageway, dual, etc.)
Weather_Conditions	Weather during the accident
Light_Conditions	Lighting conditions (daylight, darkness, etc.)
Urban_or_Rural_Area	Urban vs rural classification
Number_of_Vehicles	Vehicles involved in the accident
Number_of_Casualties	Number of casualties reported
🧩 Model Performance (Example)
Model	Accuracy	F1-score (Weighted)
Random Forest	0.86	0.81
LightGBM	0.89	0.84

(Results may vary depending on training subset and preprocessing choices.)

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/<your-username>/AI-Road-Accident-Prediction.git
cd AI-Road-Accident-Prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the notebook
jupyter notebook road_accident_prediction.ipynb

4️⃣  Run the Streamlit App
streamlit run road_accident_prediction_app.py

📈 Visualization

The notebook and app include:

Severity distribution plots

Feature importance bar charts

Correlation heatmaps

Interactive dashboards (Plotly & Streamlit)

5.Future Enhancements

Integrate real-time traffic and weather APIs

Build deep learning models for enhanced accuracy

Create geospatial risk heatmaps

Develop accident prevention recommendation systems

Deploy as a web-based early warning system

6. Real-World Applications

Traffic Police: Predict high-risk zones and times for patrol allocation

Urban Planners: Identify dangerous junctions needing redesign

Insurance Companies: Estimate accident risk for premium adjustments

Public Awareness: Provide data-driven road safety campaigns

7. Requirements

See requirements.txt
:

scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
streamlit==1.28.0
xgboost==2.0.2
plotly==5.18.0
matplotlib==3.8.2
lightgbm==4.2.2


