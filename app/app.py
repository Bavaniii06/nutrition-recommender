import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

import os

import os

DATA_PATH = os.path.join("data", "processed", "foods_clean.csv")
df = pd.read_csv(DATA_PATH)


# Function to calculate daily calories
def daily_calorie_need(sex, weight_kg, height_cm, age, activity_factor=1.2):
    if sex.lower() in ['male','m']:
        bmr = 88.362 + 13.397*weight_kg + 4.799*height_cm - 5.677*age
    else:
        bmr = 447.593 + 9.247*weight_kg + 3.098*height_cm - 4.330*age
    return bmr * activity_factor

# Streamlit UI
st.title("🍎 Nutrition Recommender")

sex = st.selectbox("Sex", ["Female", "Male"])
age = st.number_input("Age", 10, 100, 25)
weight = st.number_input("Weight (kg)", 30.0, 150.0, 55.0)
height = st.number_input("Height (cm)", 120.0, 220.0, 160.0)
activity = st.selectbox("Activity level", ["Sedentary", "Light", "Moderate", "Active"])

activity_map = {"Sedentary":1.2, "Light":1.375, "Moderate":1.55, "Active":1.725}

calories_needed = daily_calorie_need(sex, weight, height, age, activity_map[activity])
st.write(f"Estimated daily calories needed: **{int(calories_needed)} kcal**")

# Button to get recommendations
if st.button("Get Meal Recommendations"):
    protein_target = calories_needed * 0.15 / 4
    fat_target = calories_needed * 0.25 / 9
    carbs_target = calories_needed * 0.60 / 4

    target = np.array([[calories_needed, protein_target, fat_target, carbs_target]])
    features = df[['Calories', 'protein_g', 'fat_g', 'carbs_g']].fillna(0).to_numpy()
    sims = cosine_similarity(target, features)[0]

    df['similarity'] = sims
    top_recs = df.sort_values('similarity', ascending=False).head(10)

    # Show table
    st.table(top_recs[['food_name', 'Calories', 'protein_g', 'fat_g', 'carbs_g']])

    # Show pie chart
    if not top_recs.empty:
        macros = top_recs[['food_name', 'protein_g', 'fat_g', 'carbs_g']].set_index('food_name')
        macros['protein_cal'] = macros['protein_g']*4
        macros['fat_cal'] = macros['fat_g']*9
        macros['carbs_cal'] = macros['carbs_g']*4
        macros['total_cal'] = macros['protein_cal'] + macros['fat_cal'] + macros['carbs_cal']

        fig = px.pie(macros, names=macros.index, values='total_cal', title='Calories distribution in top foods')
        st.plotly_chart(fig)
