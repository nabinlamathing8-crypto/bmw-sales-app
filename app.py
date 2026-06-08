import streamlit as st
import pandas as pd
import pickle

with open('bmw_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('bmw_features.pkl', 'rb') as f:
    features = pickle.load(f)

st.set_page_config(page_title="BMW Sales Predictor", page_icon="🚗")
st.title("🚗 BMW Sales Predictor")
st.markdown("Predict whether a BMW will have **High** or **Low** sales.")

st.sidebar.header("Car Details")

all_models   = sorted(["3 Series","5 Series","7 Series",
                        "M3","M5","X1","X3","X5","X6","i3","i8"])
all_regions  = sorted(["Asia","Europe","North America",
                        "South America","Middle East","Africa"])
all_fuels    = sorted(["Petrol","Diesel","Electric","Hybrid"])
all_trans    = sorted(["Manual","Automatic"])

model_name   = st.sidebar.selectbox("Model",        all_models)
region       = st.sidebar.selectbox("Region",       all_regions)
fuel         = st.sidebar.selectbox("Fuel Type",    all_fuels)
transmission = st.sidebar.selectbox("Transmission", all_trans)
year         = st.sidebar.slider("Year",            2010, 2024, 2020)
engine       = st.sidebar.slider("Engine Size (L)", 1.5, 5.0, 2.5)
mileage      = st.sidebar.slider("Mileage (KM)",    0, 200000, 50000,
                                  step=1000)
car_age = 2024 - year

row = {
    'Year':          year,
    'Engine_Size_L': engine,
    'Mileage_KM':    mileage,
    'Car_Age':       car_age,
    'Model':         all_models.index(model_name),
    'Region':        all_regions.index(region),
    'Fuel_Type':     all_fuels.index(fuel),
    'Transmission':  all_trans.index(transmission),
}

input_data = pd.DataFrame([row])

for col in features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[features]

c1, c2, c3 = st.columns(3)
c1.metric("Model",   model_name)
c2.metric("Year",    year)
c3.metric("Mileage", f"{mileage:,} KM")

c4, c5, c6 = st.columns(3)
c4.metric("Fuel",    fuel)
c5.metric("Engine",  f"{engine}L")
c6.metric("Car Age", f"{car_age} yrs")

st.divider()

if st.button("Predict Sales", type="primary"):
    try:
        pred  = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        if pred == 1:
            st.success("HIGH Sales predicted!")
            st.metric("Confidence", f"{proba[1]*100:.1f}%")
            st.balloons()
        else:
            st.warning("LOW Sales predicted")
            st.metric("Confidence", f"{proba[0]*100:.1f}%")
    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Features expected:", features)
        st.write("Features given:", input_data.columns.tolist())
