import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Define Hugging Face repository and model file details
HF_MODEL_REPO = "yellaji/visit-with-us-wellness-model"
MODEL_FILE = "best_wellness_tourism_model.joblib"

st.set_page_config(layout="wide")
st.title("Visit With Us – Wellness Tourism Package Predictor")

@st.cache_resource
def load_model():
    """Loads the pre-trained model from Hugging Face Hub."""
    try:
        # Download the model file from Hugging Face Hub
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=MODEL_FILE,
            repo_type="model"
        )
        # Load the model using joblib
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure the model exists in the Hugging Face Hub.")
        return None

# Load the model
model = load_model()

# Define mappings for categorical features based on assumed LabelEncoder behavior (alphabetical)
# These mappings ensure the Streamlit app sends integer inputs that the model expects
# for columns that were label-encoded before OneHotEncoder was applied in the pipeline.
TYPE_OF_CONTACT_MAP = {'Company Invited': 0, 'Self Inquiry': 1}
OCCUPATION_MAP = {'Freelancer': 0, 'Large Business': 1, 'Salaried': 2, 'Small Business': 3, 'Unemployed': 4}
GENDER_MAP = {'Female': 0, 'Male': 1}
MARITAL_STATUS_MAP = {'Divorced': 0, 'Married': 1, 'Single': 2}
DESIGNATION_MAP = {'Analyst': 0, 'Director': 1, 'Executive': 2, 'Manager': 3, 'Senior Manager': 4}
PRODUCT_PITCHED_MAP = {'Basic': 0, 'Deluxe': 1, 'King': 2, 'Luxury': 3, 'Standard': 4, 'Super Deluxe': 5}

if model:
    st.header("Customer Information for Wellness Package Prediction")

    # Organize inputs into columns
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=90, value=35)
        type_of_contact = st.selectbox("Type of Contact", list(TYPE_OF_CONTACT_MAP.keys()))
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", list(OCCUPATION_MAP.keys()))
        gender = st.radio("Gender", list(GENDER_MAP.keys()))
        number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)

    with col2:
        preferred_property_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
        marital_status = st.selectbox("Marital Status", list(MARITAL_STATUS_MAP.keys()))
        number_of_trips = st.number_input("Number of Trips (Annually)", min_value=0, max_value=20, value=2)
        passport = st.radio("Passport", ['Yes', 'No'])
        own_car = st.radio("Own Car", ['Yes', 'No'])
        number_of_children_visiting = st.number_input("Number of Children Visiting (below 5)", min_value=0, max_value=5, value=0)

    with col3:
        designation = st.selectbox("Designation", list(DESIGNATION_MAP.keys()))
        monthly_income = st.number_input("Monthly Income", min_value=10000, max_value=200000, value=50000)
        pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
        product_pitched = st.selectbox("Product Pitched", list(PRODUCT_PITCHED_MAP.keys()))
        number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10)

    if st.button("Predict Purchase"):
        # Map selected string values to their integer representations
        mapped_type_of_contact = TYPE_OF_CONTACT_MAP[type_of_contact]
        mapped_occupation = OCCUPATION_MAP[occupation]
        mapped_gender = GENDER_MAP[gender]
        mapped_marital_status = MARITAL_STATUS_MAP[marital_status]
        mapped_designation = DESIGNATION_MAP[designation]
        mapped_product_pitched = PRODUCT_PITCHED_MAP[product_pitched]
        mapped_passport = 1 if passport == 'Yes' else 0
        mapped_own_car = 1 if own_car == 'Yes' else 0

        # Create DataFrame for prediction, ensuring correct column order
        input_data = pd.DataFrame([{
            'Age': age,
            'TypeofContact': mapped_type_of_contact,
            'CityTier': city_tier,
            'DurationOfPitch': duration_of_pitch,
            'Occupation': mapped_occupation,
            'Gender': mapped_gender,
            'NumberOfPersonVisiting': number_of_person_visiting,
            'PreferredPropertyStar': preferred_property_star,
            'MaritalStatus': mapped_marital_status,
            'NumberOfTrips': number_of_trips,
            'Passport': mapped_passport,
            'OwnCar': mapped_own_car,
            'NumberOfChildrenVisiting': number_of_children_visiting,
            'MonthlyIncome': monthly_income,
            'PitchSatisfactionScore': pitch_satisfaction_score,
            'ProductPitched': mapped_product_pitched,
            'NumberOfFollowups': number_of_followups,
            'Designation': mapped_designation
        }])

        # Make prediction
        prediction_proba = model.predict_proba(input_data)[:, 1]
        classification_threshold = 0.45 # Using the same threshold as in train.py
        prediction = (prediction_proba >= classification_threshold).astype(int)[0]
        
        st.subheader("Prediction Results:")
        if prediction == 1:
            st.success(f"This customer is likely to purchase the Wellness Tourism Package! (Confidence: {prediction_proba[0]:.2f})")
        else:
            st.warning(f"This customer is unlikely to purchase the Wellness Tourism Package. (Confidence: {prediction_proba[0]:.2f})")
else:
    st.info("Model is not loaded. Please check the model loading process.")
