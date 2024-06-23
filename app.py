import pickle
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download

# Function to load the scaler
def load_scaler():
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler

# Function to preprocess user input
def preprocess_input(user_input, scaler):
    user_input_df = pd.DataFrame([user_input], columns=feature_names)
    scaled_input = scaler.transform(user_input_df)
    return pd.DataFrame(scaled_input, columns=user_input_df.columns)

# Function to load the model
def load_model():
    model_path = hf_hub_download(repo_id="elladeandra/sports-prediction", filename="ensemble_model.pkl")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Define feature names
feature_names = ['value_eur', 'age', 'potential', 'movement_reactions', 'wage_eur']

# Streamlit app title and description
st.title('Football Player Rating Predictor')
st.markdown("""
    This application predicts the rating of a football player based on their attributes using an ensemble model.
    The model combines Random Forest, Gradient Boosting, and XGBoost algorithms for robust predictions.
""")

# Sidebar for user input
st.sidebar.header('Input Player Attributes')
def get_user_input():
    value_eur = st.sidebar.number_input('Market Value (EUR)', min_value=0, max_value=int(1e9), value=int(1e6))
    wage_eur = st.sidebar.number_input('Weekly Wage (EUR)', min_value=0, max_value=int(1e9), value=int(1e6))
    age = st.sidebar.slider('Player Age', 16, 40, 25)
    potential = st.sidebar.slider('Potential Score', 1, 100, 50)
    movement_reactions = st.sidebar.slider('Reactions', 1, 100, 50)
    
    data = {
        'value_eur': value_eur,
        'wage_eur': wage_eur,
        'age': age,
        'potential': potential,
        'movement_reactions': movement_reactions
    }
    return data

user_input = get_user_input()

try:
    # Load scaler and preprocess input
    scaler = load_scaler()
    scaled_input = preprocess_input(user_input, scaler)

    # Load model and predict
    model = load_model()
    predicted_rating = model.predict(scaled_input)

    # Display prediction
    st.subheader('Predicted Player Rating')
    st.write(f"Estimated Rating: {predicted_rating[0]:.1f}")

    # Explanation section
    if st.button('About the Prediction'):
        st.markdown("""
            This application uses an ensemble model combining Random Forest, Gradient Boosting, and XGBoost algorithms to predict football player ratings.
            
            The model is trained on data from the FIFA video game series, which includes attributes such as age, potential, market value, and reaction times.
            
            **Note**: This is a demo project and should not be used for professional scouting or analysis purposes.
        """)

except Exception as e:
    st.error(f"An error occurred: {e}")

# Additional features for better user experience
if st.sidebar.button('Reset Inputs'):
    st.experimental_rerun()

st.sidebar.markdown("""
    **Instructions**:
    - Adjust the player attributes using the input fields.
    - Click the 'Predict' button to see the estimated rating.
    - Use the 'About the Prediction' button for more information.
""")
