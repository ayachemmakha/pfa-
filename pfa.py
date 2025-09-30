
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ----------------------------
# Configuration de la page
# ----------------------------
st.set_page_config(
    page_title="MyHealth Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Style personnalisé
# ----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f9fdfb;
        font-family: "Segoe UI", sans-serif;
    }
    .title {
        color: #2e7d32;
        text-align: center;
        font-size: 36px !important;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 20px;
    }
    .stButton button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: bold;
        border: none;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #1b5e20;
        color: white;
    }
    .card {
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Charger le modèle
# ----------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("pfa.pkl")
        return model
    except Exception as e:
        st.error(f" Erreur lors du chargement du modèle : {e}")
        return None

model = load_model()

# ----------------------------
# Fonction prédiction
# ----------------------------
def safe_predict(model, inputs):
    try:
        X = np.array(inputs).reshape(1, -1)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        return pred, proba
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return None, None

# ----------------------------
# Header
# ----------------------------
st.markdown('<h1 class="title">MyHealth Predictor 🩺</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analysez vos paramètres médicaux et obtenez une estimation du risque de diabète</p>', unsafe_allow_html=True)
st.markdown("---")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("ℹ️ À propos")
    st.markdown("""
    Cet outil utilise un modèle **Random Forest** pour estimer le risque de diabète
    en fonction de 8 paramètres médicaux :  
    - Grossesses  
    - Glucose  
    - Pression artérielle  
    - Épaisseur de la peau  
    - Insuline  
    - BMI  
    - DPF (Hérédité)  
    - Âge  

    ⚠️ **Note** : Ce modèle est un outil de support, pas un diagnostic médical.
    """)

# ----------------------------
# Formulaire utilisateur
# ----------------------------
if model:
    st.subheader("📝 Entrez vos informations médicales")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Grossesses", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Pression artérielle", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Épaisseur de la peau", min_value=0, max_value=100, value=20)

    with col2:
        insulin = st.number_input("Insuline", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Âge", min_value=0, max_value=120, value=30)

    # Quand on clique sur prédire
    if st.button("🔮 Prédire"):
        inputs = [pregnancies, glucose, blood_pressure, skin_thickness,
                  insulin, bmi, dpf, age]

        pred, proba = safe_predict(model, inputs)

        if pred is not None:
            if pred == 1:
                st.error(f"⚠️ Risque élevé de diabète (Probabilité: {proba[1]*100:.2f}%)")
            else:
                st.success(f"✅ Risque faible de diabète (Probabilité: {proba[0]*100:.2f}%)")

            # ----------------------------
            # Sauvegarde des données utilisateur
            # ----------------------------
            if not os.path.exists("data"):
                os.makedirs("data")

            file_path = os.path.join("data", "users_data.csv")

            if os.path.exists(file_path):
                data = pd.read_csv(file_path)
            else:
                data = pd.DataFrame(columns=[
                    "pregnancies","glucose","blood_pressure","skin_thickness",
                    "insulin","bmi","dpf","age","prediction"
                ])

            new_row = {
                "pregnancies": pregnancies,
                "glucose": glucose,
                "blood_pressure": blood_pressure,
                "skin_thickness": skin_thickness,
                "insulin": insulin,
                "bmi": bmi,
                "dpf": dpf,
                "age": age,
                "prediction": pred
            }

            data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
            data.to_csv(file_path, index=False)
            st.info("✅ Les données ont été enregistrées avec succès.")

# ----------------------------
# Afficher les données enregistrées
# ----------------------------
st.subheader("📊 Historique des utilisateurs")
file_path = os.path.join("data", "users_data.csv")
if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    st.dataframe(data)
else:
    st.write("Aucune donnée enregistrée pour l'instant.")






    
