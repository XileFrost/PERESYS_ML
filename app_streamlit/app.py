import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.neighbors import NearestNeighbors

# Importo peresys.csv
peresys_df = pd.read_csv('../data/02_processed/peresys.csv')

# Ruta del modelo entrenado
MODEL_PATH = '../models/modelo1NearestNeighbors.pkl'

# Cargar el modelo entrenado desde el archivo .pkl
@st.cache_resource  # Cachear el modelo para no recargarlo en cada ejecución
def load_model():
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    return model

# Cargar datos preprocesados
@st.cache_data  # Cachear los datos para no recargarlos en cada ejecución
def load_data():
    # Cargar user_item_matrix, X, y desde archivos preprocesados
    user_item_matrix = pd.read_pickle('../src/user_item_matrix.pkl')
    X = np.load('../src/X.npy')
    y = pd.read_pickle('../src/y.pkl')
    return user_item_matrix, X, y

# Lista de películas para el input del usuario
input_movies = [
    "The Shawshank Redemption",
    "Forrest Gump",
    "Batman",
    "Pulp Fiction",
    "The Matrix",
    "The Silence of the Lambs",
    "Star Wars",
    "Fight Club",
    "Jurassic Park",
    "Schindler's List",
    "The Lord of the Rings: The Fellowship of the Ring",
    "The Empire Strikes Back"
]

# Función para recoger votos del usuario en Streamlit
def get_user_ratings():
    user_ratings = []
    st.write("Valora las siguientes películas (de 0 a 5):")
    for movie in input_movies:
        rating = st.number_input(
            f"{movie}",
            min_value=0.0,
            max_value=5.0,
            value=2.5,
            step=0.5,
            key=movie
        )
        user_ratings.append(rating)
    return user_ratings

# Interfaz de Streamlit
st.title("Sistema de Recomendación de Películas")

# Cargar el modelo y los datos
knn_model = load_model()
user_item_matrix, X, y = load_data()

# Obtener votos del usuario
user_ratings = get_user_ratings()

if st.button("Obtener Recomendaciones"):
    user_vector = np.array(user_ratings).reshape(1, -1)
    
    # Encuentro usuarios más similares
    distances, indices = knn_model.kneighbors(user_vector)
    similar_users_indices = indices[0]
    similar_users = user_item_matrix.iloc[similar_users_indices].index.tolist()
    
    # Busco cuántas películas del usuario ha votado cada usuario similar
    overlap_counts = []
    for user in similar_users:
        user_movies = peresys_df[peresys_df["userId"] == user]["title"].unique().tolist()
        common_movies = len(set(user_movies).intersection(set(input_movies)))
        overlap_percent = (common_movies / len(input_movies)) * 100
        overlap_counts.append(overlap_percent)
    
    # Creo DataFrame con resultados
    similar_users_df = pd.DataFrame({
        "UserID": similar_users,
        "Similitud (%)": overlap_counts
    })
    
    # Ordeno y selecciono 10 usuarios con gustos cercanos al usuario
    similar_users_df = similar_users_df.sort_values(by="Similitud (%)", ascending=False).head(10)
    
    # Muestro resultados de usuarios cercanos
    st.write("Ranking 10 usuarios con gustos más cercanos a ti:")
    st.dataframe(similar_users_df.round(2))
    
    # Genero 15 recomendaciones (excluyendo las 12 ya votadas)
    recommendations = (
        peresys_df[
            peresys_df["userId"].isin(similar_users) &
            ~peresys_df["title"].isin(input_movies)
        ]
        .groupby("title")["rating"]
        .agg(["mean", "count"])
        .sort_values(by=["mean", "count"], ascending=False)
        .head(15)
        .reset_index()
    )
    
    # Combino recomendaciones con metadatos
    movies_metadata = peresys_df[["title", "year", "filmmaker"]].drop_duplicates(subset="title")
    final_recommendations = recommendations.merge(
        movies_metadata,
        left_on="title",
        right_on="title",
        how="left"
    )
    
    # Muestro recomendaciones por título, año y director
    st.write("\nTop 10 películas recomendadas para ti:")
    st.dataframe(final_recommendations[["title", "year", "filmmaker"]]
                 .rename(columns={"title": "Título", "year": "Año", "filmmaker": "Director"}))