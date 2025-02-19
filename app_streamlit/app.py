import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

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

# Importo peresys.csv
peresys_df = pd.read_csv('../data/02_processed/peresys.csv')

# Filtro las 12 películas de referencia
input_movies_df = peresys_df[peresys_df["title"].isin(input_movies)]

# Creo la matriz usuario-película
user_item_matrix = input_movies_df.pivot_table(
    index="userId", 
    columns="title", 
    values="rating", 
    aggfunc="mean"
).fillna(0)

# Función para recoger votos del usuario
def get_user_ratings():
    user_ratings = []
    st.write("Valora las siguientes películas (de 0 a 5):")
    for movie in input_movies:
        rating = st.slider(f"{movie}", 0, 5, 3)  # Valor predeterminado 3
        user_ratings.append(rating)
    return user_ratings

# Mostrar interfaz de Streamlit
st.title("Sistema de Recomendación de Películas")
st.write("¡Bienvenido al sistema de recomendación! Por favor, valora las películas que has visto.")

# Recoger las valoraciones del usuario
user_ratings = get_user_ratings()

# Crear vector de usuario
user_vector = np.array(user_ratings).reshape(1, -1)

# Definir X e y
X = user_item_matrix.values 
y = peresys_df.groupby("userId")["rating"].mean()

# Para que los shapes de X e y coincidan
common_users = user_item_matrix.index.intersection(y.index)
X = user_item_matrix.loc[common_users]
y = y.loc[common_users]

# Dividir los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo KNN
knn_model = NearestNeighbors(n_neighbors=10, metric="cosine")
knn_model.fit(X_train)

# Encontrar usuarios más similares
distances, indices = knn_model.kneighbors(user_vector)
similar_users_indices = indices[0]
similar_users = user_item_matrix.iloc[similar_users_indices].index.tolist()

# Contar las películas votadas por cada usuario similar
overlap_counts = []
for user in similar_users:
    user_movies = peresys_df[peresys_df["userId"] == user]["title"].unique().tolist()
    common_movies = len(set(user_movies).intersection(set(input_movies)))
    overlap_percent = (common_movies / len(input_movies)) * 100
    overlap_counts.append(overlap_percent)

# Crear DataFrame con resultados
similar_users_df = pd.DataFrame({
    "UserID": similar_users,
    "Similitud (%)": overlap_counts
})

# Ordenar y seleccionar 10 usuarios con gustos más cercanos
similar_users_df = similar_users_df.sort_values(by="Similitud (%)", ascending=False).head(10)

# Mostrar resultados
st.write("### Ranking de 10 usuarios con gustos más cercanos a ti:")
st.write(similar_users_df.round(2))

# Generar 15 recomendaciones
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

# Combinar recomendaciones con metadatos
movies_metadata = peresys_df[["title", "year", "filmmaker"]].drop_duplicates(subset="title")
final_recommendations = recommendations.merge(
    movies_metadata,
    left_on="title",
    right_on="title",
    how="left"
)

# Mostrar recomendaciones en la interfaz
st.write("### Top 10 películas recomendadas para ti:")
st.write(final_recommendations[["title", "year", "filmmaker"]]
         .rename(columns={"title": "Título", "year": "Año", "filmmaker": "Director"}))