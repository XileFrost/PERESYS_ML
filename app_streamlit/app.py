import pandas as pd
import numpy as np
import joblib
import streamlit as st
import h5py
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

# Configuración de rutas
MODEL_PATH = '../models/modelo5NCF.keras'
MOVIES_METADATA_PATH = '../src/movies_metadata.pkl'
TITLE_TO_CODE_PATH = '../src/title_to_code.pkl'
USER_ENCODER_PATH = '../src/user_encoder.joblib'
USER_MOVIE_MATRIX_PATH = '../src/user_movie_matrix.h5'

# Configuración de la página
st.set_page_config(
    page_title="PERESYS - Recomendación Cinematográfica",
    page_icon="🎬",
    layout="wide"
)

# Estilos CSS y JavaScript personalizados para autoclean
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
        color: #333333;
    }
    .stNumberInput input {
        background-color: #f9f9f9;
        color: black;
    }
    .stButton>button {
        background-color: #e50914;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #b20710;
        transform: scale(1.05);
    }
    .movie-card {
        background-color: #f2f2f2;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Función para cargar la matriz HDF5 con chunking
@st.cache_data
def load_h5_matrix():
    with h5py.File(USER_MOVIE_MATRIX_PATH, 'r') as h5f:
        dataset = h5f['user_movie_matrix']
        chunk_size = 1000  # Tamaño óptimo para matrices grandes
        total_users = dataset.shape[0]
        matrix = np.empty(dataset.shape, dtype=np.float32)
        
        # Carga progresiva por chunks
        for i in range(0, total_users, chunk_size):
            end_idx = min(i + chunk_size, total_users)
            matrix[i:end_idx] = dataset[i:end_idx]
            
        return matrix

# Función para cargar todos los componentes
@st.cache_data
def load_trained_data():
    return {
        'movies_metadata': pd.read_pickle(MOVIES_METADATA_PATH),
        'title_to_code': joblib.load(TITLE_TO_CODE_PATH),
        'user_encoder': joblib.load(USER_ENCODER_PATH),
        'user_movie_matrix': load_h5_matrix()
    }

# Cargar modelo y datos
model = load_model(MODEL_PATH)
data = load_trained_data()

# Lista de películas para valoración
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

# Interfaz de usuario
st.title("🎬 PERESYS - Sistema de Recomendación Cinematográfica")
st.markdown("### Valora las películas (1-5) para obtener recomendaciones")

# Usar un formulario para agrupar las entradas
with st.form("rating_form"):
    # Sección de valoraciones
    ratings = {}
    cols = st.columns(3)
    
    # Obtener los años de las películas desde los metadatos
    movies_metadata = data['movies_metadata']
    batman_1989 = movies_metadata[
        (movies_metadata['title'] == 'Batman') & 
        (movies_metadata['year'] == 1989)
    ]

    if not batman_1989.empty:
        correct_code = batman_1989['movie_code'].values[0]
        data['title_to_code']['Batman'] = correct_code

    batman_1989_mask = (movies_metadata['title'] == 'Batman') & (movies_metadata['year'] == 1989)
    other_movies_mask = movies_metadata['title'].isin(input_movies) & (movies_metadata['title'] != 'Batman')
    filtered_metadata = movies_metadata[batman_1989_mask | other_movies_mask]

    movies_year = filtered_metadata.set_index('title')['year'].to_dict()

    for i, movie in enumerate(input_movies):
        with cols[i % 3]:
            year = movies_year.get(movie, "Desconocido") 
            ratings[movie] = st.number_input(
                label=f"**{movie}** ({year})", 
                min_value=1,
                max_value=5,
                step=1,  
                format="%d",  
                key=movie,  # Clave única para cada input
                value=None  # Aquí configuramos el valor inicial como None para que no se vea ningún número en la entrada
            )

    # Botón de recomendaciones dentro del formulario
    submit_button = st.form_submit_button("🎥 Generar Recomendaciones", use_container_width=True)

# Solo procesar cuando se presiona el botón
if submit_button:
    with st.spinner('Analizando tus gustos cinematográficos...'):
        # Convertir ratings a vector
        new_user_vector = np.zeros(data['user_movie_matrix'].shape[1])
        title_to_code = data['title_to_code']
        
        for title, rating in ratings.items():
            if title in title_to_code:
                movie_code = title_to_code[title]
                if movie_code < data['user_movie_matrix'].shape[1]:  # Validación de índice
                    new_user_vector[movie_code] = rating

        # Cálculo de similitudes 
        similarities = cosine_similarity(
            [new_user_vector], 
            data['user_movie_matrix']
        )[0]
        
        # Procesar resultados 
        similar_users = np.argsort(similarities)[::-1][:10]
        user_encoder = data['user_encoder']
        top_user_ids = user_encoder.inverse_transform(similar_users)
        similarity_percentages = np.round(similarities[similar_users] * 100, 2)
        
        # Obtener películas valoradas por usuarios similares
        recommended_movie_codes = []
        for user_idx in similar_users:
            user_ratings = data['user_movie_matrix'][user_idx]
            # Considerar películas con valoración ≥4
            rated_indices = np.where(user_ratings >= 4)[0]
            recommended_movie_codes.extend(rated_indices.tolist())

        # Conteo y filtrado
        from collections import Counter
        movie_counts = Counter(recommended_movie_codes)
        
        recommended_movies_df = pd.DataFrame.from_dict(
            movie_counts, 
            orient='index', 
            columns=['user_count']
        ).reset_index()
        recommended_movies_df.rename(columns={'index': 'movie_code'}, inplace=True)
        
        # Filtrar películas ya valoradas
        rated_movies = [title_to_code[title] for title in ratings if title in title_to_code]
        recommended_movies_df = recommended_movies_df[
            ~recommended_movies_df['movie_code'].isin(rated_movies)
        ]
        
        # Fusionar con metadatos
        movies_metadata = data['movies_metadata']
        recommended_movies = recommended_movies_df.merge(
            movies_metadata[['movie_code', 'title', 'year', 'filmmaker']].drop_duplicates(subset=['movie_code']),
            on='movie_code',
            how='left'
        ).sort_values('user_count', ascending=False).head(15)

        # Mostrar resultados (existente con ajustes)
        st.success("¡Recomendaciones generadas con éxito!")
        
        st.markdown("### Usuarios con gustos similares:")
        for user_id, sim in zip(top_user_ids, similarity_percentages):
            st.markdown(f"- 🎞️ User ID `{user_id}` (Similitud: {sim}%)")
        
        st.markdown("### Películas recomendadas:")
        if not recommended_movies.empty:
            # Mostrar películas en 3 columnas por fila
            cols = st.columns(3)
            for idx, row in recommended_movies.iterrows():
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <h4>{row['title']} ({int(row['year'])})</h4>
                        <p>📽 Director: {row['filmmaker']}</p>
                        <p>👥 Recomendado por {row['user_count']} usuarios afines</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No se encontraron recomendaciones. Intenta valorar más películas.")