# PERESYS
Proyecto ML para la creación de un sistema de recomendación de películas

A partir de dos grandes dataset, procedentes de The Movie DataBase (TMDB) y MovieLens (ML), quiero crear un sistema de recomendación de películas para aficionados al Cine. Este sistema de recomendación se podría implementar en una plataforma de video on demand (VoD) como Amazon Prime Video, Filmin o Netflix, por nombrar a tres.  

Los datasets de los que dispongo son: 

**TMDB 5000 Movie Dataset**
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

4807 registros X 24 campos

*This dataset was generated from The Movie Database API (https://www.kaggle.com/code/sohier/getting-imdb-kernels-working-with-tmdb-data). This product uses the TMDb API but is not endorsed or certified by TMDb.
Their API also provides access to data on many additional movies, actors and actresses, crew members, and TV shows. You can try it for yourself here. (https://gist.github.com/SohierDane/4a84cb96d220fc4791f52562be37968b)*

El dataset se reparte en los siguientes csv: 
tmdb_5000_movies.csv
tmdb_5000_credits.csv

tmdb_5000_movies.csv incluye variables: budget, genres, homepage, id, keywords, original_language, original_title, overview, popularity, production_companies, production_countries, release_date, revenue, runtime ,spoken_languages, status, tagline, title, vote_average, vote_count <br>
tmdb_5000_credits.csv incluye variables: movie_id, title, cast, crew <br>
<br>
<br>
**Movie Lens 32M**
https://grouplens.org/datasets/movielens/32m/

*MovieLens 32M movie ratings. Stable benchmark dataset. 32 million ratings and two million tag applications applied to 87,585 movies by 200,948 users. Collected 10/2023 Released 05/2024*
•	README.txt
•	ml-32m.zip (size: 239 MB, checksum)
Permalink: https://grouplens.org/datasets/movielens/32m/

El dataset se reparte en los siguientes csv: 
movies.csv
ratings.csv
links.csv
tags.csv

links.csv incluye variables: movieId, imdbId, tmdbId <br>
movies.csv incluye variables: movieId, title, genres <br>
ratings.csv incluye variables: userId, movieId, rating, timestamp <br>
tags.csv incluye variables: userId, movieId, tag, timestamp <br>

9 keys 
