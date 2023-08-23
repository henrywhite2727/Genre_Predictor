import numpy as np
from genre_predictor.importing import get_genre_names, get_genre_data
from genre_predictor.algorithms import get_similarity_score, Genre_Predictor
from genre_predictor.graphics import make_genre_comparison_plot


make_genre_comparison_plot(True)


song_name = "Forever"

genre = Genre_Predictor(song_name)
print(genre)

