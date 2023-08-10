import numpy as np
from importing import get_genre_names, get_genre_data
from algorithms import get_similarity_score, Genre_Predictor
from graphics import make_genre_comparison_plot


make_genre_comparison_plot(False)

song_name = "Forever"

genre = Genre_Predictor(song_name)
print(genre)

