import pytest
from genre_predictor.Spotify_project import Genre_Predictor

# lalalalala
def test_pp():
    song_name = "Killer"

    genre_predicted = Genre_Predictor(song_name)
    print(genre_predicted)
    assert 0 == 0
