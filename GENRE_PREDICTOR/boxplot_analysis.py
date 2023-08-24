import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from genre_predictor.importing import get_genre_names, get_genre_data
from genre_predictor.algorithms import get_similarity_score, Genre_Predictor
from genre_predictor.graphics import make_genre_comparison_plot


# %% Importing necessary data
genre_names = get_genre_names()

(
    (
        DarkTrap,
        Und_rap,
        TrapMetal,
        Emo,
        Rap,
        RnB,
        Pop,
        Hiphop,
        Techhouse,
        Techno,
        Trance,
        Psytrance,
        Trap,
        Dnb,
        Hardstyle,
    ),
    Song_data,
    Song_names,
) = get_genre_data()

Genre_data = [
    DarkTrap,
    Und_rap,
    TrapMetal,
    Emo,
    Rap,
    RnB,
    Pop,
    Hiphop,
    Techhouse,
    Techno,
    Trance,
    Psytrance,
    Trap,
    Dnb,
    Hardstyle,
]


def make_box_plot(genre_data: list, property_index: int, property_name: str):
    """_summary_

    Args:
        genre_data (list): List of 2d arrays that contain all of the property data within a single genre
        property_index (int): index of a property within genres_v2.csv (i.e. for danceability it is zero)
        property_name (str): name of a property for axis labelling
    """
    x_pos = np.arange(len(genre_data))
    genre_list = [None] * 15  # empty list
    for i in range(len(genre_data)):
        specific_genre_data = genre_data[i]
        genre_list[i] = specific_genre_data[:, property_index]

    plt.figure(1, figsize=[13, 3])
    plt.boxplot(genre_list)
    plt.xticks(x_pos + 1, genre_names)
    plt.ylabel(property_name)

    plt.show()


DarkTrap = Genre_data[0]
make_box_plot(Genre_data, 6, "Acousticness")
