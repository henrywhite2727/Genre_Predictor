import numpy as np
import matplotlib.pyplot as plt


# Comparing average metrics for different genres
def get_genre_names():
    genre_names: list = [
        "DarkTrap",
        "Und_rap",
        "TrapMetal",
        "Emo",
        "Rap",
        "RnB",
        "Pop",
        "Hiphop",
        "Techhouse",
        "Techno",
        "Trance",
        "Psytrance",
        "Trap",
        "Dnb",
        "Hardstyle",
    ]
    return genre_names


genre_names = get_genre_names()

# Splitting song data into genres (and ignoring unnecessary columns in data set)
def get_genre_data():

    # Loading all data into one array
    Song_data = np.loadtxt(
        "genre_predictor/data/genres_v2.csv",
        skiprows=1,
        delimiter=",",
        usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        encoding="utf8",
    )

    Song_data_norm = normalize_song_data(Song_data)

    Song_names = np.loadtxt(
        "genre_predictor/data/genres_v2.csv",
        dtype=str,
        skiprows=1,
        delimiter=",",
        usecols=(19),
        encoding="utf8",
    )

    DarkTrap = Song_data_norm[0:4578, :]
    Und_rap = Song_data_norm[4579:10453, :]
    TrapMetal = Song_data_norm[10454:12409, :]
    Emo = Song_data_norm[12410:14089, :]
    Rap = Song_data_norm[14090:15937, :]
    RnB = Song_data_norm[15937:18036, :]
    Pop = Song_data_norm[18037:18497, :]
    Hiphop = Song_data_norm[18498:21525, :]
    Techhouse = Song_data_norm[21256:24500, :]
    Techno = Song_data_norm[24501:27456, :]
    Trance = Song_data_norm[27457:30455, :]
    Psytrance = Song_data_norm[30456:33416, :]
    Trap = Song_data_norm[33417:36404, :]
    Dnb = Song_data_norm[36404:39369, :]
    Hardstyle = Song_data_norm[39370:42305, :]

    return (
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
        Song_data_norm,
        Song_names,
    )


def normalize_song_data(song_data):
    song_data_norm = np.copy(
        song_data
    )  # initializing variable, all values will be replaced so it is irrelevant that it is a copy of song_data

    # looping over each property (i.e. energy, loudness, etc.) to normalize them
    for i in range(len(song_data[0, :])):
        song_data_norm[:, i] = (song_data[:, i] - np.min(song_data[:, i])) / (
            np.max(song_data[:, i]) - np.min(song_data[:, i])
        )  # from https://www.statology.org/numpy-normalize-between-0-and-1/

    return song_data_norm
