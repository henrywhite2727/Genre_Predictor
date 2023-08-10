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

genre_names=get_genre_names()

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
    Song_names = np.loadtxt(
        "genre_predictor/data/genres_v2.csv",
        dtype=str,
        skiprows=1,
        delimiter=",",
        usecols=(19),
        encoding="utf8",
    )

    DarkTrap = Song_data[0:4578, :]
    Und_rap = Song_data[4579:10453, :]
    TrapMetal = Song_data[10454:12409, :]
    Emo = Song_data[12410:14089, :]
    Rap = Song_data[14090:15937, :]
    RnB = Song_data[15937:18036, :]
    Pop = Song_data[18037:18497, :]
    Hiphop = Song_data[18498:21525, :]
    Techhouse = Song_data[21256:24500, :]
    Techno = Song_data[24501:27456, :]
    Trance = Song_data[27457:30455, :]
    Psytrance = Song_data[30456:33416, :]
    Trap = Song_data[33417:36404, :]
    Dnb = Song_data[36404:39369, :]
    Hardstyle = Song_data[39370:42305, :]
    
    return (
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
    ),Song_data,Song_names
