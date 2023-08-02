import numpy as np
import matplotlib.pyplot as plt

# Loading all data into one array
Song_data = np.loadtxt(
    "data/genres_v2.csv",
    skiprows=1,
    delimiter=",",
    usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    encoding="utf8",
)

# Splitting song data into genres (and ignoring unnecessary columns in data set)
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

# finding average metrics for each genre (and saving in one array)
Genres = [
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

N_gen = len(Genres)  # number of genres within the data set
N_metrics = len(DarkTrap[0, :])  # number of metrics associated with each song

all_avg_metrics = np.zeros(
    (N_gen, N_metrics)
)  # array that will hold the avg metrics for every single genre
avg_metric = np.zeros(
    N_metrics
)  # array that will hold the average metrics for a SINGLE genre within the following loop

# looping over each genre
for i in range(len(Genres)):
    genre = Genres[i]
    # looping over each metric for each genre
    for j in range(len(genre[0, :])):
        avg_metric[j] = np.mean(genre[:, j])

    all_avg_metrics[i, :] = avg_metric


# plotting separation from averages for each genre to find outliers
