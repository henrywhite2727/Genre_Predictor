import numpy as np
import matplotlib.pyplot as plt

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


# Comparing average metrics for different genres
genre_names = [
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

energy_avgs = all_avg_metrics[:, 1]
acoustic_avgs = all_avg_metrics[:, 6]
speech_avgs = all_avg_metrics[:, 5]
instrument_avgs = all_avg_metrics[:, 7]


song_index = 200000  # initializing variable that holds the index of the song we are trying to predict the genre of

# Creating genre predictor function
def Genre_Predictor(song_name: str):
    # finding specific song within data set
    for i in range(len(Song_names)):
        if song_name == Song_names[i]:
            song_index = i
            break
    if song_index == 200000:
        print(
            "Song name provided is not in the catalogued list so is not a valid input."
        )

    # calculating similarity scores for energy, acoustic, speechiness, and instrumentalness
    song_properties = Song_data[song_index, :]

    song_energy = song_properties[1]
    energy_similarity = np.abs(song_energy - energy_avgs)

    song_acoustic = song_properties[6]
    acoustic_similarity = np.abs(song_acoustic - acoustic_avgs)

    song_speech = song_properties[5]
    speech_similarity = np.abs(song_speech - speech_avgs)

    song_instrument = song_properties[7]
    instrument_similarity = np.abs(song_instrument - instrument_avgs)

    similarity_score = (
        energy_similarity
        + acoustic_similarity
        + speech_similarity
        + instrument_similarity
    )
    best_match = np.argmin(similarity_score)

    return genre_names[best_match]


# testing
genre = Genre_Predictor("Many Men")
print(genre)


#%% Graphics

x_pos = np.arange(len(genre_names))

# plotting
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=[15, 10])
fig.subplots_adjust(hspace=0.5)

# ax1.figure(1, figsize=[15, 5])
ax1.bar(x_pos, energy_avgs)
ax1.set_xticks(x_pos, genre_names)
ax1.set_ylabel("Energy")


# ax2.figure(2, figsize=[15, 5])
ax2.bar(x_pos, acoustic_avgs)
ax2.set_xticks(x_pos, genre_names)
ax2.set_ylabel("Acousticness")

ax3.bar(x_pos, speech_avgs)
ax3.set_xticks(x_pos, genre_names)
ax3.set_ylabel("Speechiness")


ax4.bar(x_pos, instrument_avgs)
ax4.set_xticks(x_pos, genre_names)
ax4.set_ylabel("Instrumentalness")


# plotting separation from averages for each genre to find outliers
