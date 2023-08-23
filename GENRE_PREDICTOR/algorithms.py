from genre_predictor.importing import get_genre_data, get_genre_names
import numpy as np


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


def get_avgs():
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

    # print("All Avg Metrics:", all_avg_metrics)

    return all_avg_metrics


def make_weights_array():
    N_weights = 20  # this variable controls the number of different weights we want to test per loop

    weights = np.zeros(
        (N_weights, 9)
    )  # this will be a array of size 100 x N_properties (it has 100 rows so I can loop over all potential weights)
    for i in range(9):
        weights[:, i] = np.linspace(
            -5, 5, N_weights
        )  # the range [-5,5] was picked somewhat arbitrarily

    return weights


# this function is TBD
def find_optimal_weights(weights):
    for a in range(len(weights[:, a])):
        h = 1


def get_similarity_score(Song_data, all_avg_metrics, song_index):
    # calculating similarity scores for musical statistics (energy, speechiness, etc.)
    # a lower similarity score means things are MORE similar

    song_properties = Song_data[song_index, :]

    dance_similarity = np.abs(song_properties[0] - all_avg_metrics[:, 0])

    energy_similarity = np.abs(song_properties[1] - all_avg_metrics[:, 1])

    loud_similarity = np.abs(song_properties[3] - all_avg_metrics[:, 3])

    speech_similarity = np.abs(song_properties[5] - all_avg_metrics[:, 5])

    acoustic_similarity = np.abs(song_properties[6] - all_avg_metrics[:, 6])

    instrument_similarity = np.abs(song_properties[7] - all_avg_metrics[:, 7])

    live_similarity = np.abs(song_properties[8] - all_avg_metrics[:, 8])

    valence_similarity = np.abs(song_properties[9] - all_avg_metrics[:, 9])

    tempo_similarity = np.abs(song_properties[10] - all_avg_metrics[:, 8])

    similarity_score = (
        dance_similarity
        + loud_similarity
        + live_similarity
        + valence_similarity
        + tempo_similarity
        + energy_similarity
        + acoustic_similarity
        + speech_similarity
        + instrument_similarity
    )
    return similarity_score


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


# Creating genre predictor function
def Genre_Predictor(song_name: str):
    # finding specific song within data set
    song_index = 200000  # initializing variable that holds the index of the song we are trying to predict the genre of
    for i in range(len(Song_names)):
        if song_name == Song_names[i]:
            song_index = i
            break
    if song_index == 200000:
        print(
            "Song name provided is not in the catalogued list so is not a valid input."
        )

    all_avg_metrics = get_avgs()

    similarity_score = get_similarity_score(Song_data, all_avg_metrics, song_index)

    best_match = np.argmin(similarity_score)

    return genre_names[best_match]
