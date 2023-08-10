from importing import get_genre_data, get_genre_names
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

    return all_avg_metrics


def get_similarity_score(
    Song_data, song_index: int, energy_avgs, acoustic_avgs, speech_avgs, instrument_avgs
):
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

    similarity_score = get_similarity_score(Song_data)

    best_match = np.argmin(similarity_score)

    return genre_names[best_match]
