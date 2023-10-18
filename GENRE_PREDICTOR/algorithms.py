from genre_predictor.importing import get_genre_data, get_genre_names
import numpy as np
from scipy.stats import kstest


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


def get_medians(Song_data: np.ndarray, property_indices: list, genre_indices: list):
    """You will one day expand this function to make your AI work for all genres. Today is not that day

    Args:
        Song_data (np.ndarray): _description_
        property_indices (list): The indices of the properties being considered for this case (for now it is acousticness, instrumentalness, and valence)
        genre_indices (list): The indices of the genres we are trying to guess between (Techno and Rap for now)
    """
    medians = np.ones((len(genre_indices), len(property_indices)))
    for i in range(len(genre_indices)):
        for j in property_indices:
            medians[i, k] = np.median()


def get_likelihood_technorap(Song_data: np.ndarray, song_index: int):
    property_indices = [
        6,
        7,
        9,
    ]  # just considering acousticness, instrumentalness, and valence for now
    genre_indices = [4, 9]  # just conisdering rap and Techno for now

    # #I will save the code below for when I expand my genre_predictor to have more than 2 genres
    # medians = get_medians(
    #     Song_data, property_indices, genre_indices
    # )  # 2d array that stores the property medians for each genre (Num rows=number of genres, Num columns=number of properties being considered )

    v_m_rap = np.median(Rap[:, 9])  # median valence for rap
    v_m_techno = np.median(Techno[:, 9])  # median valence for rap
    v_w = 0.00001  # valence weight
    a_w = 0.00001  # acosuticness weight
    i_w = 1000  # instrumentalness weight
    a_m_rap = np.median(Rap[:, 6])
    a_m_techno = np.median(Techno[:, 9])  # median valence for rap

    i_m_rap = np.median(Rap[:, 7])
    i_m_techno = np.median(Techno[:, 9])  # median valence for rap

    # The 0.0000001 is used to avoid dividing by zero
    likelihood_rap = 1 / (
        (v_w * (Song_data[song_index, 9] - v_m_rap + 0.0000000001) ** 2)
        + (a_w * (Song_data[song_index, 6] - a_m_rap + 0.0000000001) ** 2)
        + (i_w * (Song_data[song_index, 7] - i_m_rap + 0.0000000001) ** 2)
    )

    likelihood_techno = 1 / (
        (v_w * (Song_data[song_index, 9] - v_m_techno + 0.0000000001) ** 2)
        + (a_w * (Song_data[song_index, 6] - a_m_techno + 0.0000000001) ** 2)
        + (i_w * (Song_data[song_index, 7] - i_m_techno + 0.0000000001) ** 2)
    )

    return [likelihood_rap, likelihood_techno]


def get_likelihood_emorap(Song_data: np.ndarray, song_index: int):
    d_m_rap = np.median(
        Rap[(len(Rap) // 2) :, 0]
    )  # median danceability for rap (note that the second half of the data set is the "training dataset")
    e_m_rap = np.median(Rap[(len(Rap) // 2) :, 1])
    s_m_rap = np.median(Rap[(len(Rap) // 2) :, 5])

    d_w = 1000  # danceability weight
    e_w = 1000  # energy weight
    s_w = 1000  # speechiness weight

    d_w, e_w, s_w = get_weights_kstest(Rap, Emo, [0, 1, 5])

    d_m_emo = np.median(Emo[(len(Emo) // 2) :, 0])
    e_m_emo = np.median(Emo[(len(Emo) // 2) :, 1])
    s_m_emo = np.median(Emo[(len(Rap) // 2) :, 5])

    # The 0.0000001 is used to avoid dividing by zero
    likelihood_rap = 1 / (
        (d_w * (Song_data[song_index, 0] - d_m_rap + 0.0000000001) ** 2)
        + (e_w * (Song_data[song_index, 1] - e_m_rap + 0.0000000001) ** 2)
        + (s_w * (Song_data[song_index, 5] - s_m_rap + 0.0000000001) ** 2)
    )

    likelihood_emo = 1 / (
        (d_w * (Song_data[song_index, 0] - d_m_emo + 0.0000000001) ** 2)
        + (e_w * (Song_data[song_index, 1] - e_m_emo + 0.0000000001) ** 2)
        + (s_w * (Song_data[song_index, 5] - s_m_emo + 0.0000000001) ** 2)
    )

    return [likelihood_rap, likelihood_emo]


def get_weights_kstest(
    data_genre1: np.ndarray, data_genre2: np.ndarray, property_indices: list
):
    """This function calculates the weight for a specific property (i.e. how useful this property is for distinguishing between two genres).
    This is done using a KS test which is a useful for statistical tool for comparing two different distributions

    Args:
        data_genre1 (np.ndarray): _description_
        data_genre2 (np.ndarray): _description_
        property_indices (list): list of indices associated with different properties (for dance and energy and speechiness, the list would be [0,1,5]). This

    Returns:
        _type_: _description_
    """
    weight0 = kstest(
        data_genre1[0:500, property_indices[0]], data_genre2[0:500, property_indices[0]]
    )
    weight1 = kstest(
        data_genre1[0:500, property_indices[1]], data_genre2[0:500, property_indices[1]]
    )
    weight2 = kstest(
        data_genre1[0:500, property_indices[2]], data_genre2[0:500, property_indices[2]]
    )
    return [weight0[0] * 1000, weight1[0] * 1000, weight2[0] * 1000]


def get_song_index(song_name: str):
    """This function find the index of a song within the database. It searches the database based on title and records the corresponding index once it finds it.

    Args:
        song_name (str): _description_
    """
    song_index = 200000  # initializing variable that holds the index of the song we are trying to predict the genre of
    for i in range(len(Song_names)):
        if song_name == Song_names[i]:
            song_index = i
            break
    if song_index == 200000:
        print(
            "Song name provided is not in the catalogued list so is not a valid input."
        )
    return song_index


def Techno_Or_Rap(song_name: str):
    # finding specific song within data set
    song_index = get_song_index(song_name)

    genre_likelihoods = np.zeros(
        len(genre_names)
    )  # genre_likelihoods is going to be an array that will record the likelihood that the chosen song is within each genre (note that it is not currently used)

    genre_likelihood = get_likelihood_technorap(Song_data, song_index)
    if genre_likelihood[0] > genre_likelihood[1]:
        return "rap"
    else:
        return "Techno"


def Emo_or_rap(song_name: str):
    # finding specific song within data set
    song_index = get_song_index(song_name)
    genre_likelihoods = np.zeros(
        len(genre_names)
    )  # genre_likelihoods is going to be an array that will record the likelihood that the chosen song is within each genre (note that it is not currently used)

    genre_likelihood = get_likelihood_emorap(Song_data, song_index)
    if genre_likelihood[0] > genre_likelihood[1]:
        return "rap"
    else:
        return "emo"


def get_likelihood(genre_data1: np.ndarray, genre_data2: np.ndarray):
    # calculating median values for each song property (for each genre)
    medians_1 = np.zeros(
        7
    )  # there are 7 relevant properties for each song (danceability, energy,...)
    medians_2 = np.zeros(7)
    property_indices = [0, 1, 5, 6, 7, 8, 9]

    for i, item in enumerate(property_indices):
        medians_1[i] = np.median(genre_data1[(len(genre_data1) // 2) :, item])
        medians_2[i] = np.median(genre_data2[(len(genre_data2) // 2) :, item])

    # next steps
    # 1. expand get_wights_kstest to all properties
    # 2. Create general likelihood function based on all properties
    # 3. Split the median code above into its own function
    # 4. test our new general genre comparison algorithm
