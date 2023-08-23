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


# %% Necessary functions
def get_moments(genre_property: np.ndarray):
    """This function will calculate the moments (mean, variance, skewness,kurtosis) for a given genre property.

    Args:
        genre_property (np.ndarray): A 1D array of all of the properties (e.g. energy) associated with songs within a given genre (e.g. hiphop)

    returns:
        a list of floats that describes the statistical moments of a given property within a genre
    """

    mean = np.mean(genre_property)
    variance = np.var(genre_property)
    skewness = scipy.stats.skew(genre_property)
    kurtosis = scipy.stats.kurtosis(genre_property)
    return [mean, variance, skewness, kurtosis]


def calc_all_moments(genre_names: list, Genre_data: list):
    """This function is used to get the moments for a given property across every genre. It calls the get_moments function.

    Args:
        genre_names (list): list of strings with all genre_names
        Genre_data (list): list of 2D arrays. Each array within the list holds all of the property data within a single genre
    """
    # Initializing arrays that will store the 4 moments (mean, varoance, skew, kurtosis) for energy, acoustic, speech, instrument, for every genre
    energy_moments = np.zeros((len(genre_names), 4))
    acoustic_moments = np.zeros((len(genre_names), 4))
    speech_moments = np.zeros((len(genre_names), 4))
    instrument_moments = np.zeros((len(genre_names), 4))
    dance_moments = np.zeros((len(genre_names), 4))
    live_moments = np.zeros((len(genre_names), 4))
    valence_moments = np.zeros((len(genre_names), 4))
    tempo_moments = np.zeros((len(genre_names), 4))

    # Looping over every genre
    for i in range(len(genre_names)):
        # gathering specific genre data (i.e. Rap's data set) from Genre_data (list of arrays)
        specific_genre_data = Genre_data[i]

        # Calculating the moments for energy, acoustic, speech, instrument
        energy_moments[i, :] = get_moments(specific_genre_data[:, 1])
        acoustic_moments[i, :] = get_moments(specific_genre_data[:, 6])
        speech_moments[i, :] = get_moments(specific_genre_data[:, 5])
        instrument_moments[i, :] = get_moments(specific_genre_data[:, 7])
        dance_moments[i, :] = get_moments(specific_genre_data[:, 0])
        live_moments[i, :] = get_moments(specific_genre_data[:, 8])
        valence_moments[i, :] = get_moments(specific_genre_data[:, 9])
        tempo_moments[i, :] = get_moments(specific_genre_data[:, 10])
    return (
        energy_moments,
        acoustic_moments,
        speech_moments,
        instrument_moments,
        dance_moments,
        live_moments,
        valence_moments,
        tempo_moments,
    )


(
    energy_moments,
    acoustic_moments,
    speech_moments,
    instrument_moments,
    dance_moments,
    live_moments,
    valence_moments,
    tempo_moments,
) = calc_all_moments(genre_names, Genre_data)


def make_average_plot(plot_desired: bool, stats_displayed: str):
    """_summary_

    Args:
        plot_desired (bool): can be True or False to determine if a plot is made or not
        stats_displayed (str): can be "energy" or "dance" to determine the suite of statistics plotted
    """
    if plot_desired == True:
        x_pos = np.arange(len(genre_names))

        # plotting
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=[13, 8])
        fig.subplots_adjust(hspace=0.5)

        ax1.title.set_text("Property Averages Across Genres")

        # determining suite of statistics plotted
        if stats_displayed == "energy":
            # ax1.figure(1, figsize=[15, 5])
            ax1.bar(x_pos, energy_moments[:, 0])
            ax1.set_xticks(x_pos, genre_names)
            ax1.set_ylabel("Energy")

            # ax2.figure(2, figsize=[15, 5])
            ax2.bar(x_pos, acoustic_moments[:, 0])
            ax2.set_xticks(x_pos, genre_names)
            ax2.set_ylabel("Acousticness")

            ax3.bar(x_pos, speech_moments[:, 0])
            ax3.set_xticks(x_pos, genre_names)
            ax3.set_ylabel("Speechiness")

            ax4.bar(x_pos, instrument_moments[:, 0])
            ax4.set_xticks(x_pos, genre_names)
            ax4.set_ylabel("Instrumentalness")

        if stats_displayed == "dance":
            # ax1.figure(1, figsize=[15, 5])
            ax1.bar(x_pos, dance_moments[:, 0])
            ax1.set_xticks(x_pos, genre_names)
            ax1.set_ylabel("Danceability")

            # ax2.figure(2, figsize=[15, 5])
            ax2.bar(x_pos, live_moments[:, 0])
            ax2.set_xticks(x_pos, genre_names)
            ax2.set_ylabel("Liveness")

            ax3.bar(x_pos, valence_moments[:, 0])
            ax3.set_xticks(x_pos, genre_names)
            ax3.set_ylabel("Valence")

            ax4.bar(x_pos, tempo_moments[:, 0])
            ax4.set_xticks(x_pos, genre_names)
            ax4.set_ylabel("Tempo")

        plt.show()


def make_variance_plot(plot_desired: bool, stats_displayed: str):
    if plot_desired == True:
        x_pos = np.arange(len(genre_names))

        # plotting
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=[13, 8])
        fig.subplots_adjust(hspace=0.5)

        ax1.title.set_text("Property Variances Across Genres")

        # ax1.figure(1, figsize=[15, 5])
        if stats_displayed == "energy":
            ax1.bar(x_pos, energy_moments[:, 1])
            ax1.set_xticks(x_pos, genre_names)
            ax1.set_ylabel("Energy")

            # ax2.figure(2, figsize=[15, 5])
            ax2.bar(x_pos, acoustic_moments[:, 1])
            ax2.set_xticks(x_pos, genre_names)
            ax2.set_ylabel("Acousticness")

            ax3.bar(x_pos, speech_moments[:, 1])
            ax3.set_xticks(x_pos, genre_names)
            ax3.set_ylabel("Speechiness")

            ax4.bar(x_pos, instrument_moments[:, 1])
            ax4.set_xticks(x_pos, genre_names)
            ax4.set_ylabel("Instrumentalness")

        if stats_displayed == "dance":
            # ax1.figure(1, figsize=[15, 5])
            ax1.bar(x_pos, dance_moments[:, 1])
            ax1.set_xticks(x_pos, genre_names)
            ax1.set_ylabel("Danceability")

            # ax2.figure(2, figsize=[15, 5])
            ax2.bar(x_pos, live_moments[:, 1])
            ax2.set_xticks(x_pos, genre_names)
            ax2.set_ylabel("Liveness")

            ax3.bar(x_pos, valence_moments[:, 1])
            ax3.set_xticks(x_pos, genre_names)
            ax3.set_ylabel("Valence")

            ax4.bar(x_pos, tempo_moments[:, 1])
            ax4.set_xticks(x_pos, genre_names)
            ax4.set_ylabel("Tempo")

        plt.show()


def make_skewness_plot(plot_desired: bool, stats_displayed: str):
    if plot_desired == True:
        x_pos = np.arange(len(genre_names))

        # plotting
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=[13, 8])
        fig.subplots_adjust(hspace=0.5)

        ax1.title.set_text("Property Skewness' Across Genres")

        # ax1.figure(1, figsize=[15, 5])
        if stats_displayed == "energy":
            ax1.bar(x_pos, energy_moments[:, 2])
            ax1.set_xticks(x_pos, genre_names)
            ax1.set_ylabel("Energy")

            # ax2.figure(2, figsize=[15, 5])
            ax2.bar(x_pos, acoustic_moments[:, 2])
            ax2.set_xticks(x_pos, genre_names)
            ax2.set_ylabel("Acousticness")

            ax3.bar(x_pos, speech_moments[:, 2])
            ax3.set_xticks(x_pos, genre_names)
            ax3.set_ylabel("Speechiness")

            ax4.bar(x_pos, instrument_moments[:, 2])
            ax4.set_xticks(x_pos, genre_names)
            ax4.set_ylabel("Instrumentalness")

        if stats_displayed == "dance":
            # ax1.figure(1, figsize=[15, 5])
            ax1.bar(x_pos, dance_moments[:, 2])
            ax1.set_xticks(x_pos, genre_names)
            ax1.set_ylabel("Danceability")

            # ax2.figure(2, figsize=[15, 5])
            ax2.bar(x_pos, live_moments[:, 2])
            ax2.set_xticks(x_pos, genre_names)
            ax2.set_ylabel("Liveness")

            ax3.bar(x_pos, valence_moments[:, 2])
            ax3.set_xticks(x_pos, genre_names)
            ax3.set_ylabel("Valence")

            ax4.bar(x_pos, tempo_moments[:, 2])
            ax4.set_xticks(x_pos, genre_names)
            ax4.set_ylabel("Tempo")

        plt.show()


def make_kurtosis_plot(plot_desired: bool, stats_displayed: str):
    if plot_desired == True:
        x_pos = np.arange(len(genre_names))

        # plotting
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=[13, 8])
        fig.subplots_adjust(hspace=0.5)

        ax1.title.set_text("Property Kurtosis' Across Genres")

        # ax1.figure(1, figsize=[15, 5])
        if stats_displayed == "energy":
            ax1.bar(x_pos, energy_moments[:, 3])
            ax1.set_xticks(x_pos, genre_names)
            ax1.set_ylabel("Energy")

            # ax2.figure(2, figsize=[15, 5])
            ax2.bar(x_pos, acoustic_moments[:, 3])
            ax2.set_xticks(x_pos, genre_names)
            ax2.set_ylabel("Acousticness")

            ax3.bar(x_pos, speech_moments[:, 3])
            ax3.set_xticks(x_pos, genre_names)
            ax3.set_ylabel("Speechiness")

            ax4.bar(x_pos, instrument_moments[:, 3])
            ax4.set_xticks(x_pos, genre_names)
            ax4.set_ylabel("Instrumentalness")

        if stats_displayed == "dance":
            # ax1.figure(1, figsize=[15, 5])
            ax1.bar(x_pos, dance_moments[:, 3])
            ax1.set_xticks(x_pos, genre_names)
            ax1.set_ylabel("Danceability")

            # ax2.figure(2, figsize=[15, 5])
            ax2.bar(x_pos, live_moments[:, 3])
            ax2.set_xticks(x_pos, genre_names)
            ax2.set_ylabel("Liveness")

            ax3.bar(x_pos, valence_moments[:, 3])
            ax3.set_xticks(x_pos, genre_names)
            ax3.set_ylabel("Valence")

            ax4.bar(x_pos, tempo_moments[:, 3])
            ax4.set_xticks(x_pos, genre_names)
            ax4.set_ylabel("Tempo")

        plt.show()


make_average_plot(True, "dance")
make_variance_plot(True, "dance")
make_skewness_plot(True, "dance")
make_kurtosis_plot(True, "dance")
