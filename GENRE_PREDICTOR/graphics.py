import matplotlib.pyplot as plt
import numpy as np
from importing import get_genre_names
from algorithms import get_avgs

all_avg_metrics = get_avgs()

genre_names = get_genre_names()


energy_avgs = all_avg_metrics[:, 1]
acoustic_avgs = all_avg_metrics[:, 6]
speech_avgs = all_avg_metrics[:, 5]
instrument_avgs = all_avg_metrics[:, 7]

#%% Graphics
def make_genre_comparison_plot(plot_desired: bool):
    if plot_desired == True:
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
        plt.show()

