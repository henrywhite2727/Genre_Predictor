#%% This class will be used to extract metrics from the source data
class Genre:
    def __init__(
        self,
        energy,
        key,
        loudness,
        mode,
        speech,
        acoustic,
        instrument,
        live,
        valence,
        tempo,
    ):
        self.energy = energy
        self.key = key
        self.loudness = loudness


data = Genre(5, 6)

print(data.energy)
