import pandas as pd
import matplotlib.pyplot as plt


def get_entry_data(GET_ROW=0):
    """Get the data corresponding to the metadata on the row specified.
    https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468010
    """

    EEG_PATH = 'train_eegs/'
    SPEC_PATH = 'train_spectrograms/'

    train = pd.read_csv('train.csv')
    row = train.iloc[GET_ROW]

    eeg = pd.read_parquet(f'{EEG_PATH}{row.eeg_id}.parquet')
    eeg_offset = int( row.eeg_label_offset_seconds )
    # The sampling frequency is 200 per second
    eeg = eeg.iloc[eeg_offset*200:(eeg_offset+50)*200]

    spectrogram = pd.read_parquet(f'{SPEC_PATH}{row.spectrogram_id}.parquet')
    spec_offset = int( row.spectrogram_label_offset_seconds )
    # The time is recorded here already.
    spectrogram = spectrogram.loc[(spectrogram.time>=spec_offset)
                        &(spectrogram.time<spec_offset+600)]

    return eeg, spectrogram


def plot_spectrogram(spectrogram_path: str):
    """
    Source: https://www.kaggle.com/code/mvvppp/hms-eda-and-domain-journey
    Visualize spectogram recordings from a parquet file.
    :param spectrogram_path: path to the spectogram parquet.
    """
    sample_spect = pd.read_parquet(spectrogram_path)

    split_spect = {
        "LL": sample_spect.filter(regex='^LL', axis=1),
        "RL": sample_spect.filter(regex='^RL', axis=1),
        "RP": sample_spect.filter(regex='^RP', axis=1),
        "LP": sample_spect.filter(regex='^LP', axis=1),
    }

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    axes = axes.flatten()
    label_interval = 5
    for i, split_name in enumerate(split_spect.keys()):
        ax = axes[i]
        img = ax.imshow(np.log(split_spect[split_name]).T, cmap='viridis', aspect='auto', origin='lower')
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('Log(Value)')
        ax.set_title(split_name)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time")

        ax.set_yticks(np.arange(len(split_spect[split_name].columns)))
        ax.set_yticklabels([column_name[3:] for column_name in split_spect[split_name].columns])
        frequencies = [column_name[3:] for column_name in split_spect[split_name].columns]
        ax.set_yticks(np.arange(0, len(split_spect[split_name].columns), label_interval))
        ax.set_yticklabels(frequencies[::label_interval])
    plt.tight_layout()
    plt.show()
