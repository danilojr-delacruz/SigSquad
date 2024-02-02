import numpy as np
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


def generate_non_overlapping_data(df):
    """eeg_df is train.csv read in.
    This modifies the train_csv (metadata) so that we only have one segment per eeg_id.
    https://www.kaggle.com/code/yorkyong/exploring-eeg-a-beginner-s-guide
    """
    TARGETS = df.columns[-6:]
    # Creating a Unique EEG Segment per eeg_id:
    # The code groups (groupby) the EEG data (df) by eeg_id. Each eeg_id represents a different EEG recording.
    # It then picks the first spectrogram_id and the earliest (min) spectrogram_label_offset_seconds for each eeg_id. This helps in identifying the starting point of each EEG segment.
    # The resulting DataFrame train has columns spec_id (first spectrogram_id) and min (earliest spectrogram_label_offset_seconds).
    train = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
        {'spectrogram_id':'first','spectrogram_label_offset_seconds':'min'})
    train.columns = ['spec_id','min']


    # Finding the Latest Point in Each EEG Segment:
    # The code again groups the data by eeg_id and finds the latest (max) spectrogram_label_offset_seconds for each segment.
    # This max value is added to the train DataFrame, representing the end point of each EEG segment.
    tmp = df.groupby('eeg_id')[['spectrogram_id','spectrogram_label_offset_seconds']].agg(
        {'spectrogram_label_offset_seconds':'max'})
    train['max'] = tmp


    tmp = df.groupby('eeg_id')[['patient_id']].agg('first') # The code adds the patient_id for each eeg_id to the train DataFrame. This links each EEG segment to a specific patient.
    train['patient_id'] = tmp


    tmp = df.groupby('eeg_id')[TARGETS].agg('sum') # The code sums up the target variable counts (like votes for seizure, LPD, etc.) for each eeg_id.
    for t in TARGETS:
        train[t] = tmp[t].values

    y_data = train[TARGETS].values # It then normalizes these counts so that they sum up to 1. This step converts the counts into probabilities, which is a common practice in classification tasks.
    y_data = y_data / y_data.sum(axis=1,keepdims=True)
    train[TARGETS] = y_data

    tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first') # For each eeg_id, the code includes the expert_consensus on the EEG segment's classification.
    train['target'] = tmp

    train = train.reset_index() # This makes eeg_id a regular column, making the DataFrame easier to work with.
    print('Train non-overlapp eeg_id shape:', train.shape )
    train.head()

    return train
