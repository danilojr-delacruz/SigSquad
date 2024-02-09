# 4. Test Model ---------------------------------------------------------------
# Infer test and crate submission
del all_eegs, spectrograms; gc.collect()
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape",test.shape)
test.head()


# READ ALL SPECTROGRAMS
PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
files2 = os.listdir(PATH2)
print(f"There are {len(files2)} test spectrogram parquets")

spectrograms2 = {}
for i,f in enumerate(files2):
    if i%100==0: print(i,", ",end="")
    tmp = pd.read_parquet(f"{PATH2}{f}")
    name = int(f.split(".")[0])
    spectrograms2[name] = tmp.iloc[:,1:].values

# RENAME FOR DATALOADER
test = test.rename({"spectrogram_id":"spec_id"},axis=1)


import pywt, librosa

USE_WAVELET = None

NAMES = ["LL","LP","RP","RR"]

FEATS = [["Fp1","F7","T3","T5","O1"],
         ["Fp1","F3","C3","P3","O1"],
         ["Fp2","F8","T4","T6","O2"],
         ["Fp2","F4","C4","P4","O2"]]

# DENOISE FUNCTION
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet="haar", level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])

    ret=pywt.waverec(coeff, wavelet, mode="per")

    return ret

def spectrogram_from_eeg(parquet_path, display=False):

    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]

    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128,256,4),dtype="float32")

    if display: plt.figure(figsize=(10,7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):

            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0

            # DENOISE
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256,
                  n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

            # LOG TRANSFORM
            width = (mel_spec.shape[1]//32)*32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db+40)/40
            img[:,:,k] += mel_spec_db

        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0

        if display:
            plt.subplot(2,2,k+1)
            plt.imshow(img[:,:,k],aspect="auto",origin="lower")
            plt.title(f"EEG {eeg_id} - Spectrogram {NAMES[k]}")

    if display:
        plt.show()
        plt.figure(figsize=(10,5))
        offset = 0
        for k in range(4):
            if k>0: offset -= signals[3-k].min()
            plt.plot(range(10_000),signals[k]+offset,label=NAMES[3-k])
            offset += signals[3-k].max()
        plt.legend()
        plt.title(f"EEG {eeg_id} Signals")
        plt.show()
        print(); print("#"*25); print()

    return img


# READ ALL EEG SPECTROGRAMS
PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
DISPLAY = 1
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print("Converting Test EEG to Spectrograms..."); print()
for i,eeg_id in enumerate(EEG_IDS2):

    # CREATE SPECTROGRAM FROM EEG PARQUET
    img = spectrogram_from_eeg(f"{PATH2}{eeg_id}.parquet", i<DISPLAY)
    all_eegs2[eeg_id] = img


# INFER EFFICIENTNET ON TEST
preds = []
model = build_model()
#TODO: Should have a different DataGenerator for test
test_gen = PreloadedDataset(test, spectrograms2, all_eegs2,
                         shuffle=False, batch_size=64, mode="test")

for i in range(5):
    print(f"Fold {i+1}")
    if LOAD_MODELS_FROM:
        model.load_weights(f"{LOAD_MODELS_FROM}EffNet_v{VER}_f{i}.h5")
    else:
        model.load_weights(f"EffNet_v{VER}_f{i}.h5")
    pred = model.predict(test_gen, verbose=1)
    preds.append(pred)
pred = np.mean(preds,axis=0)
print()
print("Test preds shape",pred.shape)


sub = pd.DataFrame({"eeg_id":test.eeg_id.values})
sub[TARGETS] = pred
sub.to_csv("submission.csv",index=False)
print("Submissionn shape",sub.shape)
sub.head()


# SANITY CHECK TO CONFIRM PREDICTIONS SUM TO ONE
sub.iloc[:,-6:].sum(axis=1)