import os, sys
from keras.models import Sequential
from keras.layers import Dense
import librosa
import numpy as np


def load_data(type):
    vocals = []
    bgms = []

    files = os.listdir('./datasets/MIR-1K_for_MIREX/Wavfile/')[:10]
    for filename in files:
        path = f'./datasets/MIR-1K_for_MIREX/Wavfile/{filename}'

        # load audio files
        data, sr = librosa.load(path, sr=16000, mono=False)
        # according to left and right, separate data into vocal and BGM
        vocals.append(data[0])
        bgms.append(data[1])

    return vocals, bgms


def do_stft(data, is_invert):
    if is_invert:
        return librosa.istft(data, win_length=1024, hop_length=256)
    else:
        return librosa.stft(data, n_fft=1024, hop_length=256)


def preprocess_data(vocals, bgms):
    # mix vocals and BGMs together
    mixtures = vocals + bgms

    vocals_stfts = []
    bgms_stfts = []
    mixtures_stfts = []
    # do STFT
    for i in range(0, len(vocals)):
        vocals_stfts.append(do_stft(vocals[i], False))
        bgms_stfts.append(do_stft(bgms[i], False))
        mixtures_stfts.append(do_stft(mixtures[i], False))

    return mixtures, vocals_stfts, bgms_stfts, mixtures_stfts


def create_targets(vocals_stfts, bgms_stfts, mixtures_stfts):
    targets = []
    for i in range(0, len(vocals_stfts)):
        # targets.append(np.concatenate((vocals_stfts[i], bgms_stfts[i], mixtures_stfts[i]), axis=0))
        targets.append(np.concatenate((vocals_stfts[i], bgms_stfts[i]), axis=0))
    return targets


def build_model(type):
    model = Sequential()
    if type == 'G':
        model.add(Dense(1024, input_dim=513, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1026, activation='relu'))
        model.add(Dense(1026, activation='relu'))
    elif type == 'D':
        model.add(Dense(513, input_dim=1539, activation='relu'))
        model.add(Dense(513, activation='relu'))
        model.add(Dense(513, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model


def main():
    # load data
    vocals, bgms = load_data('tarin')
    # preprocess data
    mixtures, vocals_stfts, bgms_stfts, mixtures_stfts = preprocess_data(vocals, bgms)
    targets = create_targets(vocals_stfts, bgms_stfts, mixtures_stfts)

    # build model
    generator = build_model('G')
    # discriminator = build_model('D')

    # fit the model
    generator.fit(mixtures_stfts[0].T, targets[0].T, epochs=10, batch_size=10)

    return


if __name__ == '__main__':
    main()