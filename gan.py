import os, sys
import random
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
        targets.append(np.concatenate((vocals_stfts[i], bgms_stfts[i]), axis=0))
    return targets


def create_D_data(predicts, targets, mixtures):
    size = 2
    index = random.sample(range(0, size), int(size/2))
    inputs = []
    outputs = []
    for i in range(0, size):
        if i in index:
            temp = predicts[i]
            output = np.zeros(temp.shape[1], dtype=int)
        else:
            temp = targets[i]
            output = np.ones(temp.shape[1], dtype=int)

        inputs.append(np.concatenate((temp, mixtures[i]), axis=0))
        outputs.append(output)
    return inputs, outputs


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
    g_predicts = [
        np.concatenate((mixtures_stfts[0], mixtures_stfts[0]), axis=0),
        np.concatenate((vocals_stfts[1], bgms_stfts[1]), axis=0),
    ]

    # build generator model
    generator = build_model('G')

    # fit the model
    generator.fit(mixtures_stfts[0].T, targets[0].T, epochs=10, batch_size=10)

    # predict
    g_predicts = generator.predict(mixtures_stfts[0].T)
    print(g_predicts.shape)
    return

    # build discriminator model
    d_inputs, d_targets = create_D_data(g_predicts, targets, mixtures_stfts)
    discriminator = build_model('D')
    discriminator.fit(d_inputs[0].T, d_targets[0], epochs=10, batch_size=10)
    d_predict = discriminator.predict(d_inputs[0].T)
    print(d_predict)
    return


if __name__ == '__main__':
    main()