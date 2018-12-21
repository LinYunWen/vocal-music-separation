import os, sys
from keras.models import Sequential
from keras.layers import Dense
from scipy import signal
from scipy.io import wavfile
import numpy


def load_data(type):
    filename = '' if type == 'train' else  ''
    # load audio files
    fs, data = wavfile.read(filename)

    # according to left and right, separate data into vocal and BGM
    vocals = data[0]
    bgms = data[1]
    return vocals, bgms


def do_stft(data):
    return scipy.signal.stft(data, fs=16000)


def preprocess_data(vocals, bgms):
    # mix vocals and BGMs together
    mix = vocals + bgms

    # do STFT
    vocals_stft = do_stft(vocals)
    bgms_stft = do_stft(bgms)
    mix_stft = do_stft(mix)

    return mix, vocals_stft, bgms_stft, mix_stft


def build_model(type):
    model = Sequential()
    if type == 'G':
        model.add(Dense(1024, input_dim=256, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(1024, activation='sigmoid'))
    elif type == 'D':
        model.add(Dense(12, input_dim=8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def main():
    # fix random seed for reproducibility
    numpy.random.seed(7)

    # load data
    vocals, bgms = load_data('tarin')
    # preprocess data
    mix, vocals_stft, bgms_stft, mix_stft = preprocess_data(vocals, bgms)

    # build model
    generator = build_model('G')
    discriminator = build_model('D')

    # fit the model
    generator.fit(mix_stft, , epochs=150, batch_size=10)

    return


if __name__ != '__main__':
    main()