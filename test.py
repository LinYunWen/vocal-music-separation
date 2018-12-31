import os, sys
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate
import librosa
import numpy as np

def load_data():
    vocals = []
    bgms = []

    files = os.listdir('./datasets/MIR-1K_for_MIREX/Wavfile/')[:10]
    for filename in files:
        path = './datasets/MIR-1K_for_MIREX/Wavfile/' + filename

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

class GAN():
    def __init__(self):
        # build generator model
        self.generator = self.build_model('G')
        self.generator.compile(loss='mean_squared_error', optimizer='adam')
        #self.generator.compile(loss='binary_crossentropy', optimizer='adam')
        # build discriminator model
        self.discriminator = self.build_model('D')
        self.discriminator.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        #self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # build combined model
        self.combined = self.build_model('C')
        self.combined.compile(loss='mean_squared_error', optimizer='adam')
        #self.combined.compile(loss='binary_crossentropy', optimizer='adam')

    def build_model(self, type):
        model = Sequential()
        if type == 'G':
            model.add(Dense(1024, input_dim=513, activation='relu'))
            model.add(Dense(1024, activation='relu'))
            model.add(Dense(1024, activation='relu'))
            model.add(Dense(1026, activation='relu'))
            model.add(Dense(1026, activation='sigmoid'))
            data = Input(shape=(513, ))
            output = model(data)
            return Model(data, output)
        elif type == 'D':
            model.add(Dense(513, input_dim=1539, activation='relu'))
            model.add(Dense(513, activation='relu'))
            model.add(Dense(513, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            data = Input(shape=(1539, ))
            validity = model(data)
            return Model(data, validity)
        elif type == 'C':
            self.discriminator.trainable = False
            data = Input(shape=(513, ))
            tmp = self.generator(data)
            tmp = concatenate([tmp, data], axis=1)
            validity = self.discriminator(tmp)
            return Model(data, validity)

    def train(self, epochs, batch_size, sample_interval):
        # load data
        vocals, bgms = load_data()
        # preprocess data(do stft), dimension (data*513*579)
        mixtures, vocals_stfts, bgms_stfts, mixtures_stfts = preprocess_data(vocals, bgms)
        # mixtures_stfts 10(pick data)*513(frequency)*579(time frame)
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            song_indexes = np.random.randint(0, len(mixtures_stfts), batch_size)
            indexes = []
            train_data = []
            for idx in song_indexes:
                # find how many time frames in song[idx]
                time_frames = len(mixtures_stfts[idx][0])
                # random pick a time frame
                pick_time_frame = random.randint(0, time_frames-1)
                tmp = []
                # save bgm, vocal, mixture in that time frame of the song
                tmp += [bgms_stfts[idx][i][pick_time_frame] for i in range(0, 513)]
                tmp += [vocals_stfts[idx][i][pick_time_frame] for i in range(0, 513)]
                tmp += [mixtures_stfts[idx][i][pick_time_frame] for i in range(0, 513)]
                train_data.append(tmp)
                indexes.append((idx, pick_time_frame))

            train_data = np.array(train_data)
            noise = np.random.normal(-1, 1, (batch_size, 513))
            gen_data = self.generator.predict(noise)
            gen_data = np.concatenate((gen_data, noise), axis=1)
            # train discriminator
            d_loss_real = self.discriminator.train_on_batch(train_data, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # train generator
            noise = np.random.normal(-1, 1, (batch_size, 513))
            g_loss = self.combined.train_on_batch(noise, valid)
            print 'g_loss: '+str(g_loss)+', d_loss: '+str(d_loss)

def main():
    gan = GAN()
    gan.train(epochs=100, batch_size=32, sample_interval=200)
    return

if __name__ == '__main__':
    main()
