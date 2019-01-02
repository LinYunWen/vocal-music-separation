import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Input, multiply, concatenate

class GAN():
    def __init__(self):
        # build generator model
        self.generator = self.build_model('G')
        self.generator.compile(loss='mean_squared_error', optimizer='adam')

        # build discriminator model
        self.discriminator = self.build_model('D')
        self.discriminator.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        # build combined model
        self.combined = self.build_model('C')
        self.combined.compile(loss='mean_squared_error', optimizer='adam')


    def build_model(self, type):
        if type == 'G':
            inputs = Input(shape=(513, ))
            xs = Dense(1024, activation='relu')(inputs)
            xs = Dense(1024, activation='relu')(xs)
            xs = Dense(1024, activation='relu')(xs)
            xs = Dense(1026, activation='relu')(xs)
            masks = Dense(1026, activation='sigmoid')(xs)
            z = concatenate([inputs, inputs], axis=1)
            predicts = multiply([z, masks])
            return Model(inputs=inputs, outputs=predicts)
        elif type == 'D':
            inputs = Input(shape=(1539, ))
            xs = Dense(513, activation='relu')(inputs)
            xs = Dense(513, activation='relu')(xs)
            xs = Dense(513, activation='relu')(xs)
            validity = Dense(1, activation='sigmoid')(xs)
            return Model(inputs=inputs, outputs=validity)
        elif type == 'C':
            self.discriminator.trainable = False
            data = Input(shape=(513, ))
            tmp = self.generator(data)
            tmp = concatenate([tmp, data], axis=1)
            validity = self.discriminator(tmp)
            return Model(data, validity)


    def prepare_train_data(self, vocals_stfts, bgms_stfts, mixtures_stfts, batch_size):
        # dimension of stfts are (data num, 513(frequency), frame num)

        indexes = []
        train_data = []

        song_indexes = np.random.randint(0, len(mixtures_stfts), batch_size)
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

        return np.array(train_data)


    def train(self, vocals_stfts, bgms_stfts, mixtures_stfts, epochs, batch_size, sample_interval):
        # dimension of stfts are (data num, 513(frequency), frame num)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for i in range(epochs):
            # dimension of train_data are (batch, 1539)
            train_data = self.prepare_train_data(vocals_stfts, bgms_stfts, mixtures_stfts, batch_size)
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

            if i % 100 == 0:
                print(f'-- train epoch {i} --')

        return g_loss, d_loss


    def predict(self, mixtures_stfts):
        vocals_stfts_predict = []
        bgms_stfts_predict = []

        for mixture_stfts in mixtures_stfts:
            result = self.generator.predict(mixture_stfts.T)
            # dimension of result is (frame num, 1026)
            vocals_stfts_predict.append(result.T[: 513])
            bgms_stfts_predict.append(result.T[513: ])
        
        return vocals_stfts_predict, bgms_stfts_predict
