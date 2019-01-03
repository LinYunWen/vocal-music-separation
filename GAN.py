import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Input, multiply, concatenate, Lambda
from keras.optimizers import Adam
import tensorflow as tf

def mask(inputs):
    y = cast_complex([inputs[0], inputs[1]])
    z_real = inputs[2]
    z_imag = inputs[3]
    bgm = y[:, 0:513]
    vocal = y[:, 513:1026]
    m1 = abs(bgm)/(abs(bgm)+abs(vocal))
    m2 = abs(vocal)/(abs(bgm)+abs(vocal))
    y1 = cast_complex([multiply([m1, z_real]), multiply([m1, z_imag])])
    y2 = cast_complex([multiply([m2, z_real]), multiply([m2, z_imag])])
    result = concatenate([y1, y2])
    return result

def cast_complex(inputs):
    return tf.dtypes.complex(inputs[0], inputs[1])

def combine(inputs):
    # y complex size 1026
    y = inputs[0]
    # z real size 1026
    z = inputs[1]
    # split z
    z_real = z[:, 0:513]
    z_imag = z[:, 513:1026]
    # split y
    y_real = tf.math.real(y[:, 0:1026])
    y_imag = tf.math.imag(y[:, 0:1026])
    # comcat
    result = concatenate([y_real, z_real, y_imag, z_imag])
    return result

def calc_complex(inputs):
    real = inputs[0]
    imag = inputs[1]
    return tf.math.sqrt(real*real+imag*imag)

class GAN():
    def __init__(self):
        optimizer = Adam(0.0001, 0.5)
        # build generator model
        self.generator = self.build_model('G')
        self.generator.compile(loss='mean_absolute_error', optimizer=optimizer)

        # build discriminator model
        self.discriminator = self.build_model('D')
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # build combined model
        self.combined = self.build_model('C')
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_model(self, type):
        if type == 'G':
            inputs = Input(shape=(513*2, ))
            inputs_real = Lambda(lambda inputs:inputs[:,0:513])(inputs)
            inputs_imag = Lambda(lambda inputs:inputs[:,513:1026])(inputs)
            xs = Dense(1024, activation='sigmoid')(inputs_real)
            xs = Dense(1024, activation='sigmoid')(xs)
            xs = Dense(1024, activation='sigmoid')(xs)
            y_tail_real = Dense(1026, activation='sigmoid')(xs)
            xs = Dense(1024, activation='sigmoid')(inputs_imag)
            xs = Dense(1024, activation='sigmoid')(xs)
            xs = Dense(1024, activation='sigmoid')(xs)
            y_tail_imag = Dense(1026, activation='sigmoid')(xs)
            y = Lambda(mask)([y_tail_real, y_tail_imag, inputs_real, inputs_imag])
            return Model(inputs=inputs, outputs=y)
        elif type == 'D':
            inputs = Input(shape=(1539*2, ))
            inputs_real = Lambda(lambda inputs:inputs[:,0:1539])(inputs)
            inputs_imag = Lambda(lambda inputs:inputs[:,1539:3078])(inputs)
            xs = Dense(513, activation='sigmoid')(inputs_real)
            xs = Dense(513, activation='sigmoid')(xs)
            xs = Dense(513, activation='sigmoid')(xs)
            validity_real = Dense(1, activation='sigmoid')(xs)
            xs = Dense(513, activation='sigmoid')(inputs_imag)
            xs = Dense(513, activation='sigmoid')(xs)
            xs = Dense(513, activation='sigmoid')(xs)
            validity_imag = Dense(1, activation='sigmoid')(xs)
            validity = Lambda(calc_complex)([validity_real, validity_imag])
            return Model(inputs=inputs, outputs=validity)
        elif type == 'C':
            self.discriminator.trainable = False
            data = Input(shape=(1026, ))
            tmp = self.generator(data)
            tmp = Lambda(combine)([tmp, data])
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
            tmp += [bgms_stfts[idx][i][pick_time_frame].real for i in range(0, 513)]
            tmp += [vocals_stfts[idx][i][pick_time_frame].real for i in range(0, 513)]
            tmp += [mixtures_stfts[idx][i][pick_time_frame].real for i in range(0, 513)]
            tmp += [bgms_stfts[idx][i][pick_time_frame].imag for i in range(0, 513)]
            tmp += [vocals_stfts[idx][i][pick_time_frame].imag for i in range(0, 513)]
            tmp += [mixtures_stfts[idx][i][pick_time_frame].imag for i in range(0, 513)]
            train_data.append(tmp)
            indexes.append((idx, pick_time_frame))

        return np.array(train_data)


    def train(self, vocals_stfts, bgms_stfts, mixtures_stfts, epochs, batch_size, sample_interval):
        # dimension of stfts are (data num, 513(frequency), frame num)
        d_loss = []
        g_loss = []
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for i in range(epochs):
            # dimension of train_data are (batch, 1539)
            train_data = self.prepare_train_data(vocals_stfts, bgms_stfts, mixtures_stfts, batch_size)
            noise = np.random.normal(-1, 1, (batch_size, 1026))
            # 32*1026
            gen_data = self.generator.predict(noise)
            gen_data = np.concatenate((gen_data.real, noise[:, 0:513], gen_data.imag, noise[:, 513:1026]), axis=1)
            # train discriminator
            d_loss_real = self.discriminator.train_on_batch(train_data, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_data, fake)
            d_loss.append(0.5 * np.add(d_loss_real, d_loss_fake))
            if i % 10 == 0:
                # train generator
                noise = np.random.normal(-1, 1, (batch_size, 1026))
                g_loss.append(self.combined.train_on_batch(noise, valid))
                print(f'-- train epoch {i} --')

        return g_loss, d_loss


    def predict(self, mixtures_stfts):
        vocals_stfts_predict = []
        bgms_stfts_predict = []

        for mixture_stfts in mixtures_stfts:
            mixture_stfts = mixture_stfts.T
            mixture_stfts = np.concatenate((mixture_stfts.real, mixture_stfts.imag), axis=1)
            result = self.generator.predict(mixture_stfts)
            # dimension of result is (frame num, 1026)
            vocals_stfts_predict.append(result.T[: 513])
            bgms_stfts_predict.append(result.T[513: ])
        
        return vocals_stfts_predict, bgms_stfts_predict
