import numpy as np
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Input, multiply, concatenate, Lambda
from keras.optimizers import Adam
import tensorflow as tf

# do frequency mask
def mask(inputs):
    # cast y from float to complex, inputs[0] is real part, inputs[1] is imaginary part
    y = cast_complex([inputs[0], inputs[1]])
    z_real = inputs[2]
    z_imag = inputs[3]
    # split y into bgm and vocal
    bgm = y[:, 0:513]
    vocal = y[:, 513:1026]
    # calculate mask
    m1 = abs(bgm)/(abs(bgm)+abs(vocal))
    m2 = abs(vocal)/(abs(bgm)+abs(vocal))
    # do mask, calculate real part and imaginary part separately, then combine to complex number
    y1 = cast_complex([multiply([m1, z_real]), multiply([m1, z_imag])])
    y2 = cast_complex([multiply([m2, z_real]), multiply([m2, z_imag])])
    # return predicted bgm and vocal data
    result = concatenate([y1, y2])
    return result

# cast number to complex
def cast_complex(inputs):
    # inputs[0] for real part, inputs[1] for imaginary part
    return tf.dtypes.complex(inputs[0], inputs[1])

# combine predicted bgm, vocal with mixture data
def combine(inputs):
    # y complex number, size 1026
    y = inputs[0]
    # z float number, size 1026, first 513 stands for real part, others stands for imaginary part
    z = inputs[1]
    # split z into real and imaginary number
    z_real = z[:, 0:513]
    z_imag = z[:, 513:1026]
    # split y into real and imaginary number
    y_real = tf.math.real(y[:, 0:1026])
    y_imag = tf.math.imag(y[:, 0:1026])
    # concat, real numbers in front, imaginary numbers at back
    result = concatenate([y_real, z_real, y_imag, z_imag])
    return result

# calculate complex number absolute value
def calc_complex(inputs):
    real = inputs[0]
    imag = inputs[1]
    return tf.math.sqrt(real*real+imag*imag)

class GAN():
    def __init__(self):
        # set optimizer learning rate 0.0001
        optimizer = Adam(0.0001, 0.5)
        # build generator model
        self.generator = self.build_model('G')
        # generator's loss function must be absolute, because of result will be complex number
        self.generator.compile(loss='mean_absolute_error', optimizer=optimizer)

        # build discriminator model
        self.discriminator = self.build_model('D')
        self.discriminator.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['accuracy'])

        # build combined model
        self.combined = self.build_model('C')
        self.combined.compile(loss='mean_absolute_error', optimizer=optimizer)

    def build_model(self, type):
        if type == 'G':
            inputs = Input(shape=(513*2, ))
            inputs_real = Lambda(lambda inputs:inputs[:,0:513])(inputs)
            inputs_imag = Lambda(lambda inputs:inputs[:,513:1026])(inputs)
            # calculate real part
            xs = Dense(1024, activation='sigmoid')(inputs_real)
            xs = Dense(1024, activation='sigmoid')(xs)
            xs = Dense(1024, activation='sigmoid')(xs)
            y_tail_real = Dense(1026, activation='sigmoid')(xs)
            # calculate imaginary part
            xs = Dense(1024, activation='sigmoid')(inputs_imag)
            xs = Dense(1024, activation='sigmoid')(xs)
            xs = Dense(1024, activation='sigmoid')(xs)
            y_tail_imag = Dense(1026, activation='sigmoid')(xs)
            # combine real part and imaginary part, then do mask
            y = Lambda(mask)([y_tail_real, y_tail_imag, inputs_real, inputs_imag])
            return Model(inputs=inputs, outputs=y)
        elif type == 'D':
            inputs = Input(shape=(1539*2, ))
            inputs_real = Lambda(lambda inputs:inputs[:,0:1539])(inputs)
            inputs_imag = Lambda(lambda inputs:inputs[:,1539:3078])(inputs)
            # calculate real part
            xs = Dense(513, activation='sigmoid')(inputs_real)
            xs = Dense(513, activation='sigmoid')(xs)
            xs = Dense(513, activation='sigmoid')(xs)
            validity_real = Dense(1, activation='sigmoid')(xs)
            # calculate imaginary part
            xs = Dense(513, activation='sigmoid')(inputs_imag)
            xs = Dense(513, activation='sigmoid')(xs)
            xs = Dense(513, activation='sigmoid')(xs)
            validity_imag = Dense(1, activation='sigmoid')(xs)
            # combine real part and imaginary part, then calculate absolute value
            validity = Lambda(calc_complex)([validity_real, validity_imag])
            return Model(inputs=inputs, outputs=validity)
        elif type == 'C':
            self.discriminator.trainable = False
            data = Input(shape=(1026, ))
            tmp = self.generator(data)
            # combine predicted data with mixture data
            tmp = Lambda(combine)([tmp, data])
            validity = self.discriminator(tmp)
            return Model(data, validity)

    # pick batch_size song, then random pick a time frame, save the frequency value
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
            # save real part
            tmp += [bgms_stfts[idx][i][pick_time_frame].real for i in range(0, 513)]
            tmp += [vocals_stfts[idx][i][pick_time_frame].real for i in range(0, 513)]
            tmp += [mixtures_stfts[idx][i][pick_time_frame].real for i in range(0, 513)]
            # save imaginary part
            tmp += [bgms_stfts[idx][i][pick_time_frame].imag for i in range(0, 513)]
            tmp += [vocals_stfts[idx][i][pick_time_frame].imag for i in range(0, 513)]
            tmp += [mixtures_stfts[idx][i][pick_time_frame].imag for i in range(0, 513)]
            train_data.append(tmp)
            indexes.append((idx, pick_time_frame))

        return np.array(train_data)

    # supervised learning
    def pre_train(self, vocals_stfts, bgms_stfts, mixtures_stfts, epochs, batch_size, sample_interval):
        d_loss = []
        g_loss = []
        valid = np.ones((batch_size, 1))
        for i in range(epochs):
            # dimension of train_data are (batch, 3078)
            train_data = self.prepare_train_data(vocals_stfts, bgms_stfts, mixtures_stfts, batch_size)
            # train discriminator
            d_loss_real = self.discriminator.train_on_batch(train_data, valid)
            d_loss.append(d_loss_real)
            # train generator
            # get origin bgm, vocal, mixture datas from train_data
            mixture_data = np.concatenate((train_data[:, 1026:1539], train_data[:, 2565:3078]), axis=1)
            target_bgm = train_data[:, 0:513] + train_data[:, 1539:2052]*1j
            target_vocal = train_data[:, 513:1026] + train_data[:, 2052:2565]*1j
            # concat bgm, vocal data as target
            target = np.concatenate((target_bgm, target_vocal), axis=1)
            # predict, and calculate loss
            g_loss.append(self.generator.train_on_batch(mixture_data, target))
        print(g_loss)
        print(d_loss)
        return

    def train(self, vocals_stfts, bgms_stfts, mixtures_stfts, epochs, batch_size, sample_interval):
        # dimension of stfts are (data num, 513(frequency), frame num)
        d_loss = []
        g_loss = []
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        # supervised learning first
        self.pre_train(vocals_stfts, bgms_stfts, mixtures_stfts, int(epochs/10), batch_size, sample_interval)
        for i in range(epochs):
            # train D 10 times, G 1 time
            for j in range(0, 10):
                # dimension of train_data are (batch, 3078)
                train_data = self.prepare_train_data(vocals_stfts, bgms_stfts, mixtures_stfts, batch_size)
                noise = np.random.normal(-1, 1, (batch_size, 1026))
                # 32*1026
                gen_data = self.generator.predict(noise)
                gen_data = np.concatenate((gen_data.real, noise[:, 0:513], gen_data.imag, noise[:, 513:1026]), axis=1)
                # train discriminator
                d_loss_real = self.discriminator.train_on_batch(train_data, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_data, fake)
                d_loss.append(0.5 * np.add(d_loss_real, d_loss_fake))
            noise = np.random.normal(-1, 1, (batch_size, 1026))
            g_loss.append(self.combined.train_on_batch(noise, valid))
            if i % 100 == 0:
                # train generator
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
