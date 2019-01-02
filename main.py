import os, sys
import random
import librosa
import numpy as np
from GAN import GAN

args = sys.argv
epoch = 100 if '--epoch' not in args else int(args[args.index('--epoch') + 1])
data_num = 10 if '--data-num' not in args else int(args[args.index('--data-num') + 1])


def load_data(type):
    global data_num
    vocals = []
    bgms = []

    files = os.listdir('./datasets/MIR-1K_for_MIREX/Wavfile/')[:data_num]
    for filename in files:
        path = './datasets/MIR-1K_for_MIREX/Wavfile/' + filename

        # load audio files
        data, _ = librosa.load(path, sr=16000, mono=False)
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
    vocals_stfts = []
    bgms_stfts = []
    mixtures_stfts = []

    # mix vocals and BGMs together
    mixtures = vocals + bgms

    # do STFT
    for i in range(0, len(vocals)):
        vocals_stfts.append(do_stft(vocals[i], False))
        bgms_stfts.append(do_stft(bgms[i], False))
        mixtures_stfts.append(do_stft(mixtures[i], False))

    return mixtures, vocals_stfts, bgms_stfts, mixtures_stfts


def main():
    global epoch
    # load data
    vocals, bgms = load_data('train')

    # preprocess data (do stft), dimension (data*513*579)
    # mixtures_stfts 10(pick data)*513(frequency)*579(time frame)
    mixtures, vocals_stfts, bgms_stfts, mixtures_stfts = preprocess_data(vocals, bgms)

    # create model
    gan = GAN()

    # fit the model
    g_loss, d_loss = gan.train(vocals_stfts, bgms_stfts, mixtures_stfts,
        epochs=epoch, batch_size=32, sample_interval=200)
    print(f'g_loss: {g_loss}, d_loss: {d_loss}')

    # predict

    # verify

    return


if __name__ == '__main__':
    main()