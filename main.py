import os, sys
import random, time
import numpy as np
from GAN import GAN
from tools import load_data, do_stft, load_wav, save_wav, plot_data, plot_data_comp

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

args = sys.argv
epoch = 200 if '--epoch' not in args else int(args[args.index('--epoch') + 1])
data_num = 10 if '--data-num' not in args else int(args[args.index('--data-num') + 1])
is_suffle = False if '--shuffle' not in args else True


def create_result_directory():
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('results/vocals'):
        os.mkdir('results/vocals')
    if not os.path.exists('results/bgms'):
        os.mkdir('results/bgms')


def get_filenames():
    files = os.listdir('./datasets/MIR-1K_for_MIREX/Wavfile/')[:data_num]
    if is_suffle:
        random.shuffle(files)
    rate = 0.8
    num = int(len(files)*rate)
    filenames = {
        'train': files[: num],
        'test': files[num: ],
    }
    return filenames


def preprocess_data(vocals, bgms):
    vocals_stfts = []
    bgms_stfts = []
    mixtures_stfts = []

    # mix vocals and BGMs together
    mixtures = np.add(vocals, bgms)

    # do STFT
    for i in range(0, len(vocals)):
        vocals_stfts.append(do_stft(vocals[i], False))
        bgms_stfts.append(do_stft(bgms[i], False))
        mixtures_stfts.append(do_stft(mixtures[i], False))
    return mixtures, vocals_stfts, bgms_stfts, mixtures_stfts


def do_verify(vocals_stfts_test, vocals_stfts_predict, bgms_stfts_test, bgms_stfts_predict):
    # dimension of stfts_tests, stfts_predicts are (data num, 513, frame num)
    vocals_mse = []
    bgms_mse = []

    for i in range(0, len(vocals_stfts_predict)):
        num = vocals_stfts_predict[i].shape[0]
        vocal_mse = np.sum(np.mean(abs(vocals_stfts_test[i].T - vocals_stfts_predict[i].T)**2))
        bgm_mse = np.sum(np.mean(abs(bgms_stfts_test[i].T - bgms_stfts_predict[i].T)**2))
        vocals_mse.append(vocal_mse/num)
        bgms_mse.append(bgm_mse/num)

    plot_data_comp([vocals_mse, bgms_mse], ['Vocal MSE', 'BGM MSE'], True, 'mse', 2, 1)
    return vocals_mse, bgms_mse


def do_save_result(vocals_test, vocals_stfts_predict, bgms_test, bgms_stfts_predict):
    for i in range(0, len(vocals_stfts_predict)):
        vocal_wav_predict = do_stft(vocals_stfts_predict[i], True)
        save_wav(vocal_wav_predict, f'vocals/vocal-{i}')

        bgm_wav_predict = do_stft(bgms_stfts_predict[i], True)
        save_wav(bgm_wav_predict, f'bgms/bgm-{i}')

        plot_data_comp([vocals_test[i], bgms_test[i], vocal_wav_predict, bgm_wav_predict],
            ['True Vocal', 'True BGM', 'Predict Vocal', 'Predict BGM'],
            True, f'result-wav-{i}', 2, 2)


def main():
    global epoch
    filenames = get_filenames()

    # load data and preprocess data
    # dimension of mixtures_stfts are (data num, 513(frequency), frame num)
    vocals, bgms = load_data(filenames['train'])
    _, vocals_stfts, bgms_stfts, mixtures_stfts = preprocess_data(vocals, bgms)

    # create model
    gan = GAN()

    # fit the model
    g_loss, d_loss = gan.train(vocals_stfts, bgms_stfts, mixtures_stfts,
        epochs=epoch, batch_size=32, sample_interval=200)
    plot_data(g_loss, 'Generator Loss', True, 'g_loss')
    plot_data(d_loss, 'Discriminator Loss', True, 'd_loss')
    # print(f'g_loss: {g_loss}, d_loss: {d_loss}')

    # predict
    # dimension of stfts_tests, stfts_predicts are (data num, 513, frame num)
    vocals_test, bgms_test = load_data(filenames['test'])
    _, vocals_stfts_test, bgms_stfts_test, mixtures_stfts_test = preprocess_data(vocals_test, bgms_test)
    vocals_stfts_predict, bgms_stfts_predict = gan.predict(mixtures_stfts_test)
    # verify
    vocals_mse, bgms_mse = do_verify(vocals_stfts_test, vocals_stfts_predict, bgms_stfts_test, bgms_stfts_predict)

    # save result
    do_save_result(vocals_test, vocals_stfts_predict, bgms_test, bgms_stfts_predict)

    return


if __name__ == '__main__':
    create_result_directory()
    start = time.time()
    main()
    end = time.time()
    print(f'cost time: {end - start}')
