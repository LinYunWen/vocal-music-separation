import librosa
import matplotlib.pyplot as plt

plt.subplots_adjust(hspace=0.5)


def load_data(filenames):
    vocals = []
    bgms = []

    for filename in filenames:
        path = './datasets/MIR-1K_for_MIREX/Wavfile/' + filename

        # load audio files
        bgm, vocal = load_wav(path)
        vocals.append(vocal)
        bgms.append(bgm)

    return vocals, bgms


def do_stft(data, is_invert=False):
    if is_invert:
        return librosa.istft(data, win_length=1024, hop_length=256)
    else:
        return librosa.stft(data, n_fft=1024, hop_length=256)


def load_wav(path):
    data, _ = librosa.load(path, sr=16000, mono=False)
    # according to left and right, separate data into vocal and BGM
    return data[0], data[1]


def save_wav(wav, filename):
    librosa.output.write_wav(f'results/{filename}.wav', wav, 16000)


def plot_data(data, title, is_save=False, filename=None):
    plt.plot(data)
    plt.title(title)

    if is_save and filename is not None:
        plt.savefig(f'results/{filename}.png')
    else:
        plt.show()
    plt.gcf().clear()


def plot_data_comp(data, titles, is_save=False, filename=None, row=1, col=1):
    if len(data) != row*col:
        print('error on plot')
        return

    for i in range(0, row*col):
        plt.subplot(row, col, i + 1)
        plt.plot(data[i])
        plt.title(titles[i])

    if is_save and filename is not None:
        plt.savefig(f'results/{filename}.png')
    else:
        plt.show()
    plt.gcf().clear()