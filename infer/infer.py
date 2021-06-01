import tensorflow as tf
import functools
from tensorflow.python.ops.signal import window_ops
from scipy.special import exp1
from scipy.io import loadmat, savemat
import sys
import os
import soundfile
import shutil
import glob

from network import ResNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

frame_size = 512
frame_shift = 256
fft_size = 512
W = functools.partial(window_ops.hamming_window, periodic=False)
n_feat = fft_size // 2 + 1


def analysis(x):
    STFT = tf.signal.stft(x,
                          frame_size,
                          frame_shift,
                          fft_size,
                          window_fn=W,
                          pad_end=True)
    return tf.abs(STFT), tf.math.angle(STFT)


def synthesis(STMS, STPS):
    STFT = tf.cast(STMS, tf.complex64) * tf.exp(
        1j * tf.cast(STPS, tf.complex64))
    return tf.signal.inverse_stft(
        STFT, frame_size, frame_shift, fft_size,
        tf.signal.inverse_stft_window_fn(frame_shift, W))


def mmse_lsa(xi, gamma):
    xi = tf.maximum(xi, 1e-12)
    gamma = tf.maximum(gamma, 1e-12)
    v_1 = tf.math.truediv(xi, tf.math.add(1.0, xi))
    nu = tf.math.multiply(v_1, gamma)
    v_2 = exp1(nu)
    # v_2 = tf.math.negative(tf.math.special.expint(tf.math.negative(nu))) # E_1(x) = -E_i(-x)
    return tf.math.multiply(v_1, tf.math.exp(tf.math.multiply(
        0.5, v_2)))  # MMSE-LSA gain function.


def read_mat(path):
    if not path.endswith('.mat'): path = path + '.mat'
    return loadmat(path)


def save_mat(path, data, name):
    if not path.endswith('.mat'): path = path + '.mat'
    savemat(path, {name: data})


if len(sys.argv) != 4:
    print('Usage: infer.py source_folder dest_folder config_folder')
    sys.exit(1)

src_dir = sys.argv[1]
dest_dir = sys.argv[2]
conf_dir = sys.argv[3]

if not os.path.exists(conf_dir):
    print(conf_dir + ' not exist!')
    sys.exit(1)

if not os.path.exists(src_dir):
    print(src_dir + ' not exist!')
    sys.exit(1)

if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

try:
    os.makedirs(dest_dir)
except OSError:
    raise

mu_mat = read_mat(os.path.join(conf_dir, 'mu.mat'))
mu = mu_mat['mu']
sigma_mat = read_mat(os.path.join(conf_dir, 'sigma.mat'))
sigma = sigma_mat['sigma']

# print(mu, sigma)

# 0/1 is same for the result
load_model_with_pb = 1

if load_model_with_pb:
    model = tf.keras.models.load_model(os.path.join(conf_dir, 'model/'))
else:
    inp = Input(name='inp', shape=[None, 257], dtype='float32')
    network = ResNetV2(inp=inp,
                   n_outp=257,
                   n_blocks=20,
                   d_model=256,
                   d_f=64,
                   k=3,
                   max_d_rate=16,
                   padding='causal',
                   unit_type='ReLU->LN->W+b',
                   outp_act='Sigmoid')
    model = Model(inputs=inp, outputs=network.outp)
    model.load_weights('./config/model/variables/variables')

model.summary()

for wavfile in glob.iglob(os.path.join(src_dir, '**/*.wav'), recursive=True):
    basename = os.path.basename(wavfile)
    filenames = os.path.splitext(basename)
    try:
        wav, fs = soundfile.read(wavfile)
    except:
        print('cannot read wav {}!'.format(wavfile))
        continue
    x_STMS, x_STPS = analysis(wav)
    feature = tf.reshape(x_STMS, [1, x_STMS.shape[0], x_STMS.shape[1]])
    hat_bar_xi = model.predict(feature, batch_size=1, verbose=1)
    hat_bar_xi = tf.squeeze(hat_bar_xi)
    v_1 = tf.math.multiply(sigma, tf.math.sqrt(2.0))
    v_2 = tf.math.multiply(2.0, hat_bar_xi)
    v_3 = tf.math.erfinv(tf.math.subtract(v_2, 1))
    v_4 = tf.math.multiply(v_1, v_3)
    x = tf.math.add(v_4, mu)
    hat_xi = tf.math.pow(10.0, tf.truediv(x, 10.0))
    save_mat(os.path.join(dest_dir, filenames[0] + '_hat_xi.mat'),
             hat_xi.numpy(), 'hat_xi')
    hat_gamma = tf.math.add(hat_xi, 1.0)
    gain = mmse_lsa(hat_xi, hat_gamma)
    gain = tf.cast(gain, tf.float64)
    y_STMS = tf.math.multiply(x_STMS, gain)
    out_wav = synthesis(y_STMS, x_STPS)
    soundfile.write(os.path.join(dest_dir, basename), out_wav, fs)
