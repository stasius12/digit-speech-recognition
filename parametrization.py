from python_speech_features import mfcc, delta
import numpy as np
from sklearn import mixture as skm
from collections import defaultdict
from itertools import chain


def get_mfcc(signal, samplerate, winlen, numcep, nfilt, nfft, appendEnergy, delta_, deltadelta_):
    """
    Function that from given wave file, parametrize it using mfcc algorithm
    :param signal: the audio signal from which to compute mfcc, should be N*1 array
    :param samplerate: the samplerate of the signal we are working  with
    :param winlen: the length of the analysis window in seconds
    :param numcep: the number of cepstrum to return
    :param nfilt: the number of filters in the filterbank
    :param nfft: the FFT size
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy
    :param delta_: bool whether to compute also delta matrix from mfcc matrix
    :param deltadelta_: bool whether to compute also delta delta matrix from delta matrix
    :return: mfcc matrix
    """
    mfcc_matrix = mfcc(signal,
                       winlen=winlen,
                       numcep=numcep,
                       nfilt=nfilt,
                       nfft=nfft,
                       samplerate=samplerate,
                       appendEnergy=appendEnergy,
                       )
    if delta_:
        mfcc_delta_matrix = delta(mfcc_matrix, 5)
        mfcc_matrix = np.concatenate((mfcc_matrix, mfcc_delta_matrix), 1)
        if deltadelta_:
            mfcc_matrix_delta_delta = delta(mfcc_delta_matrix, 5)
            mfcc_matrix = np.concatenate((mfcc_matrix, mfcc_matrix_delta_delta), 1)
    return mfcc_matrix


def get_mfcc_matrices_for_each_speaker(data, winlen=0.025, numcep=13, nfilt=26, nfft=512, appendEnergy=True, delta_=True, deltadelta_=True):
    """
    Function to get mfcc matrices for each digit, for each speaker
    :param data: list of tuples in format [(name_pf_the_file, wave data, sampling frequency of data),...]
    :the rest of params the same as function above
    :return: dictionary, where keys are speakers_id, and values are tuples in format (mcc_matrix, digit_number)
    """
    ret = defaultdict(lambda: [])
    for el in data:
        speaker_id = el[0].split('_')[0]
        digit = el[0].split('_')[1]
        assert(len(speaker_id) == 5 and len(digit) == 1)
        signal = el[1]
        sample_rate = el[2]
        mfcc_ = get_mfcc(signal, sample_rate, winlen, numcep, nfilt, nfft, appendEnergy, delta_, deltadelta_)
        ret[speaker_id].append((mfcc_, digit))
    return ret


def get_concatenated_mfcc_matrix_for_each_digit(speakers_matrix):
    """
    Function to concatenate all the matrices, from all of the users where keys are digit number
    We want to have as many matrices as digits
    :param speakers_matrix:
    :return: dictionary, where keys are digits
    """
    flat_list = list(chain(*speakers_matrix.values()))
    digit_labels = list({x[1] for x in flat_list})
    return {k: np.concatenate((np.array([x[0] for x in flat_list if x[1] == k])), 0) for k in digit_labels}


def get_gmm_models(mfcc_matrix_for_each_digit, n_components, n_iter, cov_type):
    """
    Function which given a dictionary with mfcc matrix for each digit computes the GMM model for each digit
    :param mfcc_matrix_for_each_digit: MFCC matrix
    :param n_components: the number of mixture components
    :param n_iter: the number of EM iterations to perform
    :param cov_type: string describing the type of covariance parameters to use. Must be one of: 'full', 'tied', 'diag', 'spherical'
    :return: GMM models for each digit as a dictionary
    """
    gmm_models = {}
    for label, mfcc_matrix in mfcc_matrix_for_each_digit.items():
        gmm_obj = skm.GaussianMixture(n_components=n_components,
                                      covariance_type=cov_type,
                                      init_params='random',
                                      max_iter=n_iter,
                                      n_init=5,
                                      tol=0.001,
                                      warm_start=True,
                                      random_state=4,
                                      )
        gmm_obj.fit(mfcc_matrix)
        gmm_models[label] = gmm_obj
    return gmm_models
