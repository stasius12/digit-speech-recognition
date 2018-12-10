"""
Project which aim is to recognise which digit speaker says
We can divide our project for few parts:
1.  Creating the GMM models from all of the data we have
    a) Recordings of 10 digits spoken by 22 speakers -> 220 recordings
    b) Parametrize this recordings by creating MFCC matrices with 39 features (13 normal features + delta + delta delta)
        -> 220 MFCC matrices
    c) Creating only 10 matrices for each digit concatenating matrices for each speaker
    d) Making 10 GMM models out of each MFCC matrix
2.  Validate the models by doing the cross validation - 11 tests, 2 speakers in each test group
    Recognition ratio after cross validation - 100%
3.  Doing the optimization tests to find the best parameters in both MFCC algorithm and GMM
4.  Evaluation of the data which has ben never used before to check how models are efficient
"""
from file_handler import read_wav_files_in_directory
from parametrization import get_mfcc_matrices_for_each_speaker
from cross_validation import cross_validation, calc_recogn_ratio
from display import plot_confusion_matrix
from optimization_tests import  test_winlen, test_numcep, test_nfilt, test_nfft, test_ncomponents, test_niters
from evaluation import score_evaluate_waves, evaluate


# MFCC params
WINLEN = 0.025
NUMCEP = 13
NFILT = 26
NFFT = 512
APPEND_ENERGY = True
DELTA = True
DELTA_DELTA = True

# GMM parms
N_COMPONENTS = 8
N_ITERS = 100
COV_TYPE = 'diag'

wav_files_for_each_speaker = read_wav_files_in_directory()
speakers_mfcc_matrices = get_mfcc_matrices_for_each_speaker(wav_files_for_each_speaker, WINLEN, NUMCEP, NFILT, NFFT,
                                                            APPEND_ENERGY, DELTA, DELTA_DELTA)

"""
Choose operation to perform:
0 - cross validation
1 - evaluation
2 - optimization tests
"""
OPTION = 2

if OPTION == 0:
    conf_matrix = cross_validation(speakers_mfcc_matrices, N_COMPONENTS, N_ITERS, COV_TYPE,  n_of_tests_ex=2)
    print(conf_matrix)
    print("Recognition Ratio: ", calc_recogn_ratio(conf_matrix))
    plot_confusion_matrix(conf_matrix, percentage_vals=True, title="Recognition ratio: %s" %
                                                                   calc_recogn_ratio(conf_matrix, as_str=True))
    plot_confusion_matrix(conf_matrix, percentage_vals=False, title="Recognition ratio: %s" %
                                                                    calc_recogn_ratio(conf_matrix, as_str=True))

elif OPTION == 1:
    score_evaluate_waves(speakers_mfcc_matrices, WINLEN, NUMCEP, NFILT, NFFT, APPEND_ENERGY, DELTA, DELTA_DELTA,
                         N_COMPONENTS, N_ITERS, COV_TYPE)
    evaluate()

elif OPTION == 2:
    test_winlen(wav_files_for_each_speaker, 'window_length_tests.csv')
    test_numcep(wav_files_for_each_speaker, 'num_cepstrum_tests.csv')
    test_nfilt(wav_files_for_each_speaker, 'nfilt_tests.csv')
    test_ncomponents(wav_files_for_each_speaker, 'ncomponents_tests2.csv')
    test_nfft(wav_files_for_each_speaker, 'nfft_tests.csv')
    test_niters(wav_files_for_each_speaker, 'niters_tests.csv')
