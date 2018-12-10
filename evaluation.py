from file_handler import write_to_csv, read_wav_files_in_directory, load_from_csv
from parametrization import get_concatenated_mfcc_matrix_for_each_digit, get_gmm_models, get_mfcc
from cross_validation import classificate_mfcc_to_gmm_model
import numpy as np
from sklearn.metrics import confusion_matrix


def score_evaluate_waves(mfcc_matrices_for_each_speaker, winlen, numcep, nfilt, nfft, appendEnergy,
                         delta, delta_delta, n_components, n_iters, cov_type):
    """
    Function that do the evaluation on given data - assign each wave to a digit which was denounced
    And then this function wwites the result to the CSV file
    :param mfcc_matrices_for_each_speaker: the matrix where keys are speakers ids and values are MFCC matrices for each digit
    :param winlen: :param numcep: :param nfilt: :param nfft: :param appendEnergy: :param delta:  :param delta_delta:
    Above params are described in parametrization.get_mfcc
    :param n_components: :param n_iters: :param cov_type:
    Above params are described in parametrization.get_gmm_models
    :return: None
    """
    # GMM MODELS
    gmm_models_ = get_gmm_models(get_concatenated_mfcc_matrix_for_each_digit(mfcc_matrices_for_each_speaker),
                                 n_components=n_components, n_iter=n_iters, cov_type=cov_type)

    # EVALUATION SET
    evaluation_wave_list = sorted(read_wav_files_in_directory('eval'), key=lambda x: x[0])
    mfcc_matrices_for_evaluation_set = {k[0]: get_mfcc(k[1],
                                                       k[2],
                                                       winlen=winlen,
                                                       numcep=numcep,
                                                       nfilt=nfilt,
                                                       nfft=nfft,
                                                       appendEnergy=appendEnergy,
                                                       delta_=delta,
                                                       deltadelta_=delta_delta
                                                       ) for k in evaluation_wave_list}
    classification = [(k, ) + classificate_mfcc_to_gmm_model(v, gmm_models_)
                      for k, v in mfcc_matrices_for_evaluation_set.items()]
    write_to_csv(classification, 'results.csv', delimiter=',', option='w')


def evaluate(results_file='results.csv', true_values_file='true_values.csv'):
    """
    Function which do the evaluation based on two dictionaries:
    :param results_file: The path to the file that stores predicted values,
        load as dictionary ['wav_filename'] -> digit label
    :param true_values_file: The path to the file that stores true values,
        load as dictionary ['wav_filename'] -> digit label
    :return: The confusion matrix
    """
    prediction_dict = load_from_csv(results_file)
    true_dict = load_from_csv(true_values_file)
    prediction_list = []
    true_list = []
    for k, v in true_dict.items():
        if k not in prediction_dict:
            raise Exception(f'No prediction for file {k}')
        true_list.append(v)
        prediction_list.append(prediction_dict[k])
    cm = confusion_matrix(true_list, prediction_list)
    return np.transpose(cm)
