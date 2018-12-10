import numpy as np
from parametrization import get_concatenated_mfcc_matrix_for_each_digit, get_gmm_models
from sklearn.model_selection import KFold


def divide_set_to_test_and_train(speakers_ids, n_of_tests_ex=2):
    """
    Function which given a list divides it into to groups.
    Especially when talking about cross validation - it takes a list of user's id and divide it into a test and training set
    :param speakers_ids: List of speakers id that is going to be divided several times
    :param n_of_tests_ex: this parameter defines the amount of tests users in cross validation, default is 2
    :return: A list of tuples where first element of tuple is training set and second is test set for current cross validation
    """
    if n_of_tests_ex == 2:
        kf = KFold(11)
        splitted = [(
                list(map(lambda z: speakers_ids[z], i)),
                list(map(lambda z: speakers_ids[z], j))
              ) for i, j in kf.split(speakers_ids)]
        return splitted


def classificate_mfcc_to_gmm_model(mfcc_matrix, gmm_models, ll=True):
    """
    Function, given a mfcc matrix and several gmm models computes the best match for this matrix using score function
    In this project function computes score for each GMM model given as parameter mfcc matrix and decided which digit it is
    :param mfcc_matrix: MFCC matrix which is going to be predicted
    :param gmm_models: GMM models along which we are choosing
    :param ll: bool value, if True function return also log likelihood value for the best match, default to True
    :return: Either a digit (index of gmm model) which has the greatest value of score
            or a tuple containing both the digit and value of this log likelihood computed by score function
    """
    log_likelihoods = []
    for number, model in sorted(gmm_models.items(), key=lambda x: int(x[0])):
        log_likelihoods.append(np.exp(model.score(mfcc_matrix)))
    ll_val = max(log_likelihoods)
    idx = log_likelihoods.index(ll_val)
    return idx if not ll else idx, ll_val


def calc_recogn_ratio(confusion_matrix, as_str=False):
    """
    Function which computes recognition ratio for the confusion matrix
    :param confusion_matrix: The confusion matrix going to be calculated
    :param as_str: bool val, if True returns the result as string, if not returns it as float, default to False
    :return:
    """
    value = np.sum(confusion_matrix.diagonal()) / np.sum(confusion_matrix)
    return value if not as_str else "%s%%" % int(value*100)


def validate(training_set, test_set, n_components, n_iter, cov_type):
    """
    Function to do the single cross validation
    :param training_set: Training set from which GMM models will be computed as dictionary
    :param test_set: Test set on which the validation is going to occur as dictionary
    :param n_components: Number of components of GMM model
    :param n_iter: Number of iterations the GMM model will be trained
    :param cov_type: Type of covariance type in GMM model
            More information about these params in function parametrization.get_gmm_models
    :return: The computed confusion matrix
    """
    words_count = 10
    confusion_matrix = np.zeros((words_count, words_count))
    labels_dict = get_concatenated_mfcc_matrix_for_each_digit(training_set)
    gmm_models = get_gmm_models(labels_dict,  n_components, n_iter, cov_type)
    for speaker_id, speaker_data in test_set.items():
        for label_data in speaker_data:
            curr_mfcc = label_data[0]
            curr_label = label_data[1]
            classif_idx = classificate_mfcc_to_gmm_model(curr_mfcc, gmm_models)
            confusion_matrix[classif_idx[0], int(curr_label)] += 1
    return confusion_matrix


def cross_validation(speakers_mfcc_matrices, n_components=8, n_iter=50, cov_type='diag', n_of_tests_ex=2):
    """
    Main function handling the cross validation which make several calls to function validate above
    This function is creating sum matrix of all cross validations
    :param speakers_mfcc_matrices: the matrix where keys are speakers ids and values are MFCC matrices for each digit
    :param n_components :param n_ite :param cov_type these are descirbed in function parametrization.get_gmm_models
    :param n_of_tests_ex: Number of tests examples in each cross validation, default is 2
    :return: Matrix being a sum of all cross validations
    """
    splitted_set = divide_set_to_test_and_train(speakers_mfcc_matrices.keys(), n_of_tests_ex=n_of_tests_ex)
    sum_matrix = np.zeros((10, 10))
    for training_set, test_set in splitted_set:
        train_set_params = dict((k, speakers_mfcc_matrices[k]) for k in training_set)
        test_set_params = dict((k, speakers_mfcc_matrices[k]) for k in test_set)
        conf_matrix = validate(train_set_params, test_set_params, n_components, n_iter, cov_type)
        sum_matrix = np.add(sum_matrix, conf_matrix)
    return sum_matrix
