from matplotlib import pyplot as plt
from itertools import product
import numpy as np


def plot_confusion_matrix(matrix, title="Confusion matrix", percentage_vals=True, cmap='coolwarm'):
    """
    Plot the confusion matrix of given data
    :param matrix: Square matrix
    :param title: Title of the plot
    :param percentage_vals: Whether or not to display values in cells as percentage values
    :param cmap: cmap parameter of pyplot.imshow function
    :return: None
    """
    fig, axs = plt.subplots()
    plt.imshow(matrix, cmap=cmap)
    plt.colorbar()
    if matrix.shape[0] is not matrix.shape[1]:
        raise Exception("Error: plot_confusion_matrix -> matrix should be square")
    dim = matrix.shape[0]
    sum_of_all_elements = np.sum(matrix)
    for i, j in product(range(dim), range(dim)):
        if not matrix[j, i] == 0:
            result = int((matrix[j, i] / sum_of_all_elements / dim) * 100) if percentage_vals else int(matrix[j, i])
            font_size = 12 if i == j else 9
            if not result == 0:
                result = "%s%%" % result if percentage_vals else result
                plt.text(i, j, result, horizontalalignment="center", fontname='serif', fontsize=font_size, fontweight=700)
    ticks = list(range(dim))
    plt.xticks(ticks)
    plt.yticks(ticks)
    axs.tick_params(labeltop=True, labelbottom=False, top=True, bottom=False, labelsize=12)
    plt.title(title, y=1.05, fontsize=15, fontweight=70)
    # plt.title("Recognition ratio: %s" % calc_recogn_ratio(matrix, as_str=True), 0)
    plt.show()
