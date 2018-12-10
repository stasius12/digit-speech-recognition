import csv
import numpy as np
import os
from scipy.io import wavfile


def read_wav_files_in_directory(path_='waves'):
    """
    Function to read all of the wave files in given directory or relative path
    :param path_: name of the directory or relative path containing wave files
    :return: List of tuples in format: [(name_pf_the_file, wave data, sampling frequency of data),...]
    """
    names_of_files = os.listdir(path_)
    list_of_waves = []
    for el in names_of_files:
        freq, data = wavfile.read(path_ + '/%s' % el)
        list_of_waves.append((el, data, freq))
    return list_of_waves


def write_to_csv(data, file, delimiter=' ', option='w'):
    """
    Function that write your data into csv file
    :param data: data you want to write in csv file, either tuple, list or dict
                 * when given a list or a tuple it writes its content to a single line delimit by given delimiter
                 * when given a dictionary it write as many lines as number of keys in dict in format "key<delimiter>value"
                 * when given 2D list or numpy array it writes as many lines as the sublist (of the list),
                        where in each line elements of a sublist are delimited by delimiter param
    :param file: name of the file to write, should be in .csv format
    :param delimiter: delimiter which will be used to separate elements in each line, defaults to ' ' (space)
    :param option: method opening the file 'r', 'a', 'w', 'x', defaults to 'w'
    :return: None
    """
    if not (isinstance(data, list) or isinstance(data, dict) or isinstance(data, tuple) or isinstance(data, np.ndarray)):
        raise Exception("Error: write_to_csv can only write list, tuple, dict or numpy array")
    if isinstance(data, dict):
        data = data.items()
    with open(file, option) as csv_file:
        writer = csv.writer(csv_file, delimiter=delimiter)
        if len(np.array(data).shape) == 1:
            writer.writerow(data)
        else:
            for el in data:
                writer.writerow(el)


def load_from_csv(filename):
    """
    Function which load the content of the CSV file into a dictionary
    :param filename: filepath of CSV file
    :return: dictionary where keys are first elements in row, and values are the second elements
    """
    results = dict()
    with open(filename, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if not row == []:
                results[row[0]] = int(row[1])
    return results
