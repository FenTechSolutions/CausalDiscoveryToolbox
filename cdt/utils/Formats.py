"""
Formatting functions
Author: Diviyan Kalainathan
Date : 2/06/17

"""
from pandas import DataFrame, read_csv
from numpy import array


def CCEPC_PairsFileReader(filename):
    """ Converts a ChaLearn Cause effect pairs challenge format into numpy.ndarray

    :param filename:
    :type filename: str
    :return: Dataframe composed of (SampleID, a (numpy.ndarray) , b (numpy.ndarray))
    :rtype: pandas.DataFrame
    """

    def convert_row(row):
        """ Convert a CCEPC row into numpy.ndarrays

        :param row:
        :type row: pandas.Series
        :return: tuple of sample ID and the converted data into numpy.ndarrays
        :rtype: tuple
        """
        a = row["A"].split(" ")
        b = row["B"].split(" ")

        if a[0] == "":
            a.pop(0)
            b.pop(0)

        if a[-1] == "":
            a.pop(-1)
            b.pop(-1)

        a = array([float(i) for i in a])
        b = array([float(i) for i in b])

        return row['SampleID'], a, b

    data = read_csv(filename)
    conv_data = []

    for idx, row in data.iterrows():
        conv_data.append(convert_row(row))
    df = DataFrame(conv_data, columns=['SampleID', 'A', 'B'])
    return df


