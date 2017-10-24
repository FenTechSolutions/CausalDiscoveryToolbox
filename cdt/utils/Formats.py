"""
Formatting functions
Author: Diviyan Kalainathan
Date : 2/06/17

"""
from pandas import DataFrame, read_csv
from numpy import array
from sklearn.preprocessing import scale as scaler


def CCEPC_PairsFileReader(filename, scale=True):
    """ Converts a ChaLearn Cause effect pairs challenge format into numpy.ndarray

    :param filename:
    :type filename: str
    :return: Dataframe composed of (SampleID, a (numpy.ndarray) , b (numpy.ndarray))
    :rtype: pandas.DataFrame
    """

    def convert_row(row, scale):
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
        if scale:
            a = scaler(a)
            b = scaler(b)
        return row['SampleID'], a, b

    data = read_csv(filename)
    conv_data = []

    for idx, row in data.iterrows():
        conv_data.append(convert_row(row, scale))
    df = DataFrame(conv_data, columns=['SampleID', 'A', 'B'])
    return df


def reshape_data(df_data, list_variables, type_variables):

    list_array = []

    dim_variables = {}


    for var in list_variables:
        if (type_variables[var] == "Categorical"):
            data = df_data[var].values
            data = pd.get_dummies(data).as_matrix()
            data = data.reshape(data.shape[0], data.shape[1])

        elif (type_variables[var] == "Numerical"):
            data = scale(df_data[var].values)
            data = data.reshape(data.shape[0], 1)

        dim_variables[var] = data.shape[1]

        list_array.append(data)

    return np.concatenate(list_array, axis=1), dim_variables
