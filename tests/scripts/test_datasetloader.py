"""Testing dataset import"""

from cdt.data import load_dataset


def test_tuebingen():
    data, labels = load_dataset('tuebingen')
    print(data.head(), labels.head())


if __name__ == '__main__':
    test_tuebingen()
