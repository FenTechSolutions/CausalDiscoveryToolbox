"""Testing dataset import"""

from cdt.data import load_dataset


def test_tuebingen():
    data, labels = load_dataset('tuebingen')
    # print(data.head(), labels.head())


def test_sachs():
    data, graph = load_dataset('sachs')


def test_dream():
    data, graph = load_dataset('dream4-2')


def test_unknown_dataset():
    try:
        data, graph = load_dataset('asdasd')
    except ValueError:
        pass


if __name__ == '__main__':
    test_tuebingen()
