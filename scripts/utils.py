import datasets


def load_dataset(path):
    try:
        return datasets.load_from_disk(path)
    except:
        return datasets.load_dataset(path)
