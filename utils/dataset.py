import json
import os
import glob
import re

import numpy as np

from torch.utils.data import Dataset


# TODO: get ID and geometry in export
# TODO: reshape data?
# TODO: keep all features in memory? or read from file everytime?
# TODO: how to deal with class imbalance? -> Dataloader task?
class TreeClassifDataset(Dataset):
    def __init__(self, data_dir, identifier, verbose=False, *args, **kwargs):
        """
        A dataset class for the Tree Classification task.

        :param data_dir: The directory where to find the geojson files.
        :type data_dir: str
        :param identifier: The identifier to filter dataset files.
        :type identifier: str
        :param verbose: whether to print on instantiation, defaults to True
        :type verbose: bool, optional
        """

        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.identifier = identifier

        # find files based on identifier
        search = os.path.join(data_dir, f"[A-Z]*_{identifier}.geojson")
        self.class_files = glob.glob(search)

        # collect information about the dataset and build an index
        self.classes = [re.search(f"([A-Z][a-z]+_[a-z]+)_{identifier}.geojson", fn_).group(1) for fn_ in self.class_files]
        self.samples_per_class = {cn_: self._count_samples(fn_) for cn_, fn_ in zip(self.classes, self.class_files)}
        self._cumulative = np.cumsum(list(self.samples_per_class.values()))    

        if verbose: print(str(self))

    def __len__(self): return sum(list(self.samples_per_class.values()))

    def __getitem__(self, index, verbose=False):
        if index >= len(self): raise IndexError(f"Index {index} too large for dataset of length {len(self)}.")
        
        # determine which file to open
        idx_offsets = np.roll(self._cumulative, 1)
        idx_offsets[0] = 0
        larger = self._cumulative > index
        file_idx = np.argmax(larger)

        # determine which feature to load from the file
        rel_idx = index - idx_offsets[file_idx]
        file = self.class_files[file_idx]

        # load feature
        with open(file) as f: collection = json.load(f)
        feature = collection["features"][rel_idx]

        # transform into numpy array
        property_names = list(feature["properties"].keys())
        b = len(feature["properties"])                           # number of bands
        h = len(feature["properties"][property_names[0]])        # number of rows
        w = len(feature["properties"][property_names[0]][0])     # number of columns (values per row)
        data = np.zeros((h, w, b))  # TODO: dtype?
        for b_, band in enumerate(feature["properties"].values()):
            for r_, row in enumerate(band):
                data[r_, :, b_] = row
        
        if verbose:
            print("idx:", index, "file idx:", file_idx, "file:", file, "rel idx", rel_idx)
            print("selected feature:", feature["properties"]["B11"][0])

        return data

    def __str__(self):
        s = f"Dataset found for identifier '{self.identifier}':\n\t" + "\n\t".join(self.class_files)
        s += f"\n  -> Classes ({len(self.classes)}):"
        for c, n in self.samples_per_class.items(): s+= f"\n\t{c:<20} {n:>5} samples"
        s += f"\n  -> Samples: {len(self)}"
        return s

    def _count_samples(self, filename):
        with open(filename) as f:
            data = json.load(f)
        return len(data["features"])
        
        
# test dataset
if __name__ == "__main__":
    ds = TreeClassifDataset("data", "1102")
    print("len ds:", len(ds))
    sample00 = ds[0]
    print(sample00, sample00.shape, sep="\n")
    # sample0l = ds[980]
    # sample10 = ds[981]
    # sample1l = ds[4742]
    # sample20 = ds[4743]
    # sample2l = ds[4747]
    # sampleerror = ds[4748]