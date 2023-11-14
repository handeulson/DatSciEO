import json
import os
import glob
import re

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset


# TODO: get ID and geometry in export
# TODO: reshape data?
# TODO: keep all features in memory? or read from file everytime? -> quite slow, replace with in memory? or preprocessing
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
        self.classes = [self.file_to_classname(fn_) for fn_ in self.class_files]
        self.samples_per_class = {cn_: self.count_samples(fn_) for cn_, fn_ in zip(self.classes, self.class_files)}
        self.cumulative = np.cumsum(list(self.samples_per_class.values()))
        self._create_band_index(self.class_files[0])
        self.determine_dimensions()

        if verbose: print(str(self))

    def __len__(self): return sum(list(self.samples_per_class.values()))

    def __getitem__(self, index, verbose=False):
        if index >= len(self): raise IndexError(f"Index {index} too large for dataset of length {len(self)}.")
        
        # determine which file to open
        file_idx, file = self.index_to_file(index, return_index=True)

        # determine which feature to load from the file
        idx_offsets = np.roll(self.cumulative, 1)
        idx_offsets[0] = 0
        rel_idx = index - idx_offsets[file_idx]

        # load feature
        with open(file) as f: collection = json.load(f)
        feature = collection["features"][rel_idx]

        # transform into numpy array
        data = self.feature_to_numpy(feature)
        
        if verbose:
            print("idx:", index, "file idx:", file_idx, "file:", file, "rel idx", rel_idx)
            print("selected feature:", feature["properties"]["B11"][0])

        # return data as a numpy array and the class label, which is equal to file_idx
        return data, file_idx

    def __str__(self):
        s = f"Dataset found for identifier '{self.identifier}':\n\t" + "\n\t".join(self.class_files)
        s += f"\n  -> Classes ({len(self.classes)}):"
        for c, n in self.samples_per_class.items(): s+= f"\n\t{c:<20} {n:>5} samples"
        s += f"\n  -> Samples: {len(self)}"
        return s

    def _determine_dimension(self, collection):
        w, h, b = None, None, None
        for feature_ in collection["features"]:
            property_names = list(feature_["properties"].keys())
            for prop_name_ in property_names:
                rows = feature_["properties"][prop_name_]
                if rows is not None:
                    b = len(feature_["properties"])
                    h = len(feature_["properties"][prop_name_])        # number of rows
                    w = len(feature_["properties"][prop_name_][0])     # number of columns (values per row)
                    break
            if h is not None: break
        return w, h, b

    def determine_dimensions(self):
        w, h, b = None, None, None
        for f_ in self.class_files:
            with open(f_) as f: collection = json.load(f)
            w, h, b = self._determine_dimension(collection)
            if w is not None: break
        
        if w is None:
            raise ValueError("Dataset seems to be empty.")
        else:
            self.width = w
            self.height = h
            self.depth = b
    

    def feature_to_numpy(self, feature):
        b = self.depth
        h = self.height
        w = self.width

        data = np.full((h, w, b), np.nan)
        for b_, band in enumerate(feature["properties"].values()):
            if band is None: continue       # TODO: how to handle NaN?
            for r_, row in enumerate(band):
                data[r_, :, b_] = row
        return data


    def index_to_file(self, index, return_index=False):
        larger = self.cumulative > index
        file_idx = np.argmax(larger)
        file = self.class_files[file_idx]

        return (file_idx, file) if return_index else file

    def file_to_classname(self, filename):
        classname = re.search(f"([A-Z][a-z]+_[a-z]+)_{self.identifier}.geojson", filename).group(1)
        return classname
    
    def index_to_classname(self, index):
        file = self.index_to_file(index)
        return self.file_to_classname(file)

    def label_to_classname(self, label):
        return self.file_to_classname(self.class_files[label])

    def _create_band_index(self, file):
        with open(file) as f: collection = json.load(f)
        feature = collection["features"][0]
        self.bands = list(feature["properties"].keys())
        self.band2index = {band_: i for (i, band_) in enumerate(self.bands)}
        self.index2band = {i: band_ for (i, band_) in enumerate(self.bands)}

    def count_samples(self, filename):
        with open(filename) as f:
            data = json.load(f)
        return len(data["features"])


    def visualize_samples(self, indices, subplots, band_names=["B5", "B4", "B3"], **kwargs):
        fig, axs = plt.subplots(*subplots, **kwargs)
        fig.suptitle(f"Tree Samples (bands {list(band_names)})")
        axs = axs.flatten() if len(indices) != 1 else [axs]

        band_indices = [self.band2index[b_] for b_ in band_names]

        for i_, ax_ in zip(indices, axs):
            data, label = self[i_]
            ax_.imshow(data[:, :, band_indices])
            ax_.set_title(f"{i_}: {self.label_to_classname(label)}")
        
        return fig
    
    def band_nan_histogram(self, normalize_nan_sum=True):
        # TODO: find empty samples
        """
        Plots a histogram of the dataset's data availability, i.e. the histogram of bands that are NaN.
        """
        empty_bands_per_class = []
        hist_all = []
        nan_all = {}
        for f_ in self.class_files:
            empty_bands_per_feature = []
            nan_per_class = np.zeros(self.depth)
            nan_concat = None
            with open(f_) as f: collection = json.load(f)
            for feature_ in collection["features"]:
                data = self.feature_to_numpy(feature_)
                nan_per_band = np.isnan(data).sum(axis=(0,1))
                # empty_bands = np.isnan(data).sum()
                empty_bands = np.isnan(data).sum()/self.width/self.height
                # sum nan: either 0, 250 (one season missing), 750 (all seasons missing)
                nan_per_class += nan_per_band

                if nan_concat is None:
                    nan_concat = nan_per_band
                else:
                    nan_concat = np.vstack((nan_concat, nan_per_band))

                empty_bands_per_feature.append(empty_bands)

            empty_bands_per_class.append(empty_bands_per_feature)
            nan_all[self.file_to_classname(f_)]  = nan_per_class
            hist_all.append(nan_concat)

        
        fig = plt.figure(figsize=(20, 10))
        ax_bar = fig.add_subplot(2, 1, 1)
        axs_hist = [fig.add_subplot(2, 3, i) for i in range(4, 7)]

        x = np.arange(self.depth)  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0
        norm = self.width*self.height if normalize_nan_sum else 1

        for attribute, measurement in nan_all.items():
            offset = width * multiplier
            rects = ax_bar.bar(x + offset, measurement/norm, width, label=attribute, align="edge")
            ax_bar.bar_label(rects, padding=3, rotation=90)
            multiplier += 1
        
        ax_bar.set_title("Total NaNs per band per class")
        ax_bar.set_xticks(x+width*len(self.classes)/2, self.bands, rotation=90)
        ax_bar.legend(loc='upper left')
        
        print(empty_bands_per_class)
        for i, (ax_, nan_class_) in enumerate(zip(axs_hist, empty_bands_per_class)):
            ax_.hist(nan_class_)
            ax_.set_title(self.classes[i])
            ax_.set_xlabel("NaNs per sample")
            ax_.set_ylabel("Samples")

        plt.show()




# test dataset
if __name__ == "__main__":
    # test initialization
    ds = TreeClassifDataset("data", "1102")
    print("len ds:", len(ds))
    
    # test getitem
    # sample00, label00 = ds[0]
    # print(sample00, sample00.shape, sep="\n")

    # sample0l, label0l =  ds[980]
    # sample10, label10 =  ds[981]
    # sample1l, label1l =  ds[4742]
    # sample20, label20 =  ds[4743]
    # sample2l, label2l =  ds[4747]
    # sampleerror, labelerror = ds[4748]

    # test visualization
    # fig = ds.visualize_samples(np.random.randint(0, len(ds)-1, 12), (3,4))
    # plt.show()

    # test histogram
    ds.band_nan_histogram()
