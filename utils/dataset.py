import json
import os
import glob
import re
from typing import Iterable
from matplotlib.container import BarContainer

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset

from .utils import determine_dimensions, file_to_tree_type_name, sample_file_to_tree_type


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


    def determine_dimensions(self):
        w, h, b = None, None, None
        for f_ in self.class_files:
            with open(f_) as f: collection = json.load(f)
            w, h, b = determine_dimensions(collection)
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
        """
        Visualizes the dataset's data availability.

        - top subplot: number of samples per class and band that are missing
        - middle subplot: histogram of how many bands are missing per sample
        - bottom subplots: histograms of NaN occurance per layer (for our case obsolete)
        """
        # collect some statistics
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

        # initialize subplots
        fig = plt.figure(figsize=(20, 10))
        ax_bar1 = fig.add_subplot(3, 1, 1)
        ax_bar2 = fig.add_subplot(3, 1, 2)
        axs_hist = [fig.add_subplot(3, 3, i) for i in range(7, 10)]

        # plot first plot
        x = np.arange(self.depth)  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0
        norm = self.width*self.height if normalize_nan_sum else 1

        for attribute, measurement in nan_all.items():
            offset = width * multiplier
            rects = ax_bar1.bar(x + offset, measurement/norm, width, label=attribute, align="edge")
            ax_bar1.bar_label(rects, padding=3, rotation=90)
            multiplier += 1
        
        ax_bar1.set_title("Total NaNs per band per class")
        ax_bar1.set_xticks(x+width*len(self.classes)/2, self.bands, rotation=90)
        ax_bar1.legend(loc='upper left')
        ylim = ax_bar1.get_ylim()
        ax_bar1.set_ylim(ylim[0], 1.2*ylim[1])
        

        # plot second plot
        hist = ax_bar2.hist(empty_bands_per_class, bins=self.depth, align="mid")
        for bar_container_ in hist[-1]:
            bar_container_nonzero = self._filter_empty_bars(bar_container_)
            ax_bar2.bar_label(bar_container_nonzero, padding=3, rotation=90)
        
        ax_bar2.set_xlabel("NaNs per sample")
        ax_bar2.set_ylabel("Samples")


        # plot third subplots
        for i, (ax_, nan_class_) in enumerate(zip(axs_hist, hist_all)):
            ax_.hist(nan_class_)
            # ax_.set_title(self.classes[i])
            ax_.legend(title=self.classes[i])
            ax_.set_xlabel("NaNs per band")
            ax_.set_ylabel("Samples")

        plt.tight_layout(pad=3, h_pad=.4)
        plt.show()

    def _filter_empty_bars(self, bar_container):
        rects = np.array([rect for rect in bar_container])
        datavalues = bar_container.datavalues
        heights = np.array([rect.get_height() for rect in rects])
        mask_nonzero = heights > 0

        rects_nonzero = rects[mask_nonzero]
        datavals_nonzero = datavalues[mask_nonzero]
        bar_container_nonzero = BarContainer(rects_nonzero, datavalues=datavals_nonzero, orientation="vertical")
        return bar_container_nonzero


# TODO: add band information?
class TreeClassifPreprocessedDataset(Dataset):
    def __init__(self, data_dir:str, torchify:bool=False, indices:Iterable=None):
        """
        A dataset class for the Tree Classification task.
        Samples need to be created using preprocessing.preprocess_geojson_files() first.

        :param data_dir: Path to the preprocessed data directory
        :param torchify: Whether to flip the data dimensions for torch. Will be removed soon, after Chris
                            implements the flipping per default in the preprocessing.
        :param indices: Optional array indices to load a subset of the dataset; useful for testing purposes
        
        """
        super().__init__()
        self.data_dir = data_dir
        self.torchify = torchify

        self.files = [file_ for file_ in os.listdir(data_dir) if file_.endswith(".npy")]
        if indices: self.files = [self.files[idx] for idx in indices]


        self.classes = list(np.unique([sample_file_to_tree_type(file_) for file_ in os.listdir(data_dir) if file_.endswith(".npy")]))
        self._set_dimensions()

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        data = np.load(os.path.join(self.data_dir, self.files[index]))
        tree_type = sample_file_to_tree_type(self.files[index])
        class_idx = self.labelname_to_label(tree_type)
        
        if self.torchify:
            # size as saved by preprocessing: W, H, C
            # size as required by torch: C, H, W
            data = np.moveaxis(data, [0,1,2], [2,1,0])
        return data, class_idx
    
    def _set_dimensions(self):
        x = np.load(os.path.join(self.data_dir, self.files[0]))
        print(os.path.join(self.data_dir, self.files[0]))
        print(x)
        self.width, self.height, self.depth = x.shape
        self.n_classes = len(self.classes)
    
    def labelname_to_label(self, labelname):
        return self.classes.index(labelname)
    
    def label_to_labelname(self, label):
        return self.classes[label]
        
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



# test dataset
if __name__ == "__main__":
    # test initialization
    # ds = TreeClassifDataset("data", "1102")
    # print("len ds:", len(ds))
    
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
    # ds.band_nan_histogram()

    dsp = TreeClassifPreprocessedDataset("data/1102_apply_nan_mask_B2")
    x0, y0 = dsp[0]
    print("data shape:", x0.shape)
    print("label:", y0)
    print("labelname:", dsp.label_to_labelname(y0))
