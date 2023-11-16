import numpy as np
import os
import glob
import re
import sys
import json


##############################


def preprocess_geojson_files(identifier: int, data_dir: str, preprocess_method: str):
    '''
    Description in work.
    '''
    preprocess_methods = ['no_filter'] # valid preprocessing methods implemented in this python script

    if not preprocess_method in preprocess_methods:
        sys.exit(f"{preprocess_method} not in list of valid preprocessing methods: {preprocess_methods}. Terminating.")

    # find all files based on identifier
    search = os.path.join(data_dir, f"[A-Z]*_{identifier}.geojson")
    tree_type_files = glob.glob(search)

    if not tree_type_files:
        sys.exit(f"No geojson-files found in folder {data_dir} or for identifier {identifier}. Terminating.")

    # find all tree type names
    tree_types = [file_to_classname(fn_) for fn_ in tree_type_files]

    if not tree_types:
        sys.exit(f"The found geojson-files don't seem to match the standard naming convention \'<Tree>_<species>_<identifier>.geojson\'. Terminating.")
    
    # loop over all tree type names
    for tree_type in tree_types:

        # if not existent, create folder to save numpy arrays
        output_dir = os.path.join(data_dir, f'{tree_type}.{identifier}.{preprocess_method}')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        file_name = f'{tree_type}_{identifier}.geojson'
        with open(file_name) as f: data = json.load(f)

        for sample in data["features"]:
            pass
            'to be continued'


def file_to_classname(file_name):
    '''
    Description in work.
    '''
    tree_type = re.search(f"([A-Z][a-z]+_[a-z]+)_{identifier}.geojson", file_name).group(1)
    return tree_type


def sample2numpy_no_filter(sample):
    '''
    Description in work.
    '''
    b = 30
    h = 5
    w = 5

    array = np.full((h, w, b), np.nan)
    for b_, band in enumerate(sample["properties"].values()):
        if band is None: continue       
        for r_, row in enumerate(band):
            array[r_, :, b_] = row
    return array


##############################


if __name__ == "__main__":
    identifier = 1102
    data_dir = 'data'
    preprocess_method = 'no_filter'
    preprocess_geojson_files(identifier=identifier, data_dir=data_dir, preprocess_method=preprocess_method)