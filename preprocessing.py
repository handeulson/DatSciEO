import numpy as np
import os
import glob
import re
import sys
import json

##############################
#TODO: change numpy array dimensions from (h,w,b) to (b,h,w) for less confusing array visualization 
#TODO: save all numpy-files with same preprocessing steps in same folder, even though they map different tree species?

def preprocess_geojson_files(identifier: int, data_dir: str, what_happens_to_nan: str='keep_nan', bands_to_delete: list[str]=[]):
    '''
    This function preprocesses the geojson files. The big goal is to create a numpy array for each sample and store them 
    accordingly in a dedicated folder. 

    identifier: number that identifies the way how the geojson file was created (e.g. 5x5 mask or 3x3 mask and so on)
                Only geojson-files with this identifier tag are preprocessed
    data_dir: folder where the geojson-files are stored
    what_happens_to_nan: preprocessing method of what happens to nan-values in the numpy array.
                         <keep_nan>:            all nan values are kept DEFAULT OPTION
                         <apply_nan_mask>:      for each band, a mask is generated (0 for nan, 1 for numeric) and concatenated
                                                to the original array. Number of bands are doubled.
                         <delete_nan_samples>:  samples that contain nan are not written to disk
    bands_to_delete: band names in this list are not considered when writting the arrays to disk. DEFAULT OPTION is no bands.

    The arrays are written as .npy-files to disk in certain folders. The folders have the following naming convention:
    "<identifier>.<what_happens_to_nan>.<bands_to_delete>", whereas ".<bands_to_delete>" is not added for no bands.
    '''

    # check if <what_happens_to_nan> argument contains valid element
    valid_preprocess_methods = ['keep_nan', 'apply_nan_mask', 'delete_nan_samples']
    if not what_happens_to_nan in valid_preprocess_methods:
        sys.exit(f"\n<{what_happens_to_nan}> not in list of valid preprocessing methods: {valid_preprocess_methods}. Terminating.")

    # check if <bands_to_delete> argument contains valid elements
    valid_bands = ['B11', 'B11_1', 'B11_2', 'B12', 'B12_1', 'B12_2', 'B2', 'B2_1', 'B2_2', 'B3', 'B3_1', 'B3_2', 'B4', 'B4_1', 'B4_2',
                   'B5', 'B5_1', 'B5_2', 'B6', 'B6_1', 'B6_2', 'B7', 'B7_1', 'B7_2', 'B8', 'B8A', 'B8A_1', 'B8A_2', 'B8_1', 'B8_2']
    if len(bands_to_delete) == 0:
        delete_bands_str = '' # later used for naming folder
    elif all([x in valid_bands for x in bands_to_delete]):
        delete_bands_str = '.' + '-'.join(bands_to_delete) # later used for naming folder
        valid_bands = [x for x in valid_bands if x not in bands_to_delete]
    else:
        sys.exit(f'\nSome given bands {bands_to_delete} are not part of the valid band names: {valid_bands}.')

    # find all geojson files based on identifier
    search = os.path.join(data_dir, f"[A-Z]*_{identifier}.geojson")
    tree_type_files = glob.glob(search)
    if not tree_type_files:
        sys.exit(f"\nNo geojson-files found in folder {data_dir} or for identifier {identifier}. Terminating.")

    # find all tree type names
    tree_types = [file_to_tree_type_name(fn_) for fn_ in tree_type_files]
    if not tree_types:
        sys.exit(f"\nThe found geojson-files don't seem to match the standard naming convention \'<Tree>_<species>_<identifier>.geojson\'. Terminating.")
    
    # if not existent, create folder to save numpy arrays
    output_dir = os.path.join(data_dir, f'{identifier}.{what_happens_to_nan}{delete_bands_str}')
    os.makedirs(output_dir, exist_ok = True)

    # loop over all tree type names
    for tree_type in tree_types:

        # open geojson-file
        file_name = os.path.join(data_dir, f'{tree_type}_{identifier}.geojson')
        with open(file_name) as f: data = json.load(f)

        # loop over each sample
        for s_, sample in enumerate(data["features"]):
            # create numpy array for sample
            array = sample2numpy(sample, bands_to_delete)
            
            # samples containing nan are not written to disk
            if (what_happens_to_nan == 'delete_nan_samples') and (np.isnan(array).any()):
                continue
            
            # nan mask is concatenated to numpy array (0 for nan, 1 for numeric value)
            # channel dimension is doubled
            if what_happens_to_nan == 'apply_nan_mask':
                masked_array = np.ma.masked_where(~np.isnan(array), array)
                mask = np.ma.getmaskarray(masked_array)
                array = np.concatenate((array, mask), axis=2) #TODO: change numpy array dimensions from (h,w,b) to (b,h,w) for less confusing array visualization 

            # array is saved as .npy-file in dedicated folder
            np.save(os.path.join(output_dir, f'{tree_type}-{s_}.npy'), array, allow_pickle=False)

        print(f'\n<{tree_type}> samples written to disk.')
            


def file_to_tree_type_name(file_name: str) -> str:
    '''
    This function extracts the tree type name for a given geojson file

    file_name: name of geojson file
    '''

    tree_type = re.search(f"([A-Z][a-z]+_[a-z]+)_{identifier}.geojson", file_name).group(1)
    return tree_type


def sample2numpy(sample: dict, bands_to_delete: list[str]) -> np.array:
    '''
    This function converts the geojson strcture (dicctionary) to a numpy array
    axis = 0: height
    axis = 1: width
    axis = 2: channels

    samples: dicctionary structure of the geojson file
    bands_to_delete: band names that should not be written to disk
    '''

    # size of arrays
    b = 30 - len(bands_to_delete)
    h = 5
    w = 5

    # delete bands
    properties = sample["properties"]
    for key in bands_to_delete:
        del properties[key]

    # fill up array
    array = np.full((h, w, b), np.nan) #TODO: change numpy array dimensions from (h,w,b) to (b,h,w) for less confusing array visualization 
    for b_, band in enumerate(properties.values()):
        if band is None: continue       
        for r_, row in enumerate(band):
            array[r_, :, b_] = row #TODO: change numpy array dimensions from (h,w,b) to (b,h,w) for less confusing array visualization 
    return array


##############################


if __name__ == "__main__":
    identifier = 1102
    data_dir = 'data'
    bands_to_delete = []
    what_happens_to_nan='apply_nan_mask'

    preprocess_geojson_files(identifier, data_dir, what_happens_to_nan, bands_to_delete)

    arr = np.load(r'data/1102.apply_nan_mask/Abies_alba-20.npy')
    print(arr)
    print(arr.shape)