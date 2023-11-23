import numpy as np
import os
import glob
import sys
import json
from functools import partial

from typing import List

from utils import determine_dimensions, file_to_tree_type_name


##############################
def preprocess_geojson_files(identifier: int, data_dir: str, what_happens_to_nan: str='keep_nan', bands_to_delete: List[str]=[], 
                             transformer_for_numpy_array: partial = None, verbose: bool=True):
    '''
    This function preprocesses the geojson files. The big goal is to create a numpy array for each sample and store them 
    accordingly in a dedicated folder. 

    identifier: number that identifies the way how the geojson file was created (e.g. 5x5 mask or 3x3 mask and so on)
                Only geojson-files with this identifier tag are preprocessed
    data_dir: folder where the geojson-files are stored
    what_happens_to_nan: preprocessing method of what happens to nan-values in the numpy array.
                         <keep_nan> DEFAULT:    all nan values are kept
                         <apply_nan_mask>:      for each band, a mask is generated (0 for nan, 1 for numeric) and 
                                                concatenated to the original array. Number of bands are doubled.
                         <delete_nan_samples>:  samples that contain nan are not written to disk
    bands_to_delete -> DEFAULT None: band names in this list are not considered when writting the arrays to disk. 
    transformer_for_numpy_array -> DEFAULT None: transformer function that transforms numpy array before
                                                 preprocessing method is applied (e.g. np.nanmean, np.nanmedian,...)

    The arrays are written as .npy-files to disk in certain folders. The folders have the following naming convention:
    "<identifier>_<what_happens_to_nan>_<name_of_transformer_function>_<bands_to_delete>", 
    whereas ".<bands_to_delete>" is not added for no bands.
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
        delete_bands_str = '_' + '-'.join(bands_to_delete) # later used for naming folder
        valid_bands = [x for x in valid_bands if x not in bands_to_delete]
    else:
        sys.exit(f'\nSome given bands {bands_to_delete} are not part of the valid band names: {valid_bands}. Terminating.')

    # find all geojson files based on identifier
    search = os.path.join(data_dir, f"[A-Z]*_{identifier}.geojson")
    tree_type_files = glob.glob(search)
    if not tree_type_files:
        sys.exit(f"\nNo geojson-files found in folder {data_dir} or for identifier {identifier}. Terminating.")

    # find all tree type names
    tree_types = [file_to_tree_type_name(fn_, identifier) for fn_ in tree_type_files]
    if not tree_types:
        sys.exit(f"\nThe found geojson-files don't seem to match the standard naming convention \'<Tree>_<species>_<identifier>.geojson\'. Terminating.")
    
    # if not existent, create folder to save numpy arrays
    output_dir = os.path.join(data_dir, f'{identifier}_{what_happens_to_nan}_' + 
                              f'{transformer_for_numpy_array.func.__name__}{delete_bands_str}')
    os.makedirs(output_dir, exist_ok = True)

    # statistical dictionary for output information
    sample_information = {}
    delete_information = {}

    # loop over all tree type names
    for tree_type in tree_types:

        # initialization of statistical dictionary for output information
        amount_of_samples = 0
        amount_of_samples_deleted = 0

        # open geojson-file
        file_name = os.path.join(data_dir, f'{tree_type}_{identifier}.geojson')
        with open(file_name) as f: data = json.load(f)

        # determine dimensions of data
        dimensions = determine_dimensions(data)

        # loop over each sample
        for s_, sample in enumerate(data["features"]):
            # create numpy array for sample
            array = sample2numpy(sample, bands_to_delete, *dimensions)

            # transformer with partial is applied to the array (e.g. np.nanmean)
            if transformer_for_numpy_array is not None:
                try:
                    array = transformer_for_numpy_array(array)
                except Exception as err:
                    sys.exit(f"\nThe transformer threw an unexpected error {type(err)}: {err}. Terminating.")
            
            # samples containing nan are not written to disk
            if (what_happens_to_nan == 'delete_nan_samples') and (np.isnan(array).any()):
                amount_of_samples_deleted += 1 # counter for samples
                continue
            
            # nan mask is concatenated to numpy array (0 for nan, 1 for numeric value)
            # channel dimension is doubled
            if what_happens_to_nan == 'apply_nan_mask':
                mask = (~np.isnan(array)).astype(array.dtype)
                array = np.concatenate((array, mask), axis=2) #TODO: change numpy array dimensions from (h,w,b) to (b,h,w) for less confusing array visualization 

            # array is saved as .npy-file in dedicated folder
            amount_of_samples += 1 # counter for samples
            np.save(os.path.join(output_dir, f'{tree_type}-{s_}.npy'), array, allow_pickle=False)

        # fill statistical dictionary for output information
        sample_information[tree_type] = amount_of_samples
        delete_information[tree_type] = amount_of_samples_deleted
        
        if verbose:
            print(f'\n<{tree_type}> {amount_of_samples} samples written to disk.')

    if verbose:
        print(f'\nIdentifier: {identifier}' + 
              f'\nChosen processing method: {what_happens_to_nan}' +
              f'\nNot considered bands: {bands_to_delete}' +
              f'\nTransformer: {transformer_for_numpy_array}' +
              f'\nTree types considered: {tree_types}' + 
              f'\nAmount of samples written: {sample_information}' +
              f'\nAmount of samples deleted: {delete_information}\n')
            


def sample2numpy(sample: dict, bands_to_delete: List[str], w: int=25, h: int=25, b: int=30) -> np.array:
    '''
    This function converts the geojson strcture (dicctionary) to a numpy array
    axis = 0: height
    axis = 1: width
    axis = 2: channels

    sample: dictionary structure of the geojson file
    bands_to_delete: band names that should not be written to disk
    '''
    b -= len(bands_to_delete)

    # delete bands
    properties = sample["properties"]
    for key in bands_to_delete:
        del properties[key]

    # fill up array
    #TODO?: change numpy array dimensions from (h,w,b) to (b,h,w) for less confusing array visualization 
    array = np.full((h, w, b), np.nan)
    for b_, band in enumerate(properties.values()):
        if band is None: continue       
        for r_, row in enumerate(band):
            array[r_, :, b_] = row
    return array


##############################


if __name__ == "__main__":
    identifier = 1102
    data_dir = 'data'

    bands_to_delete = ["B2"]

    # Preproessing options ################
    #what_happens_to_nan='apply_nan_mask'
    what_happens_to_nan='delete_nan_samples'
    #what_happens_to_nan='keep_nan'

    # Transformer options ################
    #transformer = None
    transformer = partial(np.nanmean, axis=(0,1))  # mean per band excluding nan values
    #transformer = partial(np.nanmedian, axis=(0,1)) # median per band excluding nan values

    preprocess_geojson_files(identifier, data_dir, what_happens_to_nan, bands_to_delete, 
                             transformer_for_numpy_array=transformer)

    # Checking one sample
    arr = np.load(r'data/1102_delete_nan_samples_B2/Abies_alba-21.npy')
    print(arr)
    print(arr.shape)