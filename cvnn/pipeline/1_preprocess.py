import argparse
from utils import tiff_to_numpy

from pathlib import Path
from typing import Tuple, List, Union, Dict

import numpy as np
import pandas as pd
import json 

def get_targets_for_tile(targets: pd.DataFrame, t_x: int, t_y : int, 
                         tile_size: int, tile_overlap: int
                         ):
    bbox_size = 64
    s = tile_size - tile_overlap
    targets_ij = targets[
        (targets['detect_scene_column'] // s  == t_x) & 
        (targets['detect_scene_row'] // s  == t_y)
        ]
    tile_targets = []

    for idx, target in targets_ij.iterrows():
        # #create bounding boxes
        # pos_x = target['detect_scene_column'] - t_x * s 
        # pos_y = target['detect_scene_row'] - t_y * s 
        # bot = pos_y - bbox_size // 2
        # top = pos_y + bbox_size // 2
        # left = pos_x - bbox_size // 2
        # right = pos_x + bbox_size // 2
        
        if any(np.isnan(target[coord]) for coord in ['bottom', 'top', 'left', 'right']):
            continue
        
        bot = target['bottom'] - t_y * s
        top = target['top'] - t_y * s
        left = target['left'] - t_x * s
        right = target['right'] - t_x * s

        if bot < 0:
            bot = 0
            top +=1 #to avoid bot and top being 0

        if top >= tile_size:
            top = tile_size - 1
            bot -= 1
        
        if left < 0:
            left = 0
            right += 1
        
        if right >= tile_size:
            right = tile_size - 1
            left -=1

        bbox = [left, bot, right, top]
        
        target_class = 3

        if target['is_vessel'] == True:
            # fishing vessel: 0
            if target['is_fishing'] == True:
                target_class = 0

            #non-fishing vessel 1
            elif target['is_fishing'] == False:
                target_class = 1
        
        # non vessel: 2
        elif target['is_vessel'] == False:
            target_class = 2
        
        
        # background class: 4
        tile_targets.append({'index': idx, 'bbox': bbox, 'class': target_class, 'confidence': target['confidence'] })

    return tile_targets

def pad_and_combine(
        images: List[np.ma.MaskedArray], tile_size: int, tile_overlap: int
        ) -> np.ndarray : 
    
    s = tile_size - tile_overlap
    img_height, img_width = images[0].shape
    new_height = int( np.ceil( (img_height - tile_overlap) / s ) * s ) + tile_overlap
    new_width = int( np.ceil( (img_width - tile_overlap) / s ) * s ) + tile_overlap
    pad_height, pad_width = new_height - img_height, new_width - img_width
    
    combined_imgs = np.zeros((len(images), new_height, new_width), dtype=np.complex64)

    for idx, img in enumerate(images):
        padded_img = np.pad(
            img, pad_width=((0, pad_height), (0, pad_width)), mode="constant", 
            constant_values=0
            )
        combined_imgs[idx] = padded_img

    return combined_imgs

def chop_and_save(combined_imgs: np.ndarray, partitions: list, targets: int, img_dir: Path, tile_size: int, tile_overlap: int, num_augmentations: int):
    s = tile_size - tile_overlap

    img_height, img_width = combined_imgs.shape[1], combined_imgs.shape[2]
    num_tiles_height, num_tiles_width = img_height // s, img_width // s

    if not img_dir.exists():
        img_dir.mkdir(parents=True, exist_ok=True)
    
    img_tiles_with_targets = [[] for _ in range(len(partitions))]

    for t_x in range(num_tiles_width):
        for t_y in range(num_tiles_height):
            start_width, end_width = t_x * s, t_x * s + tile_size
            start_height, end_height = t_y * s, t_y * s + tile_size
            
            tile_targets = get_targets_for_tile(targets, t_x, t_y, tile_size, tile_overlap)
            
            if tile_targets == []:
                continue
            
            if not img_dir.exists():
                img_dir.mkdir(parents=True, exist_ok=True)
            
            augmentations = [2 * np.pi * n / num_augmentations for n in range(num_augmentations)]
            for idx, phi in enumerate(augmentations):
                tile_fn = f"{t_x}_{t_y}_{idx}.npy"
                with open(str(Path(img_dir, tile_fn)), 'wb') as f:
                    np.save(f, combined_imgs[:, start_height : end_height, start_width : end_width]*np.exp(1j * phi))
                
                for i in range(len(img_tiles_with_targets)):
                    img_tiles_with_targets[i].append({'image_folder': str(img_dir), 'filename': tile_fn, 'partition': partitions[i], 'targets': tile_targets})
    
    return img_tiles_with_targets

def update_dataset(temp_data_files : list, dataset_files: list):
    for temp_data_file, dataset_file in zip(temp_data_files, dataset_files):

        with open(temp_data_file, 'r') as df:
            lines = df.readlines()
    
        if dataset_file.exists():

            with open(dataset_file, 'r') as df:
                data = json.load(dataset_file)
        else:
            data = []

        for line in lines:
            data.append(eval(line))
    
        with open(dataset_file, 'w') as ndf:
            json.dump(data, ndf, indent=4)

        Path.unlink(temp_data_file)

def make_tiles_and_targets(data_root_dir: Path, data_quality: str, product_type: str, fold_file: str, fold_numbers: list,
                 tile_path: Path, tile_size: int, tile_overlap: int, num_augmentations
                 ):
    temp_data_files = []
    dataset_files = []
    for fold_number in fold_numbers:
        temp_data_files.append(Path(tile_path, f'temp_fold_{fold_number}.json'))
        dataset_files.append(Path(tile_path, f'fold_{fold_number}.json'))

    # get the identifier correspondences of the scenes for the selected fold 
    identifier_correspondences = pd.read_csv(
        str( Path(data_root_dir, 'identifier_correspondences.csv') )
        )
    
    fold_df = pd.read_csv(fold_file)
    fold_scenes = fold_df['scene_id'].values
    
    products_in_fold = identifier_correspondences[
        identifier_correspondences['scene_id'].isin(fold_scenes)
        ]
    
    
    # obtain all targets in the selected fold as dataframe
    all_targets_in_fold = pd.DataFrame()

    for task in ['detection']:
        part_targets_fn = Path(
            data_root_dir, 
            f"{product_type}_{task}.csv"
            )
        
        part_targets = pd.read_csv(part_targets_fn)
        part_targets = part_targets.dropna(subset=['is_vessel'])
        part_targets_in_fold = part_targets[part_targets['scene_id'].isin(fold_scenes)]
        all_targets_in_fold = pd.concat([all_targets_in_fold, part_targets_in_fold])

    # for the selected scenes, load the appropriate products
    if product_type == 'GRD':
        raise NotImplementedError('chopping of GRD products not implemented yet')

    elif product_type =='SLC':
        
        #create a directory to store the tiles if needed
        if not tile_path.exists():
            tile_path.mkdir(parents = True, exist_ok = True)
        
        row_idx = 0
        for _, row in products_in_fold.iterrows():
            num_products = len(products_in_fold)
            product_partitions = []

            for fold_number in fold_numbers:
                product_partitions.append(fold_df.loc[fold_df['scene_id'] == row['scene_id'], f'Fold_{fold_number}'].values[0])
            
            for swath_index in [1, 2, 3]:
                print(f"tiling {row[f'{product_type}_product_identifier']}, swath {swath_index}...")
                
                img_dir = Path(tile_path, 'images', row[f"{product_type}_product_identifier"], f"swath{swath_index}")
                
                if img_dir.exists():
                    print(f'done. [{row_idx * 3 + swath_index}/{num_products * 3}]')
                    continue
                
                # load the .tiff files for the VH and VV polarizations and combine in one numpy array

                if data_quality == 'SARFish_data':
        
                    measurement_directory = Path(
                        data_root_dir, data_quality, 'raw_data', product_type, 'detection', 
                        f"{row[f'{product_type}_product_identifier']}.SAFE", "measurement", 
                        )
                    
                    vh_fn = Path(measurement_directory, row[f"SLC_swath_{swath_index}_vh"])
                    (vh_img, _, _, _) = tiff_to_numpy.load_SARFishProduct(str(vh_fn))
                    
                    
                    vv_fn = Path(measurement_directory, row[f"SLC_swath_{swath_index}_vv"])
                    (vv_img, _, _, _) = tiff_to_numpy.load_SARFishProduct(str(vv_fn))

                    # get the targets corresponding to the current image

                elif data_quality == 'SNAP_data':
                    
                    fn = Path(
                        data_root_dir, data_quality, 'raw_data', product_type, 'detection', f'swath_{swath_index}',
                        f"{row[f'{product_type}_product_identifier']}_Orb_Cal_Deb.tif")
                    
                    vh_img, vv_img = tiff_to_numpy.load_SentinelProduct(fn)


                # pad the images to accomodate tiling and combine in one array
                combined_imgs = pad_and_combine([vh_img, vv_img], tile_size, tile_overlap)

                # tile the combined images and targets and save to directory
                targets_for_product = all_targets_in_fold[
                    (all_targets_in_fold['scene_id'] == row['scene_id']) &
                    (all_targets_in_fold['swath_index'] == swath_index) 
                   ]
                
                img_tiles_with_targets = chop_and_save(
                    combined_imgs, product_partitions, targets_for_product, img_dir, tile_size, tile_overlap, num_augmentations
                    )

                for i, temp_data_file in enumerate(temp_data_files):
                    mode = 'a' if temp_data_file.exists() else 'w'
                
                    with open(temp_data_file, mode) as df:
                        for tile_info in img_tiles_with_targets[i]:
                            json.dump(tile_info, df, separators=(',', ':'))
                            df.write('\n')

                print(f'done. [{row_idx * 3 + swath_index}/{num_products * 3}]')

            row_idx += 1
            
        update_dataset(temp_data_files, dataset_files)
   


def parse_args():
    parser = argparse.ArgumentParser(description="creates a tiled dataset from sentinel-1 GRD or SLC products")

    parser.add_argument(
        '--data_root_dir', type=str,
        help='path/to/data/root/dir/'
    )
    parser.add_argument(
        '--data_quality', choices=['SARFish_data', 'SNAP_data'], type=str,
        help='usage of either the raw SARFish data or the preprocessed Sentinel 1 images'
    )

    parser.add_argument(
        '--product_type', choices=['SLC', 'GRD'], type=str,
        help='SARFish product type'
    )

    parser.add_argument(
        '--fold_file', type=str,
        help='/path/to/fold_file.csv'
    )

    parser.add_argument(
        '--fold_numbers',type=str,
        help='which fold to use'
    )

    parser.add_argument(
        '--tile_path', type=str,
        help='/path/to/stored/tiles/'
    )

    parser.add_argument(
        '--tile_size', type=int,
        help='number of pixels for the height and width of the tiles'
    )

    parser.add_argument(
        '--tile_overlap', type=int,
        help='number of pixels of horizontal and vertical overlap between neighbouring tiles'
    )
    parser.add_argument(
        '--num_augmentations', type=int, 
        help='number of phase shifts used to augment the data')
    return parser.parse_args()

def main():
    args = parse_args()

    make_tiles_and_targets(
        Path(args.data_root_dir), args.data_quality, args.product_type, args.fold_file,
        [fn for fn in args.fold_numbers.split(',')], Path(args.tile_path), 
        args.tile_size, args.tile_overlap, args.num_augmentations
        )

  
if __name__ == "__main__":
    main()

