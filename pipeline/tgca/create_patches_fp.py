import os
import sys

# Get the absolute path of the parent of the parent directory
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path)
print("Search path:", base_path)

from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
import os
import numpy as np
import time
import pandas as pd
import yaml
import argparse

def stitching(file_path, wsi_object, downscale=64):
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(255,255,255), alpha=-1, draw_grid=True)
    total_time = time.time() - start
    return heatmap, total_time

def segment(WSI_object, seg_params, filter_params):
    start_time = time.time()
    WSI_object.segmentTissue(**seg_params, filter_params=filter_params)
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
    start_time = time.time()
    magnification = WSI_object.wsi.properties['aperio.AppMag']
    kwargs['mag'] = str(magnification)
    file_path = WSI_object.process_contours(**kwargs)
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed

def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, only_mask_save_dir, stitch_save_dir,
                  slide_name_file, patch_size=256, step_size=256,
                  seg_params={'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                              'keep_ids': 'none', 'exclude_ids': 'none'},
                  filter_params={'a_t':100, 'a_h': 16, 'max_n_holes':8},
                  vis_params={'vis_level': -1, 'line_thickness': 500},
                  patch_params={'use_padding': True, 'contour_fn': 'four_pt'},
                  patch_level=0, use_default_params=False, seg=False, save_mask=True,
                  stitch=False, patch=False, auto_skip=True, process_list=None, uuid_name_file=None):

    all_data = np.array(pd.read_excel(uuid_name_file, engine='openpyxl', header=None))
    slides = []
    id_names = {}
    for data in all_data:
        slides.append(data[1])
        id_names[data[1]] = data[0]

    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, id_names[str(slide)], slide))]
    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

    mask = df['process'] == 1
    process_stack = df[mask]
    total = len(process_stack)

    legacy_support = 'a' in df.keys()
    if legacy_support:
        print('detected legacy segmentation csv file, legacy support enabled')
        df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
                          'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
                          'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
                          'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
                          'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    for i in range(total):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
        print('processing {}'.format(slide))

        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide)

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        full_path = os.path.join(source, id_names[slide], slide)
        WSI_object = WholeSlideImage(full_path)

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()
        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}

            for key in vis_params.keys():
                if legacy_support and key == 'vis_level':
                    df.loc[idx, key] = -1
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                if legacy_support and key == 'a_t':
                    old_area = df.loc[idx, 'a']
                    seg_level = df.loc[idx, 'seg_level']
                    scale = WSI_object.level_downsamples[seg_level]
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                    current_filter_params.update({key: adjusted_area})
                    df.loc[idx, key] = adjusted_area
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                if legacy_support and key == 'seg_level':
                    df.loc[idx, key] = -1
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        if current_vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params['vis_level'] = 0
            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params['vis_level'] = best_level

        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0
            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params['seg_level'] = best_level

        keep_ids = str(current_seg_params['keep_ids'])
        if keep_ids != 'none' and len(keep_ids) > 0:
            str_ids = current_seg_params['keep_ids']
            current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['keep_ids'] = []

        exclude_ids = str(current_seg_params['exclude_ids'])
        if exclude_ids != 'none' and len(exclude_ids) > 0:
            str_ids = current_seg_params['exclude_ids']
            current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['exclude_ids'] = []

        w, h = WSI_object.level_dim[current_seg_params['seg_level']]
        if w * h > 1e8:
            print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

        seg_time_elapsed = -1
        if seg:
            WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)

        if save_mask:
            mask, only_mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id + '.jpg')
            mask.save(mask_path)
            only_mask_path = os.path.join(only_mask_save_dir, slide_id + '.png')
            only_mask.save(only_mask_path)

        patch_time_elapsed = -1
        if patch:
            current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size,
                                         'save_path': patch_save_dir})
            file_path, patch_time_elapsed = patching(WSI_object=WSI_object, **current_patch_params)

        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id + '.h5')
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
                stitch_path = os.path.join(stitch_save_dir, slide_id + '.jpg')
                heatmap.save(stitch_path)

        print("segmentation took {} seconds".format(seg_time_elapsed))
        print("patching took {} seconds".format(patch_time_elapsed))
        print("stitching took {} seconds".format(stitch_time_elapsed))
        df.loc[idx, 'status'] = 'processed'

        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
    print("average stiching time in s per slide: {}".format(stitch_times))
    return seg_times, patch_times

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='seg and patch')
    parser.add_argument('--config', type=str, default='config.yaml', help='path to YAML config file')
    args = parser.parse_args()

    # Load YAML configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Extract paths from config
    paths = config['paths']
    source = paths['source_dir']
    save_dir = paths['save_dir']
    patch_save_dir = paths['patch_h5_dir']  # Use patch_h5_dir instead of constructing patches_<patch_size>
    mask_save_dir = paths['mask_save_dir']
    only_mask_save_dir = paths['only_mask_save_dir']
    stitch_save_dir = paths['stitch_save_dir']
    slide_name_file = paths['slide_name_file']
    uuid_name_file = paths['uuid_name_file']
    preset = paths['preset_file']

    # Extract processing parameters
    proc = config['processing']
    patch_size = proc['patch_size']
    step_size = proc['step_size']
    patch_level = proc['patch_level']
    seg = proc['seg']
    patch = proc['patch']
    stitch = proc['stitch']
    auto_skip = proc['auto_skip']

    # Create output directories
    directories = {
        'source': source,
        'save_dir': save_dir,
        'patch_save_dir': patch_save_dir,
        'mask_save_dir': mask_save_dir,
        'only_mask_save_dir': only_mask_save_dir,
        'stitch_save_dir': stitch_save_dir
    }

    for key, val in directories.items():
        print(f"{key} : {val}")
        if key != 'source':
            os.makedirs(val, exist_ok=True)

    # Load preset parameters if provided
    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if preset:
        preset_df = pd.read_csv(preset)
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]
        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]
        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]
        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]
    # if preset:
    #     preset_df = pd.read_csv(preset)
    #     for key in seg_params.keys():
    #         seg_params[key] = preset_df.loc[0, key]
    #     for key in filter_params.keys():
    #         filter_params[key] = preset_df.loc[0, key]
    #     for key in vis_params.keys():
    #         vis_params[key] = preset_df.loc[0, key]
    #     for key in patch_params.keys():
    #         if key in preset_df.columns:
    #             patch_params[key] = preset_df.loc[0, key]
        # Explicitly handle center_shift
        # if 'center_shift' in preset_df.columns:
        #     patch_params['center_shift'] = preset_df.loc[0, 'center_shift']
        
    parameters = {
        'seg_params': seg_params,
        'filter_params': filter_params,
        'patch_params': patch_params,
        'vis_params': vis_params
    }

    print(parameters)

    seg_times, patch_times = seg_and_patch(
        **directories,
        **parameters,
        slide_name_file=slide_name_file,
        patch_size=patch_size,
        step_size=step_size,
        seg=seg,
        use_default_params=False,
        save_mask=True,
        stitch=stitch,
        patch_level=patch_level,
        patch=patch,
        auto_skip=auto_skip,
        process_list=None,
        uuid_name_file=uuid_name_file
    )