import os
import numpy as np
import time
import argparse
import pandas as pd
import yaml
import sys 
from PIL import Image

# Setup path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path)
print("Search path:", base_path)

from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchPatches
from wsi_core.batch_process_utils import initialize_df

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def stitching(file_path, downscale=64):
    start = time.time()
    heatmap = StitchPatches(file_path, downscale=downscale, bg_color=(0, 0, 0), alpha=-1, draw_grid=False)
    return heatmap, time.time() - start

def segment(wsi_obj, seg_params, filter_params):
    start_time = time.time()
    wsi_obj.segmentTissue(**seg_params, filter_params=filter_params)
    return wsi_obj, time.time() - start_time

def extract_and_save_png_patches(wsi_obj, patch_params, slide_id, patch_png_dir):
    os.makedirs(patch_png_dir, exist_ok=True)
    patch_dir = os.path.join(patch_png_dir, slide_id)
    os.makedirs(patch_dir, exist_ok=True)

    patch_generator = wsi_obj._getPatchGenerator(
        cont=None,
        cont_idx=0,
        patch_level=patch_params['patch_level'],
        save_path=patch_dir,
        patch_size=patch_params['patch_size'],
        step_size=patch_params['step_size'],
        custom_downsample=1,
        white_black=True,
        white_thresh=patch_params['white_thresh'],
        black_thresh=patch_params['black_thresh'],
        contour_fn=patch_params['contour_fn'],
        use_padding=True
    )

    for i, patch_info in enumerate(patch_generator):
        patch = patch_info['patch_PIL']
        coord_x = patch_info['x']
        coord_y = patch_info['y']
        filename = f"{coord_x}_{coord_y}.png"
        patch.save(os.path.join(patch_dir, filename))

def seg_and_patch(config):
    paths = config['paths']
    proc = config['processing']

    patch_size = proc['patch_size']
    step_size = proc['step_size']
    patch_level = proc['patch_level']
    seg = proc['seg']
    patch = proc['patch']
    stitch = proc['stitch']
    auto_skip = proc['auto_skip']

    seg_params = config['segmentation']
    filter_params = config['filtering']
    vis_params = config['visualization']
    patch_params = config['patching']

    patch_png_dir = paths['patch_png_dir']
    mask_save_dir = paths['mask_save_dir']
    stitch_save_dir = paths['stitch_save_dir']
    save_dir = paths['save_dir']
    source = paths['source']

    process_list = paths.get('process_list')
    if process_list:
        process_list = os.path.join(save_dir, process_list)

    os.makedirs(patch_png_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)
    os.makedirs(stitch_save_dir, exist_ok=True)

    slides = sorted([f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))])

    if process_list:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params, save_patches=True)
    else:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params, save_patches=True)

    process_stack = df[df['process'] == 1]
    seg_times, patch_times, stitch_times = 0., 0., 0.

    for idx in process_stack.index:
        slide = df.loc[idx, 'slide_id']
        slide_id, _ = os.path.splitext(slide)

        if auto_skip and os.path.exists(os.path.join(patch_png_dir, slide_id)):
            print(f"[SKIP] {slide_id} already processed")
            df.loc[idx, 'status'] = 'already_exist'
            continue

        print(f"\n[INFO] Processing {slide_id}")
        full_path = os.path.join(source, slide)
        wsi_obj = WholeSlideImage(full_path)

        for key, param_dict in [('seg_level', seg_params), ('vis_level', vis_params)]:
            if param_dict[key] < 0:
                best_level = wsi_obj.getOpenSlide().get_best_level_for_downsample(64)
                param_dict[key] = best_level
                df.loc[idx, key] = best_level

        for id_key in ['keep_ids', 'exclude_ids']:
            val = seg_params[id_key]
            if isinstance(val, list) and len(val) > 0:
                seg_params[id_key] = np.array(val).astype(int)
            elif isinstance(val, str) and val.lower() != 'none' and val.strip() not in ['', '[]']:
                seg_params[id_key] = np.array(val.strip('[]').split(',')).astype(int)
            else:
                seg_params[id_key] = []

        if seg:
            wsi_obj, seg_time = segment(wsi_obj, seg_params, filter_params)
            seg_times += seg_time
            print(f"[INFO] Segmentation took {seg_time:.2f}s")

        mask_img = wsi_obj.visWSI(**vis_params)[0]
        mask_img.save(os.path.join(mask_save_dir, f"{slide_id}.png"))

        if patch:
            start = time.time()
            extract_and_save_png_patches(wsi_obj, {
                'patch_level': patch_level,
                'patch_size': patch_size,
                'step_size': step_size,
                'white_thresh': patch_params['white_thresh'],
                'black_thresh': patch_params['black_thresh'],
                'contour_fn': patch_params['contour_fn']
            }, slide_id, patch_png_dir)
            patch_times += time.time() - start
            print(f"[INFO] PNG Patching took {patch_times:.2f}s")

        df.loc[idx, 'status'] = 'processed'
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)

    total = len(process_stack)
    print(f"\n[SUMMARY] Avg segmentation time: {seg_times/total:.2f}s")
    print(f"[SUMMARY] Avg patching time: {patch_times/total:.2f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation + PNG patching via config')
    parser.add_argument('--config', type=str, required=True, help='YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)
    seg_and_patch(config)
    print("[INFO] Finished processing all slides.")
