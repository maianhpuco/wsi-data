import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

# Internal project imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df

def load_config(config_path):
	with open(config_path, 'r') as f:
		return yaml.safe_load(f)

def stitching(file_path, wsi_object, downscale=64):
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0, 0, 0), alpha=-1, draw_grid=False)
	return heatmap, time.time() - start

def segment(wsi_object, seg_params, filter_params, mask_file=None):
	start = time.time()
	if mask_file:
		wsi_object.initSegmentation(mask_file)
	else:
		wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
	return wsi_object, time.time() - start

def patching(wsi_object, **kwargs):
	start = time.time()
	file_path = wsi_object.process_contours(**kwargs)
	return file_path, time.time() - start

def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir,
				  patch_size=256, step_size=256, patch_level=0,
				  seg_params=None, filter_params=None, vis_params=None, patch_params=None,
				  seg=False, patch=False, stitch=False, save_mask=True,
				  use_default_params=False, auto_skip=True, process_list=None):

	slides = sorted([s for s in os.listdir(source) if os.path.isfile(os.path.join(source, s))])

	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	process_stack = df[df['process'] == 1]
	total = len(process_stack)

	seg_times = patch_times = stitch_times = 0.

	for i in tqdm(range(total)):
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		slide_id, _ = os.path.splitext(slide)

		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print(f"[SKIP] {slide_id} already exists.")
			df.loc[idx, 'status'] = 'already_exist'
			continue

		print(f"\n[INFO] Processing {slide_id} ({i+1}/{total})")

		full_path = os.path.join(source, slide)
		wsi = WholeSlideImage(full_path)

		current_seg_params = seg_params.copy()
		current_filter_params = filter_params.copy()
		current_vis_params = vis_params.copy()
		current_patch_params = patch_params.copy()

		# Infer best level if needed
		for k in ['vis_level', 'seg_level']:
			if current_seg_params[k if 'seg' in k else k] < 0:
				best_level = wsi.getOpenSlide().get_best_level_for_downsample(64)
				current_seg_params[k] = best_level
				current_vis_params[k] = best_level

		# Handle keep/exclude IDs
		for key in ['keep_ids', 'exclude_ids']:
			val = current_seg_params[key]
			if val != 'none' and len(str(val)) > 0:
				current_seg_params[key] = np.array(str(val).split(',')).astype(int)
			else:
				current_seg_params[key] = []

		# Check WSI size for feasibility
		w, h = wsi.level_dim[current_seg_params['seg_level']]
		if w * h > 1e8:
			print(f"[ERROR] WSI size {w}x{h} too large for segmentation. Skipping.")
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

		# Segment
		if seg:
			wsi, seg_time = segment(wsi, current_seg_params, current_filter_params)
			seg_times += seg_time
			print(f"[INFO] Segmentation took {seg_time:.2f}s")
		else:
			seg_time = -1

		# Save mask
		if save_mask:
			mask = wsi.visWSI(**current_vis_params)
			mask.save(os.path.join(mask_save_dir, f"{slide_id}.jpg"))

		# Patch
		if patch:
			current_patch_params.update({
				'patch_level': patch_level,
				'patch_size': patch_size,
				'step_size': step_size,
				'save_path': patch_save_dir
			})
			_, patch_time = patching(wsi, **current_patch_params)
			patch_times += patch_time
			print(f"[INFO] Patching took {patch_time:.2f}s")
		else:
			patch_time = -1

		# Stitch
		if stitch:
			patch_path = os.path.join(patch_save_dir, slide_id + '.h5')
			if os.path.isfile(patch_path):
				heatmap, stitch_time = stitching(patch_path, wsi)
				stitch_times += stitch_time
				heatmap.save(os.path.join(stitch_save_dir, f"{slide_id}.jpg"))
				print(f"[INFO] Stitching took {stitch_time:.2f}s")
		else:
			stitch_time = -1

		df.loc[idx, 'status'] = 'processed'
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	print(f"\n[SUMMARY] Avg segmentation time: {seg_times:.2f}s")
	print(f"[SUMMARY] Avg patching time: {patch_times:.2f}s")
	print(f"[SUMMARY] Avg stitching time: {stitch_times:.2f}s")

	return seg_times, patch_times

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Segmentation and patching pipeline")
	parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
	args = parser.parse_args()

	# Load configuration
	config = load_config(args.config)

	# Extract paths and processing settings
	paths = config['paths']
	proc = config['processing']

	source = paths['source']
	save_dir = paths['save_dir']
	patch_save_dir = paths['patch_save_dir']
	mask_save_dir = paths['mask_save_dir']
	stitch_save_dir = paths['stitch_save_dir']
	process_list = paths.get('process_list', None)
if process_list:
		process_list = os.path.join(save_dir, process_list)

patch_size = proc['patch_size']
step_size = proc['step_size']
patch_level = proc['patch_level']
seg = proc['seg']
patch = proc['patch']
stitch = proc['stitch']
auto_skip = proc['auto_skip']

# Parameters
seg_params = config['segmentation']
filter_params = config['filtering']
vis_params = config['visualization']
patch_params = config['patching']

# Create output directories
for d in [save_dir, patch_save_dir, mask_save_dir, stitch_save_dir]:
	os.makedirs(d, exist_ok=True)

# Run the pipeline
seg_and_patch(
	source=source,
	save_dir=save_dir,
	patch_save_dir=patch_save_dir,
	mask_save_dir=mask_save_dir,
	stitch_save_dir=stitch_save_dir,
	patch_size=patch_size,
	step_size=step_size,
	patch_level=patch_level,
	seg=seg,
	patch=patch,
	stitch=stitch,
	auto_skip=auto_skip,
	process_list=process_list,
	seg_params=seg_params,
	filter_params=filter_params,
	vis_params=vis_params,
	patch_params=patch_params,
	save_mask=True,
	use_default_params=False
)
