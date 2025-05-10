import numpy as np
from PIL import Image
import cv2
class Mosaic_Canvas(object):
	def __init__(self,patch_size=256, n=100, downscale=4, n_per_row=10, bg_color=(0,0,0), alpha=-1):
		self.patch_size = patch_size
		self.downscaled_patch_size = int(np.ceil(patch_size/downscale))
		self.n_rows = int(np.ceil(n / n_per_row))
		self.n_cols = n_per_row
		w = self.n_cols * self.downscaled_patch_size
		h = self.n_rows * self.downscaled_patch_size
		if alpha < 0:
			canvas = Image.new(size=(w,h), mode="RGB", color=bg_color)
		else:
			canvas = Image.new(size=(w,h), mode="RGBA", color=bg_color + (int(255 * alpha),))
		
		self.canvas = canvas
		self.dimensions = np.array([w, h])
		self.reset_coord()

	def reset_coord(self):
		self.coord = np.array([0, 0])

	def increment_coord(self):
		#print('current coord: {} x {} / {} x {}'.format(self.coord[0], self.coord[1], self.dimensions[0], self.dimensions[1]))
		assert np.all(self.coord<=self.dimensions)
		if self.coord[0] + self.downscaled_patch_size <=self.dimensions[0] - self.downscaled_patch_size:
			self.coord[0]+=self.downscaled_patch_size
		else:
			self.coord[0] = 0 
			self.coord[1]+=self.downscaled_patch_size
		

	def save(self, save_path, **kwargs):
		self.canvas.save(save_path, **kwargs)

	def paste_patch(self, patch):
		assert patch.size[0] == self.patch_size
		assert patch.size[1] == self.patch_size
		self.canvas.paste(patch.resize(tuple([self.downscaled_patch_size, self.downscaled_patch_size])), tuple(self.coord))
		self.increment_coord()

	def get_painting(self):
		return self.canvas

class Contour_Checking_fn(object):
	# Defining __call__ method 
	def __call__(self, pt): 
		raise NotImplementedError

class isInContourV1(Contour_Checking_fn):
	def __init__(self, contour):
		self.cont = contour

	def __call__(self, pt): 
		return 1 if cv2.pointPolygonTest(self.cont, pt, False) >= 0 else 0

class isInContourV2(Contour_Checking_fn):
	def __init__(self, contour, patch_size):
		self.cont = contour
		self.patch_size = patch_size

	def __call__(self, pt): 
		return 1 if cv2.pointPolygonTest(self.cont, (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2), False) >= 0 else 0

# Easy version of 4pt contour checking function - 1 of 4 points need to be in the contour for test to pass
class isInContourV3_Easy(Contour_Checking_fn):
	def __init__(self, contour, patch_size, center_shift=0.5):
		self.cont = contour
		self.patch_size = patch_size
		self.shift = int(patch_size//2*center_shift)
	def __call__(self, pt): 
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		if self.shift > 0:
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		else:
			all_points = [center]
		
		for points in all_points:
			if cv2.pointPolygonTest(self.cont, points, False) >= 0:
				return 1
		return 0

# In /project/hnguyen2/mvu9/folder_04_ma/ViLa-MIL/wsi_core/util_classes.py
#lated ---- 
import numpy as np

class isInContourV3_Easy:
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.cont = contour
        self.patch_size = patch_size
        self.center_shift = center_shift

    def __call__(self, pt):
        # Validate pt
        if not isinstance(pt, (list, tuple, np.ndarray)) or len(pt) != 2:
            # print(f"Invalid pt: {pt}")
            return False
        if not all(isinstance(x, (int, float, np.integer, np.floating)) and not np.isnan(x) for x in pt):
            # print(f"Non-numeric pt: {pt}")
            return False

        # Convert pt to integers
        pt = [int(x) for x in pt]
        center = [pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2]
        shift = int(self.patch_size * self.center_shift)
        points = [
            [center[0] - shift, center[1] - shift],
            [center[0] + shift, center[1] - shift],
            [center[0] - shift, center[1] + shift],
            [center[0] + shift, center[1] + shift]
        ]
        for point in points:
            point_tuple = tuple(int(x) for x in point)
            # print(f"Testing point: {point_tuple}")  # Debug
            if cv2.pointPolygonTest(self.cont, point_tuple, False) >= 0:
                return True
        return False
import numpy as np

class isInContourV3_Easy(Contour_Checking_fn):
    def __init__(self, contour, patch_size, center_shift=0.0):  # Set center_shift to 0.0
        self.cont = contour
        self.patch_size = patch_size
        self.shift = int(patch_size//2*center_shift)
    def __call__(self, pt): 
        print(f"Processing pt: {pt}, Type: {type(pt)}")
        if not isinstance(pt, (list, tuple, np.ndarray)) or len(pt) != 2:
            print(f"Invalid pt: {pt}")
            return 0
        for x in pt:
            if not isinstance(x, (int, float, np.integer, np.floating)):
                print(f"Non-numeric pt: {pt}, Invalid type: {type(x)}")
                return 0
            try:
                if np.isnan(float(x)):
                    print(f"Non-numeric pt: {pt}, NaN detected")
                    return 0
            except (TypeError, ValueError):
                print(f"Non-numeric pt: {pt}, Cannot convert to float")
                return 0
        pt = [int(x) for x in pt]
        center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
        all_points = [center]  # Only check center point
        for points in all_points:
            result = cv2.pointPolygonTest(self.cont, points, False)
            print(f"Testing point: {points}, Result: {result}")
            if result >= 0:
                print(f"Accepted pt: {pt}")
                return 1
        print(f"Rejected pt: {pt}, Center point outside contour")
        return 0
    
# class isInContourV3_Easy(Contour_Checking_fn):
# 	def __init__(self, contour, patch_size, center_shift=0.5):
# 		self.cont = contour
# 		self.patch_size = patch_size
# 		self.shift = int(patch_size//2*center_shift)
# 	def __call__(self, pt): 
# 		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
# 		if self.shift > 0:
# 			all_points = [(center[0]-self.shift, center[1]-self.shift),
# 						  (center[0]+self.shift, center[1]+self.shift),
# 						  (center[0]+self.shift, center[1]-self.shift),
# 						  (center[0]-self.shift, center[1]+self.shift)
# 						  ]
# 		else:
# 			all_points = [center]
		
# 		for points in all_points:
# 			if cv2.pointPolygonTest(self.cont, points, False) >= 0:
# 				return 1
# 		return 0  


# lated end 

# class isInContourV3_Easy:
#     def __init__(self, contour, patch_size, center_shift=0.5):
#         self.cont = contour
#         self.patch_size = patch_size
#         self.center_shift = center_shift

#     def __call__(self, pt):
#         # Validate pt
#         if not isinstance(pt, (list, tuple, np.ndarray)) or len(pt) != 2:
#             print(f"Invalid pt: {pt}")
#             return False
#         if not all(isinstance(x, (int, float)) and not np.isnan(x) for x in pt):
#             print(f"Non-numeric pt: {pt}")
#             return False

#         # Convert pt to integers
#         pt = [int(x) for x in pt]
#         center = [pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2]
#         shift = int(self.patch_size * self.center_shift)  # Ensure integer shift
#         points = [
#             [center[0] - shift, center[1] - shift],
#             [center[0] + shift, center[1] - shift],
#             [center[0] - shift, center[1] + shift],
#             [center[0] + shift, center[1] + shift]
#         ]
#         for point in points:
#             # Convert to tuple of integers
#             point_tuple = tuple(int(x) for x in point)
#             print(f"Testing point: {point_tuple}")  # Debug
#             if cv2.pointPolygonTest(self.cont, point_tuple, False) >= 0:
#                 return True
#         return False
# Hard version of 4pt contour checking function - all 4 points need to be in the contour for test to pass
class isInContourV3_Hard(Contour_Checking_fn):
	def __init__(self, contour, patch_size, center_shift=0.5):
		self.cont = contour
		self.patch_size = patch_size
		self.shift = int(patch_size//2*center_shift)
	def __call__(self, pt): 
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		if self.shift > 0:
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		else:
			all_points = [center]
		
		for points in all_points:
			if cv2.pointPolygonTest(self.cont, points, False) < 0:
				return 0
		return 1



		