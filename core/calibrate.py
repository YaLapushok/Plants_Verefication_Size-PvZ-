import cv2
import numpy as np
import glob
import os

CHECKER_SIZE_MM = 10.0
PATTERN_SIZE = (7, 4)

def calibrate_camera():
    images = glob.glob(r'D:\papka\2\IT\Python\AI_plants\dataset\calib\*.jpg')
    print(f"Found {len(images)} calibration images.")
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= CHECKER_SIZE_MM

    objpoints = []
    imgpoints = []

    collected = 0
    img_shape = None
    
    # Try multiple scales for detect if original is too big
    scales = [1.0, 0.5, 0.25]

    for fname in images:
        orig_img = cv2.imread(fname)
        if orig_img is None:
            continue
            
        found = False
        for scale in scales:
            img = cv2.resize(orig_img, None, fx=scale, fy=scale)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if img_shape is None and scale == 1.0:
                img_shape = gray.shape[::-1]

            ret, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, 
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                 cv2.CALIB_CB_NORMALIZE_IMAGE + 
                                                 cv2.CALIB_CB_FAST_CHECK)

            if ret:
                # Scale corners back to original size
                corners = corners / scale
                objpoints.append(objp)
                
                # refine on original size grayscale
                orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
                corners2 = cv2.cornerSubPix(orig_gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                collected += 1
                found = True
                print(f"[OK] Found pattern in {os.path.basename(fname)} at scale {scale}")
                break
                
        if not found:
             print(f"[FAIL] Could not find pattern in {os.path.basename(fname)}")

    print(f"\nSuccessfully collected points from {collected}/{len(images)} images.")

    if collected > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
        print(f"\nCamera Matrix:\n{mtx}")
        
        mm_per_pixel_list = []
        for i in range(len(imgpoints)):
            im_pts = imgpoints[i]
            # distance between corner 0 and corner 1 (adjacent horizontally)
            pt0 = im_pts[0][0]
            pt1 = im_pts[1][0]
            dist_px = np.linalg.norm(pt0 - pt1)
            mm_per_pixel = CHECKER_SIZE_MM / dist_px
            mm_per_pixel_list.append(mm_per_pixel)

        avg_mm_per_pixel = np.mean(mm_per_pixel_list)
        print(f"\nDirect measure mm/px from calibration images: {avg_mm_per_pixel:.8f}")
    else:
        print("Calibration failed. No patterns found.")

if __name__ == '__main__':
    calibrate_camera()
