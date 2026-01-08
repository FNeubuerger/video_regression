import cv2
import numpy as np
import glob
import os
import json
import argparse
import itertools
from tqdm import tqdm

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_crop_bounds(frame, offset=180):
    """
    Finds the bright horizontal line and returns (y_start, y_end).
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
        
    # Vertical projection (row-wise mean)
    row_means = np.mean(gray, axis=1)
    
    # Simple max might be noisy, let's use a Gaussian filtered version to find the main "band"
    # But usually the probe line is very distinct.
    peak_y = np.argmax(row_means)
    
    # Sanity check: If peak is at very top or bottom, might be an error?
    # For now, trust the max.
    
    h, w = gray.shape
    y_start = max(0, peak_y - offset)
    y_end = min(h, peak_y + offset)
    
    return int(y_start), int(y_end), int(peak_y)

def calculate_rectangularity_score(points):
    """ Axis-aligned rectangularity score (same as previous Robust logic). """
    if len(points) != 4:
        return 0.0
    
    pts = sorted(points, key=lambda p: p[1])
    top = sorted(pts[:2], key=lambda p: p[0])
    bot = sorted(pts[2:], key=lambda p: p[0])
    
    tl, tr = top[0], top[1]
    bl, br = bot[0], bot[1]
    
    width_top = np.linalg.norm(np.array(tl) - np.array(tr))
    width_bot = np.linalg.norm(np.array(bl) - np.array(br))
    height_left = np.linalg.norm(np.array(tl) - np.array(bl))
    height_right = np.linalg.norm(np.array(tr) - np.array(br))
    
    diag1 = np.linalg.norm(np.array(tl) - np.array(br))
    diag2 = np.linalg.norm(np.array(tr) - np.array(bl))
    
    avg_dim = (width_top + width_bot + height_left + height_right) / 4
    if avg_dim == 0: return 0
    
    diff_width = abs(width_top - width_bot) / avg_dim
    diff_height = abs(height_left - height_right) / avg_dim
    diff_diag = abs(diag1 - diag2) / avg_dim
    
    # Alignment penalty
    w_top_safe = width_top if width_top > 10 else 10
    w_bot_safe = width_bot if width_bot > 10 else 10
    h_left_safe = height_left if height_left > 10 else 10
    h_right_safe = height_right if height_right > 10 else 10

    dev_horiz_top = abs(tl[1] - tr[1]) / w_top_safe
    dev_horiz_bot = abs(bl[1] - br[1]) / w_bot_safe
    dev_vert_left = abs(tl[0] - bl[0]) / h_left_safe
    dev_vert_right = abs(tr[0] - br[0]) / h_right_safe
    
    alignment_penalty = (dev_horiz_top + dev_horiz_bot + dev_vert_left + dev_vert_right)

    score = 1.0 / (1.0 + diff_width + diff_height + diff_diag + 2.0 * alignment_penalty)
    return score

def detect_sensors_in_frame(frame, threshold_val=200):
    """
    Detects sensors in a single CROPPED frame.
    Returns: Dict of M1-M4 coordinates (relative to frame), or None if detection fails.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Blurring
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Thresholding
    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    min_area = 20
    max_area = 5000
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius > 60: continue
            
            bx, by, bw, bh = cv2.boundingRect(cnt)
            candidates.append({
                'center': (int(x), int(y)),
                'radius': int(radius),
                'bbox': [bx, by, bw, bh],
                'area': area
            })
            
    # Pattern Selection
    if len(candidates) >= 4:
        # Combinatorial Optimization
        top_candidates = sorted(candidates, key=lambda x: x['area'], reverse=True)[:12]
        best_score = -1
        best_tuple = None
        
        for cand_tuple in itertools.combinations(top_candidates, 4):
            points = [c['center'] for c in cand_tuple]
            score = calculate_rectangularity_score(points)
            if score > best_score:
                best_score = score
                best_tuple = cand_tuple
        
        if best_tuple and best_score > 0.1: # Score threshold
            points = [c for c in best_tuple]
            # Grid assignment
            points.sort(key=lambda x: x['center'][1])
            top_row = sorted(points[:2], key=lambda x: x['center'][0])
            bot_row = sorted(points[2:], key=lambda x: x['center'][0])
            
            return {
                "M1": top_row[0],
                "M2": top_row[1],
                "M3": bot_row[0],
                "M4": bot_row[1]
            }
            
    return None

def process_video_pipeline(input_path, output_dir, vis_dir):
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening {input_path}")
        return None

    # 1. Analyze first N frames to determine "Average Frame" and "Active Zone"
    check_frames = []
    num_check = 20
    for _ in range(num_check):
        ret, f = cap.read()
        if ret: check_frames.append(f)
        else: break
        
    if not check_frames:
        return None
        
    avg_frame = np.mean(check_frames, axis=0).astype(np.uint8)
    
    # Determine Crop
    y_start, y_end, peak_y = get_crop_bounds(avg_frame, offset=180) # 180px radius around peak
    if y_end <= y_start:
        y_start, y_end = 0, avg_frame.shape[0] # Fallback? No, this shouldn't happen.
    
    crop_h = y_end - y_start
    crop_w = avg_frame.shape[1]
    
    # 2. Detect Sensors on the AVERAGED CROPPED frame (Robust Detection)
    cropped_avg = avg_frame[y_start:y_end, :]
    detected_sensors = detect_sensors_in_frame(cropped_avg)
    
    sensors_found = False
    if detected_sensors:
        sensors_found = True
        # Save visualization of detection (first frame)
        vis_path = os.path.join(vis_dir, f"{os.path.splitext(filename)[0]}.jpg")
        
        vis_frame = cropped_avg.copy()
        for lbl, data in detected_sensors.items():
            cx, cy = data['center']
            cv2.circle(vis_frame, (cx, cy), data['radius'], (0, 255, 0), 2)
            cv2.putText(vis_frame, lbl, (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(vis_path, vis_frame)
    else:
        print(f"WARNING: Could not robustly detect sensors in {filename}")

    # 3. Process Video: Crop & Save
    # Reset video ptr
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Streaming Copy
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        cropped_frame = frame[y_start:y_end, :]
        out.write(cropped_frame)
        
    cap.release()
    out.release()
    
    return {
        'sensors': detected_sensors,
        'crop_meta': {'y_start': y_start, 'y_end': y_end, 'original_h': avg_frame.shape[0]}
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/level0_raw')
    parser.add_argument('--output_dir', default='data/level1_cropped')
    args = parser.parse_args()
    
    vis_dir = os.path.join(args.output_dir, 'vis') # Store vis inside processed folder
    ensure_dir(args.output_dir)
    ensure_dir(vis_dir)
    
    video_files = sorted(glob.glob(os.path.join(args.input_dir, "*.mp4")))
    
    results_json = {}
    
    print(f"Starting Preprocessing Pipeline: Level 0 -> Level 1 (Crop)")
    print(f"Found {len(video_files)} videos.")
    
    for v_path in tqdm(video_files):
        fname = os.path.basename(v_path)
        # Skip verification videos if they ended up here?
        if 'verified_' in fname: continue
            
        res = process_video_pipeline(v_path, args.output_dir, vis_dir)
        
        if res and res['sensors']:
            results_json[fname] = res['sensors'] # Save sensors relative to cropped frame
            results_json[fname]['meta'] = res['crop_meta']
        else:
            print(f"Failed to process {fname}")
            
    out_json_path = os.path.join(args.output_dir, 'sensor_coordinates.json')
    with open(out_json_path, 'w') as f:
        json.dump(results_json, f, indent=4)
        
    print(f"Done. Processed data saved to {args.output_dir}")
    print(f"Metadata saved to {out_json_path}")

if __name__ == "__main__":
    main()
