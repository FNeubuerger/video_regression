import cv2
import numpy as np
import glob
import os
import json
import argparse
import itertools

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_verification_video(video_path, output_path, sensors, num_frames=300):
    """Generates a short video clip with sensor overlays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened() and frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw sensors
        for label, sensor_data in sensors.items():
            if label == 'meta': continue
            x, y = sensor_data['center']
            r = sensor_data.get('radius', 10)
            
            # Draw Bounding Box if available
            if 'bbox' in sensor_data:
                bx, by, bw, bh = sensor_data['bbox']
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255, 255, 0), 1)

            # Green circle for boundary
            cv2.circle(frame, (x, y), r + 5, (0, 255, 0), 2)
            # Red dot for center
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
            # Label
            label_text = f"{label}"
            cv2.putText(frame, label_text, (x - 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print(f"Saved verification video to {output_path}")

def save_static_visualization(video_path, output_path, sensors):
    """Saves a single frame with sensor overlays for quick check."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    # Use average of first 5 frames for cleaner image
    frames = []
    for _ in range(5):
        ret, frame = cap.read()
        if ret: frames.append(frame)
    cap.release()
    
    if not frames:
        return
        
    frame = np.mean(frames, axis=0).astype(np.uint8)
        
    for label, sensor_data in sensors.items():
        if label == 'meta': continue
        x, y = sensor_data['center']
        r = sensor_data.get('radius', 10)
        
        # Draw Bounding Box if available
        if 'bbox' in sensor_data:
            bx, by, bw, bh = sensor_data['bbox']
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255, 255, 0), 1)

        # Green circle for boundary
        cv2.circle(frame, (x, y), r + 5, (0, 255, 0), 2)
        # Red dot for center
        cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
        # Label
        label_text = f"{label}"
        cv2.putText(frame, label_text, (x - 10, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                   
    cv2.imwrite(output_path, frame)

def calculate_rectangularity_score(points):
    """
    Given 4 points, how close are they to forming an AXIS-ALIGNED rectangle?
    We check side lengths, diagonals, and deviation from horizontal/vertical.
    """
    if len(points) != 4:
        return 0.0
        
    # Sort points to ensure order: TL, TR, BR, BL (or similar consistent order)
    # Simple sort by Y then X usually gives TL, TR, BL, BR
    # Let's sort by Y
    pts = sorted(points, key=lambda p: p[1])
    top = sorted(pts[:2], key=lambda p: p[0])
    bot = sorted(pts[2:], key=lambda p: p[0])
    
    tl, tr = top[0], top[1]
    bl, br = bot[0], bot[1]
    
    # Check side lengths
    width_top = np.linalg.norm(np.array(tl) - np.array(tr))
    width_bot = np.linalg.norm(np.array(bl) - np.array(br))
    height_left = np.linalg.norm(np.array(tl) - np.array(bl))
    height_right = np.linalg.norm(np.array(tr) - np.array(br))
    
    # Diagonals
    diag1 = np.linalg.norm(np.array(tl) - np.array(br))
    diag2 = np.linalg.norm(np.array(tr) - np.array(bl))
    
    # Score: Minimize difference between opposite sides and diagonals
    # Normalize by average size
    avg_dim = (width_top + width_bot + height_left + height_right) / 4
    if avg_dim == 0: return 0
    
    diff_width = abs(width_top - width_bot) / avg_dim
    diff_height = abs(height_left - height_right) / avg_dim
    diff_diag = abs(diag1 - diag2) / avg_dim
    
    # Axis Alignment Penalty
    w_top_safe = width_top if width_top > 10 else 10
    w_bot_safe = width_bot if width_bot > 10 else 10
    h_left_safe = height_left if height_left > 10 else 10
    h_right_safe = height_right if height_right > 10 else 10

    dev_horiz_top = abs(tl[1] - tr[1]) / w_top_safe
    dev_horiz_bot = abs(bl[1] - br[1]) / w_bot_safe
    dev_vert_left = abs(tl[0] - bl[0]) / h_left_safe
    dev_vert_right = abs(tr[0] - br[0]) / h_right_safe
    
    alignment_penalty = (dev_horiz_top + dev_horiz_bot + dev_vert_left + dev_vert_right)

    # Score = 1 / (1 + sum_diffs)
    score = 1.0 / (1.0 + diff_width + diff_height + diff_diag + 2.0 * alignment_penalty)
    return score

def get_crop_bounds(frame, offset=180):
    """
    Finds the bright horizontal line and returns (y_start, y_end).
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
        
    # Vertical projection
    row_means = np.mean(gray, axis=1)
    peak_y = np.argmax(row_means)
    
    h, w = gray.shape
    y_start = max(0, peak_y - offset)
    y_end = min(h, peak_y + offset)
    
    return int(y_start), int(y_end), int(peak_y)

def detect_sensors_robust_cropped(video_path, threshold_val=200):
    """
    Robust detection with Dynamic Cropping.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    # Read frames for average
    frames = []
    for _ in range(15): 
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    
    if not frames:
        return {}

    avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    
    # 1. Determine Crop
    y_start, y_end, peak_y = get_crop_bounds(avg_frame, offset=180)
    
    if y_end <= y_start:
        return {}
    
    # Crop
    cropped_avg = avg_frame[y_start:y_end, :]
    gray_crop = cv2.cvtColor(cropped_avg, cv2.COLOR_BGR2GRAY)
    
    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray_crop)
    
    # 3. Process
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    min_area = 20
    max_area = 5000
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            # Coordinates are relative to crop
            cx_crop = int(x)
            cy_crop = int(y)
            radius = int(radius)
            
            if radius > 60: continue
            
            # Map back to full frame
            cx_full = cx_crop
            cy_full = cy_crop + y_start
            
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Map bbox
            by_full = by + y_start
            
            candidates.append({
                'center': (cx_full, cy_full), # Store Global Coords
                'radius': radius,
                'bbox': [bx, by_full, bw, bh],
                'area': area
            })

    # 4. Selection (same logic as before)
    if len(candidates) == 4:
        best_tuple = candidates
    elif len(candidates) > 4:
        top_candidates = sorted(candidates, key=lambda x: x['area'], reverse=True)[:10]
        best_score = -1
        best_tuple = None
        for cand_tuple in itertools.combinations(top_candidates, 4):
            points = [c['center'] for c in cand_tuple]
            score = calculate_rectangularity_score(points)
            if score > best_score:
                best_score = score
                best_tuple = cand_tuple
    else:
        best_tuple = candidates 

    final_sensors = {}
    if best_tuple:
        points = [c for c in best_tuple]
        points.sort(key=lambda x: x['center'][1])
        if len(points) < 4:
            for i, p in enumerate(points):
                final_sensors[f"Unk{i}"] = p
        else:
            top_row = sorted(points[:2], key=lambda x: x['center'][0])
            bot_row = sorted(points[2:], key=lambda x: x['center'][0])
            final_sensors["M1"] = top_row[0]
            final_sensors["M2"] = top_row[1]
            final_sensors["M3"] = bot_row[0]
            final_sensors["M4"] = bot_row[1]
            
    # Add metadata about the detected active zone
    final_sensors['meta'] = {'active_zone_y': int(peak_y), 'crop_start': int(y_start), 'crop_end': int(y_end)}

    return final_sensors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/new_data/BiT_Projekt')
    parser.add_argument('--output_dir', type=str, default='data/new_data')
    parser.add_argument('--vis_dir', type=str, default='data/new_data/sensor_vis')
    parser.add_argument('--video', action='store_true', help="Generate verification videos")
    args = parser.parse_args()

    ensure_dir(args.vis_dir)
    video_files = glob.glob(os.path.join(args.data_dir, "*.mp4"))
    
    final_output = {}
    
    print("Processing videos with Rectangularity Constraint and Active Zone cropping...")
    for video_path in sorted(video_files):
        filename = os.path.basename(video_path)
        print(f"Processing {filename}...")
        
        # Use Cropped Robust Detection
        sensor_map = detect_sensors_robust_cropped(video_path)
        
        # Strip metadata from visualization
        vis_map = {k:v for k,v in sensor_map.items() if k != 'meta'}
        
        final_output[filename] = sensor_map
        found_keys = list(vis_map.keys())
        print(f"  > Detected: {found_keys}")
        
        # Visualize Static (Always save an image)
        vis_static_out = os.path.join(args.vis_dir, f"{os.path.splitext(filename)[0]}.jpg")
        save_static_visualization(video_path, vis_static_out, vis_map)
        
        # Generate video only if flag is set OR specific debug (US_005)
        if args.video or 'US_005' in filename:
             vis_out = os.path.join(args.vis_dir, f"verified_{filename}")
             generate_verification_video(video_path, vis_out, vis_map)

    out_file = os.path.join(args.output_dir, "sensor_coordinates.json")
    with open(out_file, 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
