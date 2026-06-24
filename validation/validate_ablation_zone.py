import os
import cv2
import numpy as np
import glob
import argparse
from tqdm import tqdm
from skimage.metrics import adapted_rand_error

# --- CEM43 Calculation ---
def calculate_cem43(temperature_sequence, dt=1.0):
    """
    Calculate CEM43 for each pixel over a sequence of temperature maps.
    temperature_sequence: np.ndarray, shape (T, H, W), temperatures in Celsius
    dt: time step in minutes (default: 1.0)
    Returns: cem43_map, shape (H, W)
    """
    cem43 = np.zeros_like(temperature_sequence[0])
    for t_map in temperature_sequence:
        r = np.where(t_map < 43, 0.25 ** (43 - t_map), 0.5 ** (t_map - 43))
        cem43 += r * dt
    return cem43

# --- Segmentation Metrics ---
def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0

def compute_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    return 2 * intersection / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 0.0

# --- Main Validation Pipeline ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', required=True, help='Folder with ablation videos')
    parser.add_argument('--phantom_dir', required=True, help='Folder with cut-open phantom images')
    parser.add_argument('--model', required=True,
                        help='Either a name registered in utils.model_registry.MODEL_REGISTRY '
                             'or the literal string "stub" to use the grayscale baseline.')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to checkpoint .pth file. Required when --model is not "stub".')
    parser.add_argument('--cem43_thresh', type=float, default=240.0,
                        help='CEM43 threshold for ablation zone (240 = canonical necrosis).')
    parser.add_argument('--mc_samples', type=int, default=1,
                        help='Number of Monte-Carlo forward passes for UQ models. '
                             'Set >1 to also output a probabilistic ablation map.')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='validation_results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model == "stub":
        def predict_temperature_maps(video_path):
            """Fallback grayscale-scaled baseline."""
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                temp_map = gray / 255.0 * 80
                frames.append(temp_map)
            cap.release()
            return np.array(frames), None
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required when --model is not 'stub'")
        from validation.model_inference import load_model, predict_video_temperatures

        model = load_model(args.model, args.checkpoint, device=args.device)

        def predict_temperature_maps(video_path):
            return predict_video_temperatures(
                model, video_path,
                device=args.device,
                num_mc_samples=args.mc_samples,
            )

    video_files = sorted(glob.glob(os.path.join(args.video_dir, '*.mp4')))
    # Assume images are named as <video_basename>_view1.png, <video_basename>_view2.png
    results = []
    for v_path in tqdm(video_files):
        base = os.path.splitext(os.path.basename(v_path))[0]
        img_paths = sorted(glob.glob(os.path.join(args.phantom_dir, f'{base}_view*.png')))
        if not img_paths:
            continue
        # 1. Predict temperature maps for video
        temp_seq, std_seq = predict_temperature_maps(v_path)
        cem43_map = calculate_cem43(temp_seq, dt=1/30)  # 1/30 min per frame at 30 fps
        # 2. Segment predicted ablation zone
        pred_mask = cem43_map > args.cem43_thresh
        iou_list, dice_list = [], []
        for img_path in img_paths:
            gt_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = gt_img > 128  # Simple threshold, adjust as needed
            # Resize gt to match prediction grid
            if gt_mask.shape != pred_mask.shape:
                gt_mask = cv2.resize(
                    gt_mask.astype(np.uint8),
                    pred_mask.shape[::-1],
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            iou = compute_iou(pred_mask, gt_mask)
            dice = compute_dice(pred_mask, gt_mask)
            iou_list.append(iou)
            dice_list.append(dice)
            vis = np.stack(
                [pred_mask.astype(np.uint8) * 255,
                 gt_mask.astype(np.uint8) * 255,
                 np.zeros_like(gt_mask, dtype=np.uint8)],
                axis=-1,
            )
            out_path = os.path.join(args.output_dir, f'vis_{base}_{os.path.basename(img_path)}.png')
            cv2.imwrite(out_path, vis)
        results.append({
            'video': base,
            'iou_mean': float(np.mean(iou_list)),
            'dice_mean': float(np.mean(dice_list)),
            'iou_views': iou_list,
            'dice_views': dice_list,
        })
    import csv
    with open(os.path.join(args.output_dir, 'metrics.csv'), 'w', newline='') as f:
        fieldnames = ['video', 'iou_mean', 'dice_mean', 'iou_views', 'dice_views']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            r['iou_views'] = str(r['iou_views'])
            r['dice_views'] = str(r['dice_views'])
            writer.writerow(r)
    print(f"Saved validation results to {args.output_dir}")

if __name__ == "__main__":
    main()
