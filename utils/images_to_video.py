import cv2
import os
import argparse
import re

def extract_frame_number(filename):
    match = re.search(r'frame_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return -1

def images_to_video(image_folder, output_video_path, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort(key=extract_frame_number)

    if not images:
        print(f"No images found in {image_folder}")
        return

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Creating video {output_video_path} from {len(images)} images...")
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    print("Video created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a sequence of images to a video.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--output_video", type=str, required=True, help="Path to the output video file.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second.")

    args = parser.parse_args()

    images_to_video(args.image_folder, args.output_video, args.fps)
