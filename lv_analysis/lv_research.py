import os
import cv2
import csv
import numpy as np
import argparse

# 自作モジュールのインポート
from lv_segment import LVSegmenter
from lv_geometry import get_geometric_mask, PARAMS

# ★解析対象の動画リスト (sixvideo_research.pyのスタイルを踏襲)
VIDEO_LIST = [
    "sample1.mp4",
    "sample2.mp4",
    # 必要な動画を追加してください
]

# 判定閾値 (Research project記述の「比率」閾値)
RATIO_THRESHOLD = 0.8

def analyze_video_series(video_files, model_path):
    print("Loading AI Model...")
    segmenter = LVSegmenter(model_path)

    for video_path in video_files:
        if not os.path.exists(video_path):
            print(f"[Warning] Not found: {video_path}")
            continue
            
        print(f"Processing: {video_path}")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        csv_out = f"log_{base_name}_lv.csv" 
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(" Error: Could not open video.")
            continue
            
        with open(csv_out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "AI_Area", "Geo_Area", "Ratio", "State"])
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break

                ai_mask = segmenter.predict_mask(frame)
                ai_area = np.count_nonzero(ai_mask)

                geo_mask = get_geometric_mask(frame, PARAMS)
                geo_area = np.count_nonzero(geo_mask)

                ratio = 0.0
                if geo_area > 0 and ai_area > 0:
                    ratio = min(ai_area, geo_area) / max(ai_area, geo_area)
                
                state = "Detected" if ratio > RATIO_THRESHOLD else "Not Detected"

                writer.writerow([frame_idx, ai_area, geo_area, f"{ratio:.4f}", state])
                
                if frame_idx % 50 == 0:
                    print(f"\r Frame {frame_idx}: Ratio={ratio:.2f} ({state})", end="")
                
                frame_idx += 1
            
            print(f"\n Saved log to {csv_out}")
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../models/mymodel_segmentation.h5", help="Path to .h5 model file")
    args = parser.parse_args()

    analyze_video_series(VIDEO_LIST, args.model)