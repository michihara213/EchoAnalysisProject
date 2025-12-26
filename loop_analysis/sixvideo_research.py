import os
import csv
import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from echo_loop import create_fan_mask, get_contour_depths, PARAMS

# ★ここに実際の動画ファイル名をリストしてください
VIDEO_LIST = [
    "Sample1.mp4", 
    "Sample2.mp4", 
    "Sample3.mp4",
    "Sample4.mp4", 
    "Sample5.mp4", 
    "Sample6.mp4"
]

def analyze_video_series(video_files):
    """指定された動画リストに対し、開閉ループ判定処理を連続実行してCSVログを出力する"""
    k_box = (PARAMS['blur_ksize'], PARAMS['blur_ksize'])
    k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PARAMS['morph_ksize'], PARAMS['morph_ksize']))
    for video_path in video_files:
        if not os.path.exists(video_path):
            print(f"[Warning] Not found: {video_path}"); continue
        print(f"Processing: {video_path}")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        csv_out = f"log_{base_name}.csv"
        cap = cv2.VideoCapture(video_path)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fan_mask, open_region_mask, boundary_pt = create_fan_mask(w, h, PARAMS)
        
        with open(csv_out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "State", "MaxArea_Depth1"])
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                masked = cv2.bitwise_and(frame, frame, mask=fan_mask)
                gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
                blur = cv2.blur(gray, k_box)
                _, mask = cv2.threshold(blur, PARAMS['threshold'], 255, cv2.THRESH_BINARY)
                mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_morph, iterations=PARAMS['iterations'])
                cv2.line(mask_closed, PARAMS['fan_center'], boundary_pt, 255, 3, cv2.LINE_8)
                mask_closed = cv2.bitwise_and(mask_closed, open_region_mask)
                contours, hierarchy = cv2.findContours(mask_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                depths = get_contour_depths(hierarchy)
                max_area = 0.0
                for i, cnt in enumerate(contours):
                    if (depths[i] if i < len(depths) else 0) == 1:
                        max_area = max(max_area, cv2.contourArea(cnt))
                writer.writerow([frame_idx, "Close" if max_area > PARAMS['area_thr'] else "Open", max_area])
                frame_idx += 1
        cap.release()

def evaluate_results(video_files):
    """出力されたCSVログと正解データを比較し、精度評価（Accuracy, Precision, Recall, F1）を行う"""
    metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    print("\n--- Evaluation ---")
    for path in video_files:
        base = os.path.splitext(os.path.basename(path))[0]
        pred_file = f"log_{base}.csv"; truth_file = f"{base}_truth.csv"
        if not os.path.exists(pred_file) or not os.path.exists(truth_file): continue
        try:
            df_pred = pd.read_csv(pred_file); df_true = pd.read_csv(truth_file)
            merged = pd.merge(df_pred, df_true, left_index=True, right_index=True, suffixes=('_pred', '_true'))
            valid = merged[merged['State_true'] != 'Ignore']
            if valid.empty: continue
            y_p, y_t = valid['State_pred'], valid['State_true']
            metrics['acc'].append(accuracy_score(y_t, y_p))
            metrics['prec'].append(precision_score(y_t, y_p, pos_label='Close', zero_division=0))
            metrics['rec'].append(recall_score(y_t, y_p, pos_label='Close', zero_division=0))
            metrics['f1'].append(f1_score(y_t, y_p, pos_label='Close', zero_division=0))
            print(f"[{base}] F1: {metrics['f1'][-1]:.3f}")
        except Exception as e: print(f"Error {base}: {e}")
    print("\n--- Overall ---")
    if metrics['f1']:
        print(f"Mean F1 Score: {np.mean(metrics['f1']):.4f}")

if __name__ == "__main__":
    analyze_video_series(VIDEO_LIST)
    evaluate_results(VIDEO_LIST)