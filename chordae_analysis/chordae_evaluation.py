import os
import argparse
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO
import chordae_detect as cd

def get_args():
    """コマンドライン引数を解析し、設定値を取得する"""
    parser = argparse.ArgumentParser(description="Chordae Detection Evaluation")
    parser.add_argument("--model", type=str, default="../models/best.pt", help="Path to YOLO model")
    parser.add_argument("--pos_dir", type=str, required=True, help="Connected images dir")
    parser.add_argument("--neg_dir", type=str, required=True, help="None images dir")
    parser.add_argument("--out_dir", type=str, default="results", help="Output dir")
    return parser.parse_args()

def process_directory(model, dir_path, gt_label, output_base, mv_class_ids):
    """指定ディレクトリ内の画像を順次処理し、正誤判定結果を返す"""
    y_true, y_pred = [], []
    undetected = 0
    if not os.path.exists(dir_path):
        print(f"Warning: Directory not found -> {dir_path}")
        return [], [], 0
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [f for f in os.listdir(dir_path) if os.path.splitext(f.lower())[1] in exts]
    print(f"Processing directory: {dir_path} ({len(files)} images)")

    for fname in files:
        img_path = os.path.join(dir_path, fname)
        img = cv2.imread(img_path)
        if img is None: continue
        results = model(img, verbose=False)[0]
        best_box = None
        for box in results.boxes:
            if int(box.cls[0]) in mv_class_ids:
                conf = float(box.conf[0])
                if best_box is None or conf > best_box['conf']:
                    best_box = {'conf': conf, 'xyxy': box.xyxy[0].cpu().numpy()}
        
        save_subdir = "undetected"
        pred_label = 0
        if best_box:
            pred_label, _ = cd.analyze_chordae(img, best_box['xyxy'])
            if gt_label == 1: save_subdir = "TP" if pred_label == 1 else "FN"
            else: save_subdir = "FP" if pred_label == 1 else "TN"
            y_true.append(gt_label)
            y_pred.append(pred_label)
        else:
            undetected += 1
        
        save_dir = os.path.join(output_base, save_subdir)
        os.makedirs(save_dir, exist_ok=True)
        vis = cd.visualize_results(img, best_box['xyxy'] if best_box else [], label=pred_label if best_box else None)
        cv2.imwrite(os.path.join(save_dir, fname), vis)
    return y_true, y_pred, undetected

def main():
    """モデルの読み込みと評価プロセス全体を実行する"""
    args = get_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    mv_ids = [k for k, v in model.names.items() if "mv" in v.lower() or "mitral" in v.lower()]
    
    all_true, all_pred = [], []
    total_und = 0
    yt, yp, und = process_directory(model, args.pos_dir, 1, args.out_dir, mv_ids)
    all_true.extend(yt); all_pred.extend(yp); total_und += und
    yt, yp, und = process_directory(model, args.neg_dir, 0, args.out_dir, mv_ids)
    all_true.extend(yt); all_pred.extend(yp); total_und += und

    if not all_true:
        print("No valid images processed.")
        return
    print("\n" + "="*40 + "\n Evaluation Report\n" + "="*40)
    print(f"Total Valid: {len(all_true)}, Undetected: {total_und}")
    print(classification_report(all_true, all_pred, target_names=["None", "Connected"]))
    tn, fp, fn, tp = confusion_matrix(all_true, all_pred, labels=[0, 1]).ravel()
    print(f"TP:{tp} | FN:{fn}\nFP:{fp} | TN:{tn}\n" + "="*40)

if __name__ == "__main__":
    main()