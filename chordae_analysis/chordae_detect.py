import cv2
import numpy as np
from typing import Tuple, Optional

# 解析パラメータ設定
CONFIG = {
    'INTENSITY_RATIO_THRESH': 0.21,  # 判定しきい値
    'NOISE_THRESH_LEFT': 30,         # 左側（腱索）のノイズ除去
    'NOISE_THRESH_VALVE': 30,        # 僧帽弁側のノイズ除去
    'GAUSSIAN_KERNEL': 3,            # 平滑化カーネルサイズ
    'BINARIZATION_THRESH': 35,       # 二値化しきい値
    'BOX_EXTEND_Y': 0.3,             # ROI拡張率（Y方向）
    'BOX_SHRINK_X': 0.3              # ROI収縮率（X方向）
}

def _to_gray(img: np.ndarray) -> np.ndarray:
    """必要に応じて画像をグレースケールに変換する"""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def _process_roi(roi: np.ndarray, noise_thresh: int) -> np.ndarray:
    """ROIに対して前処理（ノイズ除去、平滑化、二値化）を適用する"""
    processed = roi.copy()
    if noise_thresh > 0:
        processed[processed <= noise_thresh] = 0
    k = int(CONFIG['GAUSSIAN_KERNEL']) | 1
    if k > 0:
        processed = cv2.GaussianBlur(processed, (k, k), 0)
    _, processed = cv2.threshold(processed, CONFIG['BINARIZATION_THRESH'], 255, cv2.THRESH_BINARY)
    return processed

def _get_rois_coordinates(img_shape: Tuple[int, ...], bbox: np.ndarray) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """バウンディングボックスから解析領域の座標を算出する"""
    if bbox is None or len(bbox) < 4:
        return None, None
    h, w = img_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return (x1, y1, x2, y2), None

    shrink_px = int(width * CONFIG['BOX_SHRINK_X'])
    left_w = max(1, width - shrink_px)
    extend_px = int(height * CONFIG['BOX_EXTEND_Y'])
    lx1 = max(0, x1 - left_w)
    lx2 = max(lx1 + 1, x1)
    ly2 = min(h, y2 + extend_px)
    return (x1, y1, x2, y2), (lx1, y1, lx2, ly2)

def analyze_chordae(img: np.ndarray, bbox: np.ndarray) -> Tuple[int, float]:
    """腱索の接続状態を判定する (1=Connected, 0=None)"""
    valve_coords, left_coords = _get_rois_coordinates(img.shape, bbox)
    if valve_coords is None or left_coords is None:
        return 0, 0.0
    gray = _to_gray(img)
    vx1, vy1, vx2, vy2 = valve_coords
    valve_roi = _process_roi(gray[vy1:vy2, vx1:vx2], CONFIG['NOISE_THRESH_VALVE'])
    valve_sum = float(valve_roi.sum())
    lx1, ly1, lx2, ly2 = left_coords
    left_roi = _process_roi(gray[ly1:ly2, lx1:lx2], CONFIG['NOISE_THRESH_LEFT'])
    left_sum = float(left_roi.sum())
    if valve_sum <= 1e-6:
        return 0, 0.0
    ratio = left_sum / valve_sum
    label = 1 if ratio >= CONFIG['INTENSITY_RATIO_THRESH'] else 0
    return label, ratio

def visualize_results(img: np.ndarray, bbox: np.ndarray, label: int = None) -> np.ndarray:
    """判定結果を描画する"""
    valve_coords, left_coords = _get_rois_coordinates(img.shape, bbox)
    if valve_coords is None: return img.copy()
    vis_img = (img.copy() * 0.4).astype(np.uint8)
    gray = _to_gray(img)
    vx1, vy1, vx2, vy2 = valve_coords
    v_roi = _process_roi(gray[vy1:vy2, vx1:vx2], CONFIG['NOISE_THRESH_VALVE'])
    vis_img[vy1:vy2, vx1:vx2] = cv2.cvtColor(v_roi, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(vis_img, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)
    if left_coords:
        lx1, ly1, lx2, ly2 = left_coords
        l_roi = _process_roi(gray[ly1:ly2, lx1:lx2], CONFIG['NOISE_THRESH_LEFT'])
        vis_img[ly1:ly2, lx1:lx2] = cv2.cvtColor(l_roi, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis_img, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
    if label is not None:
        text = "Chordae: Connected" if label == 1 else "Chordae: None"
        color = (0, 0, 255) if label == 1 else (0, 255, 255)
        cv2.putText(vis_img, text, (20, vis_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(vis_img, text, (20, vis_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    return vis_img