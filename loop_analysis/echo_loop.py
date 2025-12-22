import cv2
import numpy as np
import argparse
import os

PARAMS = {
    'blur_ksize': 7, 'threshold': 20, 'morph_ksize': 7, 'iterations': 4,
    'area_thr': 2600, 'fan_center': (315, 62), 'fan_r_range': (0, 280),
    'fan_angles': (-300.0/259.0, 287.0/274.0), 'open_boundary_slope': 6.0
}

def create_fan_mask(w: int, h: int, params: dict):
    cx, cy = params['fan_center']
    r_min, r_max = params['fan_r_range']
    aL, aR = params['fan_angles']
    a_open = params['open_boundary_slope']
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    X = xs - cx
    dist = np.sqrt(X**2 + (ys - cy)**2)
    mask_ring = (dist >= r_min) & (dist <= r_max)
    mask_wedge = (ys >= aL * X + cy) & (ys >= aR * X + cy)
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[mask_ring & mask_wedge] = 255
    mask_open_wedge = (ys >= aL * X + cy) & (ys >= a_open * X + cy)
    open_region_mask = np.zeros((h, w), dtype=np.uint8)
    open_region_mask[mask_ring & mask_open_wedge] = 255
    dx = r_max / np.sqrt(1 + a_open**2)
    pt_end = (int(cx + dx), int(cy + a_open * dx))
    return full_mask, open_region_mask, pt_end

def get_contour_depths(hierarchy):
    if hierarchy is None: return []
    depths = []
    for i, h in enumerate(hierarchy[0]):
        d = 0; parent = h[3]
        while parent != -1: d += 1; parent = hierarchy[0][parent][3]
        depths.append(d)
    return depths

def process_video(input_path: str, output_prefix: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): print(f"Error: {input_path}"); return
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fan_mask, open_region_mask, boundary_pt = create_fan_mask(w, h, PARAMS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_mask = cv2.VideoWriter(f"{output_prefix}_mask.mp4", fourcc, fps, (w, h), False)
    out_overlay = cv2.VideoWriter(f"{output_prefix}_overlay.mp4", fourcc, fps, (w, h))
    k_box = (PARAMS['blur_ksize'], PARAMS['blur_ksize'])
    k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PARAMS['morph_ksize'], PARAMS['morph_ksize']))
    
    print(f"Processing: {input_path}")
    while True:
        ret, frame = cap.read()
        if not ret: break
        masked_frame = cv2.bitwise_and(frame, frame, mask=fan_mask)
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray, k_box)
        _, bin_img = cv2.threshold(blur, PARAMS['threshold'], 255, cv2.THRESH_BINARY)
        mask_closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k_morph, iterations=PARAMS['iterations'])
        cv2.line(mask_closed, PARAMS['fan_center'], boundary_pt, 255, 3, cv2.LINE_8)
        mask_closed = cv2.bitwise_and(mask_closed, open_region_mask)
        contours, hierarchy = cv2.findContours(mask_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        depths = get_contour_depths(hierarchy)
        is_closed = False
        for i, cnt in enumerate(contours):
            if (depths[i] if i < len(depths) else 0) == 1 and cv2.contourArea(cnt) > PARAMS['area_thr']:
                is_closed = True; break
        vis = cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        status = "Close" if is_closed else "Open"
        color = (0, 0, 255) if is_closed else (0, 255, 255)
        cv2.putText(vis, f"State: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        out_mask.write(mask_closed); out_overlay.write(vis)
    cap.release(); out_mask.release(); out_overlay.release()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video", help="Input video path")
    args = parser.parse_args()
    process_video(args.input_video, os.path.splitext(os.path.basename(args.input_video))[0])