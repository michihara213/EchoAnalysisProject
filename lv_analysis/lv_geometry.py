import cv2
import numpy as np

# 解析パラメータ設定 (echo_loop.pyの構造に合わせ、値を設定)
PARAMS = {
    'blur_ksize': 21,                # 平滑化カーネルサイズ
    'threshold': 120,                # 二値化しきい値
    'morph_ksize': 7,                # モルフォロジー演算カーネルサイズ
    'min_area': 300,                 # 最小面積しきい値 (ノイズ除去用)
    'pos_ratio': 0.45,               # 探索範囲のY座標比率（画面上部何%以下か）
    'fan_center': (315, 62),         # 扇形の中心座標 (cx, cy)
    'fan_r_range': (0, 310),         # 扇形の半径範囲 (min, max)
    'fan_angles': (-300.0/259.0, 287.0/274.0) # 扇形の左右の傾き (slope_L, slope_R)
}

def create_fan_mask(w: int, h: int, params: dict) -> np.ndarray:
    """解析パラメータに基づいて扇形のマスク画像を生成する"""
    cx, cy = params['fan_center']
    r_min, r_max = params['fan_r_range']
    aL, aR = params['fan_angles']
    
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    X = xg - cx
    
    inside_lines = (yg >= aL * X + cy) & (yg >= aR * X + cy)
    y_min = np.sqrt(np.clip(r_min**2 - X**2, 0, None)) + cy
    y_max = np.sqrt(np.clip(r_max**2 - X**2, 0, None)) + cy
    inside_rings = (yg >= y_min) & (yg <= y_max)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[inside_lines & inside_rings] = 255
    return mask

def get_geometric_mask(frame: np.ndarray, params: dict = PARAMS) -> np.ndarray:
    """フレームから幾何学的処理（2次関数フィッティング）によるLV領域マスクを生成する"""
    h, w = frame.shape[:2]
    
    fan_mask = create_fan_mask(w, h, params)
    masked_frame = cv2.bitwise_and(frame, frame, mask=fan_mask)
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    
    bk = params['blur_ksize'] | 1 
    blur = cv2.GaussianBlur(gray, (bk, bk), 0)
    _, binary = cv2.threshold(blur, params['threshold'], 255, cv2.THRESH_BINARY)
    
    mk = params['morph_ksize'] | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    binary = cv2.erode(binary, kernel, iterations=1)
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_points_x = []
    valid_points_y = []
    search_limit_y = int(h * params['pos_ratio'])
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        cy_cnt = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
        
        if area > params['min_area'] and cy_cnt > search_limit_y:
            pts = cnt.reshape(-1, 2)
            valid_points_x.extend(pts[:, 0])
            valid_points_y.extend(pts[:, 1])
            
    geo_mask = np.zeros((h, w), dtype=np.uint8)
    
    if len(valid_points_x) > 5:
        try:
            x_pts = np.array(valid_points_x)
            y_pts = np.array(valid_points_y)
            coeffs = np.polyfit(x_pts, y_pts, 2)
            poly_func = np.poly1d(coeffs)
            
            X_grid, Y_grid = np.meshgrid(np.arange(w), np.arange(h))
            Y_curve = poly_func(X_grid)
            
            condition = (fan_mask > 0) & (Y_grid < Y_curve)
            geo_mask[condition] = 255
        except:
            pass 
            
    return geo_mask