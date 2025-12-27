import os
import numpy as np
from skimage.transform import resize
# 環境に合わせて tensorflow.keras または keras を選択してください
from keras.models import load_model
from keras import backend as K
import cv2

# モデルの入力サイズ定義
INPUT_SHAPE = (384, 384)

def dice_coef(y_true, y_pred):
    """モデルロードに必要なカスタムメトリクス (Dice係数)"""
    smooth = 1
    y_true = y_true[:,:,:,1:] 
    y_pred = y_pred[:,:,:,1:]
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean((2.*intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    """Dice係数を用いた損失関数"""
    return 1 - dice_coef(y_true, y_pred)

class LVSegmenter:
    def __init__(self, model_path: str):
        """
        モデルをロードし、セグメンテーションの準備を行う

        Note:
            このクラスは 'echo-plax-segmentation' リポジトリの学習済みモデルを利用しています。
            出典: https://github.com/raventan95/echo-plax-segmentation
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = load_model(
            model_path, 
            custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef}
        )
        print(f"Model loaded: {model_path}")

    def predict_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """OpenCV形式のフレーム(BGR)からLV領域のバイナリマスク(0 or 255)を生成する"""
        original_h, original_w = frame_bgr.shape[:2]

        if frame_bgr.ndim == 3:
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img = frame_bgr

        img_resized = resize(img, INPUT_SHAPE, preserve_range=True)
        img_input = img_resized.reshape(1, INPUT_SHAPE[0], INPUT_SHAPE[1], 1)
        
        preds = self.model.predict(img_input, verbose=0)[0,:,:,:] 
        
        pred_mask_small = np.argmax(preds, axis=2).astype(np.uint8)
        lv_mask_small = (pred_mask_small == 1).astype(np.uint8) * 255
        
        final_mask = cv2.resize(lv_mask_small, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        return final_mask