# Echo Analysis Project (心エコー画像解析プロジェクト)

本リポジトリは、心エコー動画・画像を解析するための統合プロジェクトです。
解析タスクごとにモジュールが分かれており、拡張性を考慮した設計となっています。

## ディレクトリ構成

### 📂 chord_analysis/ (腱索断裂判定)
僧帽弁の腱索（Chord）が正常に接続されているか、断裂しているかを判定します。
- **手法:** YOLOv8による弁検出 + 輝度比率解析
- **メインスクリプト:** `chord_evaluation.py`

### 📂 loop_analysis/ (ループ動作解析)
心エコー動画から弁の開閉動作（Open / Close）を時系列で解析します。
- **手法:** 扇形マスク処理 + 輪郭階層解析
- **メインスクリプト:** `echo_loop.py` (単体), `sixvideo_research.py` (一括)

### 📂 models/
- 各解析で使用する学習済みモデル（`best.pt` 等）を格納します。

---

## 実行方法

1. 必要なライブラリをインストール
   ```bash
   pip install -r requirements.txt

2. 腱索判定 (Chord Analysis)
   python chord_analysis/chord_evaluation.py --pos_dir "path/to/pos" --neg_dir "path/to/neg"

3. ループ解析 (Loop Analysis)
   # 単体実行
   python loop_analysis/echo_loop.py "video.mp4"

   # 一括実行 (事前にスクリプト内のVIDEO_LISTを編集してください)
   cd loop_analysis
   python sixvideo_research.py