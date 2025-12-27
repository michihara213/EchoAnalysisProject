# Chordae Analysis (僧帽弁と腱索の繋がり検出)

学習済みの物体検出モデル（YOLOv8）と画像処理を組み合わせ、僧帽弁の先端が腱索と繋がっていないかを判定します。

## ファイル構成
- **`chordae_evaluation.py`**: 指定ディレクトリ内の画像を評価し、混同行列や評価指標を出力するスクリプト。
- **`chordae_detect.py`**: 判定ロジック（ROI定義、輝度計算）を行うコアモジュール。

## 実行方法
プロジェクトルートから以下のように実行します。

```bash
# 正解ラベルごとのディレクトリを指定して精度評価を実行
python chordae_analysis/chordae_evaluation.py \
    --model "models/best.pt" \
    --pos_dir "data/connected_images" \
    --neg_dir "data/none_images"