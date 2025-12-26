# Chordae Analysis (僧帽弁と腱索の繋がり検出)

物体検出モデル（YOLOv8）と画像処理を組み合わせ、僧帽弁（Mitral Valve）の先端が腱索（Chordae）と明瞭に繋がっているかを判定します。

## ファイル構成
- **`chordae_evaluation.py`**: 指定ディレクトリ内の画像をまとめて評価し、混同行列や評価指標を出力するスクリプト。
- **`chordae_detect.py`**: 判定ロジック（ROI定義、輝度計算）を行うコアモジュール。

## 判定ロジック
1. **物体検出:** YOLOv8を用いて「僧帽弁」の位置を特定。
2. **ROI定義:** 検出されたバウンディングボックスを基準に、左側（心尖部方向）へ「腱索探索領域」を動的に定義。
3. **輝度解析:** 僧帽弁領域と腱索領域それぞれの輝度総和を計算し、比率を算出。
   - 比率が閾値 (`0.21`) 以上の場合、「Connected（繋がりあり）」と判定します。

## パラメータ説明 (`chordae_detect.py`)
`CONFIG` 辞書にて解析条件を調整可能です。

| パラメータ | 説明 | デフォルト値 |
| --- | --- | --- |
| `INTENSITY_RATIO_THRESH` | 判定しきい値（輝度比率） | 0.21 |
| `NOISE_THRESH_LEFT` | 腱索側のノイズ除去しきい値 | 30 |
| `NOISE_THRESH_VALVE` | 僧帽弁側のノイズ除去しきい値 | 30 |
| `BOX_EXTEND_Y` | 解析領域のY方向拡張率 | 0.3 |
| `BOX_SHRINK_X` | 解析領域のX方向収縮率 | 0.3 |

## 実行方法
プロジェクトルートから以下のように実行します。

```bash
# 正解ラベルごとのディレクトリを指定して精度評価を実行
python chordae_analysis/chordae_evaluation.py \
    --model "models/best.pt" \
    --pos_dir "data/connected_images" \
    --neg_dir "data/none_images"