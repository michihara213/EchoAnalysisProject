# Loop Analysis (左心室の開・閉ループ判定)

心エコー動画の各フレームに対し、左心室の左側の領域が閉じていないかを時系列で解析・判定します。

## ファイル構成
- **`sixvideo_research.py`**: 複数の動画を一括で解析し、Accuracy/F1-score等の精度評価を行うスクリプト。
- **`echo_loop.py`**: 単体の動画を解析し、判定結果をオーバーレイした動画を出力するスクリプト。

## 実行方法
プロジェクトルートから以下のように実行します。

```bash
# 1. 単体動画の解析と可視化（結果動画が出力されます）
# 引数には解析したい動画のパスを指定してください
python loop_analysis/echo_loop.py "loop_analysis/Sample1.mp4"

# 2. 複数動画の一括解析（スクリプト内のリスト対象）
python loop_analysis/sixvideo_research.py