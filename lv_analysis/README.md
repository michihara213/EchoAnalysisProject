## Left Ventricle Detection (左心室検出)

AIモデルによるセグメンテーションと、画像処理による幾何学的推定を行い、その面積比を時系列データとして取得・記録するモジュールです。

### ⚠️ 事前準備: モデルの配置

本モジュールで使用する学習済みモデル (`mymodel_segmentation_1_0.8930.h5`) はファイルサイズが大きいため、リポジトリには含まれていません。実行前に以下の手順で配置してください。

1. **モデルのダウンロード** 以下のリポジトリ（外部出典）から学習済みモデルをダウンロードしてください。
   * **出典:** [raventan95/echo-plax-segmentation](https://github.com/raventan95/echo-plax-segmentation)
   * **対象ファイル:** `mymodel_segmentation_1_0.8930.h5`

2. **ファイルの配置** ダウンロードしたファイルを、プロジェクトルートの `models/` ディレクトリ内に保存してください。

## ファイル構成
- **`lv_research.py`**: 動画リストを読み込み、フレームごとにAIと幾何学手法の面積比を計算してCSVを出力するスクリプト。
- **`lv_segment.py`**: Kerasモデルを用いたAIセグメンテーション処理。
- **`lv_geometry.py`**: OpenCVを用いた幾何学的領域の推定。

## 実行方法
プロジェクトルートから以下のように実行します。

```bash
# モデルパスを指定して実行
python lv_analysis/lv_research.py --model "models/mymodel_segmentation.h5"