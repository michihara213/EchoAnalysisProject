## Echo Analysis Project (心エコー画像解析プロジェクト)

本リポジトリは、自動探査型心エコー検査ロボット「ORIZURU」によって取得された心エコー動画・画像の品質を自動評価するための統合解析プロジェクトです。
診断に重要な「傍胸骨左室長軸断面 (PLAX)」の画像品質を定量化するため、以下の3つの解析モジュールを実装しています。

## ディレクトリ構成

各モジュールの詳細な仕様や実行方法については、リンク先の `README.md` を参照してください。

### 📂 [lv_analysis/](lv_analysis/) (Research Topic ①)
**左心室検出 (Left Ventricle Detection)**
- **目的:** 左心室が適切に描出されているかを判定すること。
- **手法:** 外部学習済みモデルによるセグメンテーションと幾何学的処理による領域の比較。

### 📂 [loop_analysis/](loop_analysis/) (Research Topic ②)
**左心室の開・閉ループ判定 (LV Open/Closed Loop Judgment)**
- **目的:** 左心室の左側の領域が閉じていないかを判定すること。
- **手法:** 扇形マスク処理、モルフォロジー演算、輪郭の階層構造解析。

### 📂 [chordae_analysis/](chordae_analysis/) (Research Topic ⑥)
**僧帽弁と腱索の繋がり検出 (MV & Chordae Connection Detection)**
- **目的:** 僧帽弁の先端が腱索と繋がっていないかを判定すること。
- **手法:** YOLOv8による物体検出、バウンディングボックス拡張による解析領域の動的定義、および輝度比率を用いた解析。

### 📂 models/
- 腱索検出で使用する学習済みモデル（`best.pt`）を格納するディレクトリです。
- ※左心室セグメンテーション用の `mymodel_segmentation_1_0.8930.h5` は、[raventan95/echo-plax-segmentation](https://github.com/raventan95/echo-plax-segmentation) からダウンロードして配置してください。
---

## 環境構築

本プロジェクトの実行に必要なライブラリを一括でインストールします。

```bash
pip install -r requirements.txt