# Echo Analysis Project (心エコー画像解析プロジェクト)

本リポジトリは、自動探査型心エコー検査ロボット「ORIZURU」によって取得された心エコー動画・画像の品質を自動評価するための統合解析プロジェクトです。
診断に重要な「傍胸骨左室長軸断面 (PLAX)」の描出品質を定量化するため、以下の3つの解析モジュールを実装しています。

## ディレクトリ構成

各モジュールの詳細な仕様や実行方法については、リンク先の `README.md` を参照してください。

### 📂 [lv_analysis/](lv_analysis/) (Research Topic ①)
**左心室検出 (Left Ventricle Detection)**
- **目的:** 左心室が適切に描出されているか、解剖学的特徴に基づいて判定します。
- **手法:** 外部学習済みモデル（U-Netベース）によるセグメンテーションと幾何学的処理（2次関数フィッティング）による領域の一致率解析。

### 📂 [loop_analysis/](loop_analysis/) (Research Topic ②)
**左心室 開・閉ループ判定 (LV Open/Closed Loop Judgment)**
- **目的:** 左心室壁が途切れていないか（閉ループか）を判定します。
- **手法:** 扇形マスク処理、モルフォロジー演算、輪郭の階層構造解析。

### 📂 [chord_analysis/](chord_analysis/) (Research Topic ⑥)
**僧帽弁と腱索の繋がり検出 (MV & Chordae Connection Detection)**
- **目的:** 僧帽弁の先端が腱索と繋がって見えているか（断裂していないか）を判定します。
- **手法:** YOLOv8による物体検出 + ROI定義による輝度比率解析。

### 📂 models/
- 腱索検出で使用する学習済みモデル（`best.pt`）を格納するディレクトリです。

---

## 環境構築

本プロジェクトの実行に必要なライブラリを一括でインストールします。

```bash
pip install -r requirements.txt