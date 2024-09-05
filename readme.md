# Axell AI Contest 2024
https://signate.jp/competitions/1374

## ディレクトリ構造

    ├── README.md
    ├── requirements.txt
    ├── main.py                 <- 学習の実行
    ├── eval.py                 <- 検証の実行
    ├── run.ipynb               <- Goolge ColabでGPUを使用する(セキュア情報を含むため削除)
    ├── data/
    ├   ├── train/              <- 学習データ
    ├   └── test/               <- テストデータ
    ├── configs/
    ├   ├── model/              <- モデルごとの設定
    ├   └── config.yaml         <- デフォルト設定
    └── src/                    <- メインのソースコード
        ├── models/             <- モデルの定義
        ├── datasets.py         <- データセット定義
        ├── flops_caluclator.py <- モデルのパラメータ数・計算量のチェッカー
        ├── psnr_calculator.py  <- PSNR評価計算
        └── utils               <- seed値のセット

## コンテスト概要
- 一般自然画像の4x超解像度
- 約1000枚の写真が配布される
- 推論時間0.035s/imageの制限
- PSNRで評価

## 解法
### モデル
![スクリーンショット 2024-09-05 13.50.16.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3687042/c3f1042d-ec4d-901e-1c2f-13251cfa8f01.png)

ベースとなったモデルは、ESPCN,SRResNet,EDSRの３つである。<br>

入力直後の畳み込み層では、カーネルサイズとチャンネルサイズを大きくすることで、局所的な細部だけでなく、広範な領域でのテクスチャやパターンを捉えることができ、超解像に必要な広いスケールの情報を早期に取り込むことを目的としている。<br>

また、Residual Blockには、Batch Normを用いないことで精度を向上させたEDSRを参考に構築している。今回のコンペでは、推論時間0.035s/imageの制限があるため、推論時間が許す限りResidual Blaockを増やし、最終的にx5となった。<br>

アップサンプリングにはピクセルシャッフルを用いている。SRResNetやEDSRでは4x超解像度において、２回のピクセルシャッフルを通してスケール2倍を2回行うことで精度の向上を測っているが、推論時間に大きな損失を与えるため、ESPCNを参考に1回でスケール４倍にすることで計算効率を高めている。



### 結果
![スクリーンショット 2024-09-05 14.00.54.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3687042/9d789ffc-469d-ba77-0959-f595e95d66be.png)


### 学習
データは配布された画像データのみ。
学習は150epochまでしか行っていない。epochをさらに高くすることで上昇の余地はあり。
