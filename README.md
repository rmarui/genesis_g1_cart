# genesis_g1_cart

[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) 上で Unitree G1 に台車（カート）を押させる歩行ポリシーを強化学習（rsl-rl）で学習するコードです。Google Colab（GPU）での実行を想定しています。

## 基本的な使い方

1. Google Colab で GPU ランタイムを選び、`g1_cart.ipynb`（平地）または `g1_cart_rough.ipynb`（不整地）を開く
2. 上から順にセルを実行する（依存パッケージのインストール、リポジトリと `mujoco_menagerie` の clone、Google Drive のマウント）
3. 学習と評価をコマンドで実行する

```bash
cd scripts

# 学習（平地・カート）
python -u g1_cart_train.py --log-root $LOG_ROOT -b 4096 --max-iterations 1500

# 途中から再開（--max-iterations は追加分のイテレーション数）
python -u g1_cart_train.py --log-root $LOG_ROOT --resume -b 4096 --max-iterations 1500

# 評価・動画出力
python g1_cart_eval.py --log-root $LOG_ROOT --demo --video g1_cart_demo.mp4
python g1_cart_eval.py --log-root $LOG_ROOT --vx 0.5 --wz 0.2 --steps 750 --record --video cart_v05.mp4
```

- `--fix-upper-body` を付けると脚 12 関節のみを学習対象にします（実験名は `*-legs` になります）。
- 不整地は `g1_cart_rough_train.py` / `g1_cart_rough_eval.py` を使います。平地の重みから始める場合は `--init-from <LOG_ROOT>/g1-cart/model_3000.pt` を指定します。

## ファイルの内容

| ファイル | 内容 |
| --- | --- |
| `g1_cart.ipynb` | 平地でカートを押す学習・評価の Colab ノートブック |
| `g1_cart_rough.ipynb` | 不整地版の Colab ノートブック |
| `scripts/g1_env.py` / `g1_train.py` / `g1_eval.py` | カートなしの G1 歩行（速度追従）の環境・学習・評価 |
| `scripts/g1_cart_env.py` / `g1_cart_train.py` / `g1_cart_eval.py` | カートを押す環境・学習・評価（平地） |
| `scripts/g1_cart_rough_train.py` / `g1_cart_rough_eval.py` (`g1_cart_rough_env.py`) | 不整地版の環境・学習・評価 |

## 出力されるもの

`--log-root` 配下の `<log-root>/<exp-name>/`（例: `g1-cart`, `g1-cart-legs`, `g1-cart-rough`）に以下が保存されます。

- `model_*.pt` : 学習済みチェックポイント
- `cfgs.pkl` : 学習時の設定（評価時に読み込まれます）

その他:

- TensorBoard のログ（ローカルのみ、`--tb-root` 配下）
- 評価時に `--record` / `--video` を指定すると mp4 動画（例: `g1_cart_demo.mp4`）が実行ディレクトリに出力されます

## 注意: Google Drive との連携

ノートブックは Colab で Google Drive をマウントし、`LOG_ROOT = /content/drive/MyDrive/g1_logs` にチェックポイントと `cfgs.pkl` を保存します。

- 実行時に Google Drive へのアクセス許可が求められます。
- Drive 上の `MyDrive/g1_logs` が作成・書き込みされます。容量に注意してください。
- Colab のセッションが切れても Drive 上に残るため、`--resume` で再開できます。
- Drive を使いたくない場合は、`LOG_ROOT` を `/content/...` などのローカルパスに変更してください（セッション終了時に消えます）。

## ライセンス

[LICENSE](LICENSE) を参照してください。
