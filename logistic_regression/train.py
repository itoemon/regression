import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from util.dataset import LogisticRegressionDataset
from util.model import LogisticRegressionModel

# CSVファイルを読み込み
df = pd.read_csv('./data/sample_data.csv')

# 特徴量とラベルに分ける
num_features = df.shape[1] - 1  # 列の総数から1を引く
X = df.drop('正解ラベル', axis=1).values
y = df['正解ラベル'].values

# データセットの作成
train_dataset = LogisticRegressionDataset(X,y)

# データローダーの作成
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# モデルのインスタンス化
model = LogisticRegressionModel(input_size=num_features)

# 損失関数と最適化手法の定義
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ここまでのコードでデータの読み込み、前処理、モデル定義が完了しました。
# 次にモデルの学習を行います。

# 学習プロセスの設定
num_epochs = 100

# 学習ループ
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for features, labels in train_loader:
        # 勾配をゼロに初期化
        optimizer.zero_grad()

        # フォワードパス
        outputs = model(features)
        labels = labels.view(-1, 1)  # ラベルの形状を調整

        # 損失を計算
        loss = criterion(outputs, labels)

        # バックプロパゲーション
        loss.backward()

        # パラメータの更新
        optimizer.step()

        total_loss += loss.item()

    # 平均損失を計算
    avg_loss = total_loss / len(train_loader)

    # 一定の間隔で損失を出力
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 学習が完了したことを示すメッセージ
print("学習完了")

# 学習したモデルの保存（オプショナル）
# 重みとバイアスを取得
weights = model.linear.weight.data
bias = model.linear.bias.data

# 重みとバイアスの表示
print("重み:", weights)
print("バイアス:", bias)
torch.save(model.state_dict(), './data/logistic_regression_model.pth')

# 以上で、ロジスティック回帰モデルの学習が完了しました。
# 必要に応じて、モデルの評価や予測を行うことができます。

