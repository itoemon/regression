import pandas as pd
import statsmodels.api as sm

# サンプルデータの読み込み（あなたのデータに置き換えてください）
df = pd.read_csv('./data/sample_data.csv')

# ここでは架空のデータを作成しています
# data = {
#     'x1': [1, 2, 3, 4, 5],
#     'x2': [2, 3, 4, 5, 6],
#     'y': [0, 1, 2, 3, 4]
# }
# df = pd.DataFrame(data)

# 従属変数と独立変数を定義
X = df.drop(['Date','DailyVisitors'], axis=1)
y = df['DailyVisitors']

# X = df[['x1', 'x2']]
# y = df['y']

# 定数項の追加
X = sm.add_constant(X)

# ポアソン回帰モデルの作成とフィット
model = sm.Poisson(y, X)
results = model.fit()

# 結果の表示
print(results.summary())
