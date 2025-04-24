import pandas as pd

# 加载数据
file_path = r"C:\Users\5ji6r\ai_vs_human\data\AI_Human.csv"
data = pd.read_csv(file_path)

# 检查类别分布
print("Original class distribution:\n", data['generated'].value_counts())

# 分层随机采样 10% 的数据 (根据需求调整比例)
reduced_data = data.groupby('generated', group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))

# 检查采样后类别分布
print("Sampled class distribution:\n", reduced_data['generated'].value_counts())

# 保存缩小后的数据集
reduced_data.to_csv(r"C:\Users\5ji6r\ai_vs_human\data\AI_Human_reduced.csv", index=False)
print(f"Original data size: {len(data)}, Reduced data size: {len(reduced_data)}")
