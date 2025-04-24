import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Paths to the project directories and files
PROJECT_PATH = r"C:\Users\5ji6r\ai_vs_human"
DATA_PATH = os.path.join(PROJECT_PATH, "data", "AI_Human_reduced.csv")
MODEL_SAVE_PATH = os.path.join(PROJECT_PATH, "saved_model")

# Step 1: Data Preprocessing
def preprocess_data():
    print("Loading dataset from:", DATA_PATH)
    data = pd.read_csv(DATA_PATH)

    # Print column names to debug
    print("Dataset columns:", data.columns)

    # 確認正確的列名
    if "generated" in data.columns:
        data = data.rename(columns={"generated": "label"})  # 將列名 generated 重命名為 label

    print("Initial dataset size:", data.shape)
    data = data.dropna(subset=['text', 'label'])  # 刪除含有缺失值的行
    print("Dataset size after dropping missing values:", data.shape)

    data['label'] = data['label'].astype(int)  # 確保標籤為整數類型
    print("Label values:", data['label'].unique())

    # 將數據集分為訓練集和測試集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=42
    )
    print("Split dataset: Train size =", len(train_texts), ", Test size =", len(test_texts))

    # 創建 HuggingFace 的 Dataset 格式
    datasets = {
        'train': Dataset.from_dict({'text': train_texts.tolist(), 'label': train_labels.tolist()}),
        'test': Dataset.from_dict({'text': test_texts.tolist(), 'label': test_labels.tolist()})
    }

    return datasets

if __name__ == "__main__":
    print("Preprocessing data...")
    datasets = preprocess_data()
    print("Preprocessing completed.")
