# AIvsHumanClassifier：AI 與人類文本區分項目（基於 BERT）

## 項目概述
**AIvsHumanClassifier** 是一個使用 BERT（Bidirectional Encoder Representations from Transformers）模型來區分文本為 AI 生成或人類撰寫的項目。本項目包含數據預處理、模型訓練、評估和推理的完整流程，基於 Hugging Face Transformers 庫實現，並提供視覺化工具以分析模型性能。

數據集來自 Kaggle 的 [AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text/data)，通過分層採樣縮減為原始數據的 10%（`AI_Human_reduced.csv`），以確保 AI 生成與人類撰寫文本的平衡。項目設計模塊化，各階段功能分離在獨立腳本中，便於擴展和修改。

### 局限性
由於 AI 技術進步速度飛快，特別是較新的生成式 AI 模型（如 GPT-4 或後續版本）生成的文本，可能具有更高的擬人化特徵，這使得本項目基於 BERT 的模型在區分最新 AI 生成文本時可能表現不佳。模型性能高度依賴訓練數據的代表性，對於未見過的先進 AI 生成文本，檢測準確性可能下降。

## 功能
- **數據預處理**：加載並預處理數據集，分為訓練集和測試集。
- **模型訓練**：對 `bert-base-uncased` 模型進行微調，實現二分類任務。
- **模型評估**：計算準確率、精確率、召回率和 F1 分數，並生成混淆矩陣。
- **文本推理**：對新輸入的文本進行分類，判斷其為 AI 生成或人類撰寫。
- **視覺化**：生成訓練損失曲線和混淆矩陣，幫助分析模型性能。

## 視覺化結果
以下是模型訓練和評估的視覺化結果：

### 訓練損失曲線
![loss_plot](https://github.com/user-attachments/assets/f2b49559-5b2e-45ed-b82e-23a99b8d18c1)

### 混淆矩陣
![Confusion Matrix](https://github.com/user-attachments/assets/273d7878-e25f-4f16-b8bf-afe9af152002)

## 倉庫結構
```
AIvsHumanClassifier/
├── data/
│   ├── AI_Human.csv                # 原始數據集（來自 Kaggle）
│   ├── AI_Human_reduced.csv        # 縮減數據集（原始數據的 10%）
├── figures/
│   ├── loss_plot.png               # 訓練損失曲線圖
│   ├── confusion_matrix.png        # 混淆矩陣圖
├── logs/                           # 訓練日誌（例如 TensorBoard 日誌）
├── saved_model/                    # 訓練好的 BERT 模型和分詞器
├── classify_text.py                # 用於分類新文本的腳本
├── evaluate_script.py              # 用於評估模型的腳本
├── loss_plot.py                    # 用於繪製損失曲線的腳本
├── preprocess_script.py            # 用於數據預處理的腳本
├── reduced_data.py                 # 用於縮減數據集的腳本
├── train_script.py                 # 用於訓練 BERT 模型的腳本
└── README.md                       # 項目文檔
```

## 環境要求
本項目在 Conda 環境中運行，需要以下依賴：
- Python 3.8+
- Conda（Anaconda 或 Miniconda）
- 必要的 Python 庫：
  - transformers
  - torch
  - pandas
  - scikit-learn
  - datasets
  - matplotlib
  - seaborn
  - tensorflow

## 安裝步驟
1. **克隆倉庫**：
   ```bash
   git clone https://github.com/your-username/AIvsHumanClassifier.git
   cd AIvsHumanClassifier
   ```

2. **創建並激活 Conda 環境**：
   ```bash
   conda create -n ai_vs_human python=3.8
   conda activate ai_vs_human
   ```

3. **安裝依賴**：
   ```bash
   pip install transformers torch pandas scikit-learn datasets matplotlib seaborn tensorflow
   ```
   或者，您可以創建一個 `requirements.txt` 文件，內容如下，然後運行：
   ```bash
   pip install -r requirements.txt
   ```
   **requirements.txt** 示例：
   ```
   transformers
   torch
   pandas
   scikit-learn
   datasets
   matplotlib
   seaborn
   tensorflow
   ```

4. **準備數據集**：
   - 從 [Kaggle](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text/data) 下載 `AI_Human.csv` 文件，並將其放入 `data/` 目錄。
   - 如果需要縮減數據集，請運行 `reduced_data.py` 生成 `AI_Human_reduced.csv`。

## 使用方法
### 1. 數據縮減
運行 `reduced_data.py` 對原始數據集進行分層採樣，生成縮減數據集：
```bash
python reduced_data.py
```

### 2. 數據預處理
運行 `preprocess_script.py` 對數據進行預處理並分割為訓練集和測試集：
```bash
python preprocess_script.py
```

### 3. 模型訓練
運行 `train_script.py` 訓練 BERT 模型並保存結果：
```bash
python train_script.py
```
訓練完成後，模型和分詞器將保存在 `saved_model/` 目錄，損失曲線圖保存在 `figures/loss_plot.png`。

### 4. 模型評估
運行 `evaluate_script.py` 評估模型性能，生成混淆矩陣和評估指標：
```bash
python evaluate_script.py
```
混淆矩陣圖將保存在 `figures/confusion_matrix.png`。

### 5. 文本分類
使用 `classify_text.py` 對新文本進行分類。例如：
```bash
python classify_text.py "This is a sample text to classify."
```
輸出將顯示文本預測為「人類撰寫」或「AI 生成」。

### 6. 繪製損失曲線
運行 `loss_plot.py` 從 TensorBoard 日誌生成訓練損失曲線：
```bash
python loss_plot.py
```

## 注意事項
- 確保 `AI_Human.csv` 文件存在於 `data/` 目錄中，並包含 `text` 和 `generated` 列。可從 [Kaggle](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text/data) 下載。
- 項目中的路徑（如 `PROJECT_PATH`）默認為 `C:\Users\5ji6r\ai_vs_human`，請根據實際環境修改為您的項目路徑。
- 如果不使用 TensorBoard，可從 `train_script.py` 中移除相關設置。
- 模型訓練和推理需要足夠的計算資源，建議使用 GPU 加速訓練。
- 在 Conda 環境中運行時，確保已激活 `ai_vs_human` 環境。
- 圖片文件（`loss_plot.png` 和 `confusion_matrix.png`）需位於 `figures/` 目錄下，以確保 README 正確顯示。
- 由於 AI 技術快速進步，對於較新的 AI 生成文本（如來自先進模型的文本），本模型可能無法有效區分，建議定期更新訓練數據以提升性能。
