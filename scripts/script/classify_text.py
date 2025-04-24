import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_path):
    """
    加载预训练模型和分词器
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def classify_text(text, tokenizer, model):
    """
    对输入的文本进行分类
    """
    # 对文本进行编码
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
    # 使用模型进行推理
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取预测类别
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # 定义标签
    labels = ["Human-written", "AI-generated"]
    return labels[predicted_class]

if __name__ == "__main__":
    import sys

    # 定义模型路径
    MODEL_PATH = r"C:\Users\5ji6r\ai_vs_human\saved_model"
    
    # 加载模型和分词器
    print("Loading model...")
    tokenizer, model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
    
    # 检查是否有命令行参数
    if len(sys.argv) < 2:
        print("Usage: python classify_text.py \"Your text here\"")
        sys.exit(1)
    
    # 获取命令行输入的文本
    input_text = " ".join(sys.argv[1:])
    
    # 对文本进行分类
    print("Classifying text...")
    result = classify_text(input_text, tokenizer, model)
    
    # 输出分类结果
    print(f"Text: '{input_text}' -> Predicted: {result}")
