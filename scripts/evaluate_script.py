from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Paths to the project directories and files
PROJECT_PATH = r"C:\Users\5ji6r\ai_vs_human"
MODEL_SAVE_PATH = os.path.join(PROJECT_PATH, "saved_model")

def evaluate_model():
    # Load the saved model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_SAVE_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)

    # Load the test dataset
    from preprocess_script import preprocess_data
    datasets = preprocess_data()
    test_dataset = datasets['test']

    # Tokenize the test dataset
    def tokenize_function(example):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Prepare labels for evaluation
    true_labels = [example['label'] for example in test_dataset]

    # Custom compute_metrics function
    def compute_metrics(pred):
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(true_labels, preds)
        precision = precision_score(true_labels, preds, average='binary')
        recall = recall_score(true_labels, preds, average='binary')
        f1 = f1_score(true_labels, preds, average='binary')
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Evaluate the model
    trainer = Trainer(model=model)
    raw_predictions = trainer.predict(tokenized_test_dataset)
    metrics = compute_metrics(raw_predictions)

    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

    # Confusion Matrix
    preds = raw_predictions.predictions.argmax(-1)
    cm = confusion_matrix(true_labels, preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['AI', 'Human'], yticklabels=['AI', 'Human'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    print("Evaluating the model...")
    evaluate_model()
