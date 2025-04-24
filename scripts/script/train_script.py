from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import os
import matplotlib.pyplot as plt

# 设置项目目录路径
PROJECT_PATH = r"C:\Users\5ji6r\ai_vs_human"
MODEL_SAVE_PATH = os.path.join(PROJECT_PATH, "saved_model")
LOSS_PLOT_PATH = os.path.join(PROJECT_PATH, "loss_plot.png")  # 保存损失图的路径

def train_model(datasets):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    print("开始对数据集进行标注...")
    def tokenize_function(example):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

    tokenized_datasets = {
        split: datasets[split].map(tokenize_function, batched=True)
        for split in ['train', 'test']
    }
    print("数据标注完成。")

    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,  # Regularization to avoid overfitting
        save_strategy="epoch",
        logging_dir=os.path.join(PROJECT_PATH, "logs"),
        logging_steps=10,
        # If you don't use TensorBoard, remove the line below
        # report_to="tensorboard",
    )

    print("开始训练...")

    # Initialize Trainer and start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
    )

    # Start training
    trainer.train()

    print("训练完成。")

    print("将模型保存至:", MODEL_SAVE_PATH)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("模型保存成功。")

    # Plot loss curve
    logs = trainer.state.log_history
    train_losses = [log['loss'] for log in logs if 'loss' in log]

    plt.plot(train_losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")

    plt.savefig(LOSS_PLOT_PATH)
    print(f"损失曲线已保存至: {LOSS_PLOT_PATH}")

    # Display the plot (optional)
    plt.show()

if __name__ == "__main__":
    from preprocess_script import preprocess_data
    print("加载数据集...")
    datasets = preprocess_data()

    print("开始训练模型...")
    train_model(datasets)
