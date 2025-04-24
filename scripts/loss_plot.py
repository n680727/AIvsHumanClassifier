import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体为英文支持的字体
rcParams['font.family'] = 'DejaVu Sans'  # 或者使用 'Arial' 等支持英文的字体

def plot_loss_curve_from_tensorboard_log(log_file, plot_path):
    # 使用 TensorBoard 读取日志文件
    for summary in tf.compat.v1.train.summary_iterator(log_file):
        for value in summary.summary.value:
            if value.tag == 'loss':  # 假设训练损失的 tag 为 'loss'
                train_losses.append(value.simple_value)

    # 绘制损失曲线
    plt.plot(train_losses)
    plt.title("Training Loss Curve")  # 英文标题
    plt.xlabel("Steps")               # 英文x轴标签
    plt.ylabel("Loss")                # 英文y轴标签

    # 保存损失曲线到文件
    plt.savefig(plot_path)
    print(f"损失曲线已保存至: {plot_path}")

    # 显示图像（如果需要）
    plt.show()

if __name__ == "__main__":
    # 设置日志文件路径
    LOG_FILE_PATH = r"C:\Users\5ji6r\ai_vs_human\logs\events.out.tfevents.1735447759.DESKTOP-VJFASQ0.36160.0"

    # 设置损失图保存路径
    LOSS_PLOT_PATH = r"C:\Users\5ji6r\ai_vs_human\loss_plot.png"

    # 存储损失值的列表
    train_losses = []

    # 从 TensorBoard 日志读取损失值并绘制损失曲线
    plot_loss_curve_from_tensorboard_log(LOG_FILE_PATH, LOSS_PLOT_PATH)
