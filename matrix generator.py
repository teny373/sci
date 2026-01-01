import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ====== 自定义类别标签 ======
label_mapping = {
    0: 'fir',
    1: 'pine',
    2: 'spruce',
    3: 'trembling aspen',
    #4: 'background'
}
labels = list(label_mapping.values())

# ====== 原始混淆矩阵数据 ======
cm_data = np.array([
    [35,2,11,1],
    [4, 7, 0, 0],
    [7, 0, 55, 2],
    [4, 0, 3, 5],
])

# ====== 自定义标注内容（带 *） ======
#annot_data = np.array(cm_data, dtype=object)
#highlight_coords = [(0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2)]
#for i, j in highlight_coords:
 #   annot_data[i][j] = "0*"

# ====== 绘制混淆矩阵 ======
plt.figure(figsize=(6, 5))
ax = sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                 xticklabels=labels, yticklabels=labels)



plt.xlabel('Predicted Label', labelpad=10)
plt.ylabel('True Label', labelpad=10)
plt.title("Confusion Matrix", pad=10)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close()

# ====== 计算指标（控制台输出） ======
y_true = []
y_pred = []
for i in range(len(cm_data)):
    for j in range(len(cm_data[i])):
        y_true.extend([i] * cm_data[i][j])
        y_pred.extend([j] * cm_data[i][j])

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average=None, zero_division=0)
recall = recall_score(y_true, y_pred, average=None, zero_division=0)
f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

# IoU per class
iou = []
for i in range(len(cm_data)):
    TP = cm_data[i][i]
    FP = cm_data[:, i].sum() - TP
    FN = cm_data[i, :].sum() - TP
    denom = TP + FP + FN
    iou_score = TP / denom if denom != 0 else 0
    iou.append(iou_score)

# ====== 打印指标 ======
print(f"Accuracy: {accuracy:.4f}")
for i, label in label_mapping.items():
    print(f"\nClass '{label}':")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall:    {recall[i]:.4f}")
    print(f"  F1-score:  {f1[i]:.4f}")
    print(f"  IoU:       {iou[i]:.4f}")
