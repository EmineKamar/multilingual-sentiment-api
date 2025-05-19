# src/report.py

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Tahminleri oku
pred_df = pd.read_csv("test-results/predictions.csv")

# Rapor oluştur
report = classification_report(pred_df["label"], pred_df["prediction"], output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Klasör yoksa oluştur
os.makedirs("reports", exist_ok=True)

# Raporu CSV olarak kaydet
report_df.to_csv("reports/classification_report.csv")

# Confusion matrix görselleştir
conf_mat = confusion_matrix(pred_df["label"], pred_df["prediction"], labels=pred_df["label"].unique())
plt.figure(figsize=(6,4))
sns.heatmap(conf_mat, annot=True, fmt="d", xticklabels=pred_df["label"].unique(), yticklabels=pred_df["label"].unique())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png")
