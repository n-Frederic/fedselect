import re
import pandas as pd

input_file = "12.12-fedavg.txt"
output_file = "12.12-fedavg.xlsx"

results = []

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# 每一轮作为一个 block（以 Round 开头）
blocks = re.split(r"📊\s*Round\s+\d+\s+Aggregated Metrics:", text)

for block in blocks:
    recall = re.search(r"Aggregated Recall:\s*([\d.]+)", block)
    precision = re.search(r"Aggregated Precision:\s*([\d.]+)", block)
    f1 = re.search(r"Aggregated F1:\s*([\d.]+)", block)
    auc = re.search(r"Average AUC:\s*([\d.]+)", block)
    prauc = re.search(r"Average PR-AUC:\s*([\d.]+)", block)

    # 只在指标齐全时记录
    if all([recall, precision, f1, auc, prauc]):
        results.append({
            "recall": float(recall.group(1)),
            "precision": float(precision.group(1)),
            "f1": float(f1.group(1)),
            "auc": float(auc.group(1)),
            "prauc": float(prauc.group(1)),
        })

df = pd.DataFrame(results)
df.to_excel(output_file, index=False)

print(f"提取完成，共 {len(df)} 条记录，已保存为 {output_file}")