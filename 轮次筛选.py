import pandas as pd

# 1. 读取 CSV 文件
# 请确保文件名和路径正确，如果文件在同一目录下直接用文件名
file_path = '12.12-fedavg.csv'

try:
    df = pd.read_csv(file_path)

    # 2. 计算每一行的平均值 (axis=1 表示按行计算)
    # 假设所有列都是数值，如果有非数值列，需要指定列名，例如:
    # cols_to_mean = ['recall', 'precision', 'f1', 'auc', 'prauc']
    # df['average'] = df[cols_to_mean].mean(axis=1)

    # 这里直接对所有列求平均（根据你的数据截图，全是数值）
    df['average'] = df.mean(axis=1)

    # 3. 找到平均值最大的那一行的索引
    best_round_index = df['average'].idxmax()

    # 4. 获取那一整行的数据
    best_row = df.loc[best_round_index]

    # 5. 打印结果
    print("=" * 30)
    print(f"表现最好的是第 {best_round_index} 轮 (Index: {best_round_index})")
    print(f"该轮平均分: {best_row['average']:.4f}")
    print("=" * 30)
    print("详细数据:")
    print(best_row)

    # 如果你想把结果保存回一个新的 CSV
    df.to_csv("2.12-fedrwa+样本数量.csv", index=False)

except FileNotFoundError:
    print(f"错误: 找不到文件 '{file_path}'，请确认文件路径。")
except Exception as e:
    print(f"发生错误: {e}")