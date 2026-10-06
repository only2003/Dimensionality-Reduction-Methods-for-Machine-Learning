# ============================================================
# 数据生成脚本：生成 300样本 × 50特征 的模拟数据集
# 数据设计：前10个特征与标签强相关，中间15个弱相关，后25个纯噪声
# 这样 LASSO 应该能把后25个噪声特征的系数压成0，实现特征筛选
# ============================================================

import numpy as np  # 导入numpy用于数值计算和随机数生成
import pandas as pd  # 导入pandas用于构建DataFrame和保存Excel

np.random.seed(42)  # 固定随机种子，保证每次运行生成的数据完全一致，方便复现

n_samples = 300  # 设置样本数量为300行
n_features = 50  # 设置特征数量为50列

# 生成50个特征的原始数据，每列服从标准正态分布(均值0，标准差1)
# X 的形状为 (300, 50)，即300行50列
X = np.random.randn(n_samples, n_features)

# 定义每个特征的真实系数（用于生成标签）
# 前10个特征：系数较大，与标签强相关（重要特征）
# 中间15个特征：系数较小，与标签弱相关（边缘特征）
# 后25个特征：系数为0，与标签无关（噪声特征）
true_coef = np.zeros(n_features)  # 初始化系数数组全为0
true_coef[0:10] = np.array([2.5, -2.2, 1.8, -1.5, 1.2, -1.0, 0.9, -0.8, 0.7, -0.6])  # 前10个强相关特征系数
true_coef[10:25] = np.array([0.3, -0.25, 0.2, -0.18, 0.15, -0.12, 0.1, -0.08, 0.07, -0.06, 0.05, -0.04, 0.03, -0.02, 0.01])  # 中间15个弱相关特征系数
# 后25个特征(索引25-49)系数保持为0，即纯噪声

# 计算线性组合：每个样本的特征值 × 对应系数，然后求和
# 加上高斯噪声模拟真实数据中的随机扰动
linear_combination = X @ true_coef + np.random.randn(n_samples) * 0.5  # @ 是矩阵乘法，0.5是噪声强度

# 根据线性组合的正负生成二分类标签：大于0为类别1，小于等于0为类别0
# 这样标签和特征之间存在真实的线性关系
y = (linear_combination > 0).astype(int)

# 将特征矩阵转换为DataFrame，列名命名为 feature_1 到 feature_50
feature_names = [f"feature_{i+1}" for i in range(n_features)]  # 生成特征列名列表
df_X = pd.DataFrame(X, columns=feature_names)  # 创建特征DataFrame

# 将标签列添加到DataFrame中，列名为 label
df_X["label"] = y  # 新增label列，值为0或1

# 保存为Excel文件，不保存行索引
output_path = "dataset_300x50.xlsx"  # 定义输出文件名
df_X.to_excel(output_path, index=False)  # 将DataFrame写入Excel，index=False表示不写入行号

# 打印数据基本信息，验证生成结果
print(f"数据已保存到: {output_path}")  # 输出保存路径
print(f"数据形状: {df_X.shape}")  # 输出数据维度（行数, 列数）
print(f"标签分布:\n{df_X['label'].value_counts()}")  # 输出类别0和类别1各有多少样本
print(f"前5行数据预览:")  # 输出预览提示
print(df_X.head())  # 打印前5行数据
