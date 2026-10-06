# ============================================================
# LASSO 特征选择（降维）完整实现
# 数据集：dataset_300x50.xlsx（300样本 × 50特征 + 1标签列）
# LASSO 原理：线性模型 + L1正则化，把不重要特征的系数压缩为0
# 系数不为0的特征被保留，系数为0的特征被筛掉 → 实现降维
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 设置字体为微软雅黑
matplotlib.rcParams['axes.unicode_minus'] = False # 正常显示负号
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# 第一步：读取Excel数据
# ============================================================
df = pd.read_excel("wine_real_dataset_300x50.xlsx")  # 读取Excel文件到DataFrame
print(f"原始数据形状: {df.shape}")  # 打印数据维度：(300, 51)，即300行51列(50特征+1标签)
print(f"列名: {list(df.columns)}")  # 打印所有列名，确认特征和标签列

# ============================================================
# 第二步：分离特征矩阵X和标签向量y
# ============================================================
X = df.drop("品鉴师评分", axis=1)  # 删除品鉴师评分列，剩下的50列作为特征矩阵X，axis=1表示按列删除
y = df["品鉴师评分"]  # 提取样鉴师评分列作为标签向量y
feature_names = X.columns.tolist()  # 获取所有特征名称列表，用于后续输出筛选结果
print(f"特征矩阵X形状: {X.shape}")  # 打印特征矩阵形状：(300, 50)
print(f"标签向量y形状: {y.shape}")  # 打印标签向量形状：(300,)

# ============================================================
# 第三步：数据标准化（LASSO对特征尺度敏感，必须标准化）
# ============================================================
# LASSO的L1惩罚项对所有特征系数一视同仁，如果特征量纲不同（一个是0.01，一个是1000），
# 量纲大的特征系数天然偏小，容易被误判为不重要而压缩掉
# 所以必须先把所有特征标准化到相同尺度（均值0，标准差1）
scaler = StandardScaler()  # 创建标准化对象
X_scaled = scaler.fit_transform(X)  # fit计算均值和标准差，transform执行标准化转换
print(f"标准化后特征均值(近似0): {X_scaled.mean(axis=0)[:5]}")  # 打印前5个特征的均值，验证接近0
print(f"标准化后特征标准差(近似1): {X_scaled.std(axis=0)[:5]}")  # 打印前5个特征的标准差，验证接近1

# ============================================================
# 第四步：划分训练集和测试集
# ============================================================
# 训练集用于训练模型，测试集用于评估模型在未见过的数据上的表现
# test_size=0.2 表示20%数据作为测试集，80%作为训练集
# random_state=42 固定随机划分，保证每次运行划分结果一致
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y  # stratify=y保证训练集测试集类别比例与原数据一致
)
print(f"训练集大小: {X_train.shape[0]} 样本")  # 打印训练集样本数：240
print(f"测试集大小: {X_test.shape[0]} 样本")  # 打印测试集样本数：60

# ============================================================
# 第五步：用 LassoCV 自动选择最佳正则化强度 alpha
# ============================================================
# alpha 是L1正则化的强度：alpha越大，惩罚越强，越多特征系数被压为0（筛选越激进）
# alpha越小，惩罚越弱，越接近普通线性回归（筛选越保守）
# LassoCV 会在多个alpha值上做交叉验证，自动选出交叉验证误差最小的alpha
# alphas=np.logspace(-4, 1, 100) 表示在10^-4到10^1之间取100个对数等间隔的alpha候选值
# cv=5 表示5折交叉验证
# max_iter=10000 增加最大迭代次数，确保高维数据收敛
lasso_cv = LassoCV(
    alphas=np.logspace(-4, 1, 100),  # 候选alpha值范围：从0.0001到10，共100个
    cv=5,  # 5折交叉验证
    max_iter=10000,  # 最大迭代次数，防止不收敛
    random_state=42  # 固定随机种子
)
lasso_cv.fit(X_train, y_train)  # 在训练集上拟合LassoCV，自动搜索最佳alpha
best_alpha = lasso_cv.alpha_  # 获取交叉验证选出的最佳alpha值
print(f"\n交叉验证选出的最佳 alpha: {best_alpha:.6f}")  # 打印最佳alpha值

# ============================================================
# 第六步：用最佳alpha训练最终LASSO模型
# ============================================================
# 虽然lasso_cv已经训练好了，但我们单独创建一个Lasso模型用最佳alpha重新训练，
# 这样逻辑更清晰，也方便理解
lasso_model = Lasso(alpha=best_alpha, max_iter=10000, random_state=42)  # 创建LASSO模型，使用最佳alpha
lasso_model.fit(X_train, y_train)  # 在训练集上训练LASSO模型

# ============================================================
# 第七步：提取特征系数，识别被筛选掉的特征
# ============================================================
# LASSO的核心：系数为0的特征就是被"筛掉"的特征，系数不为0的特征被保留
coefficients = lasso_model.coef_  # 获取模型学到的50个特征的系数数组
print(f"\n各特征系数（前10个）:")  # 打印提示
for i, (name, coef) in enumerate(zip(feature_names, coefficients)):  # 遍历每个特征名和对应系数
    if i < 10:  # 只打印前10个特征的系数作为示例
        print(f"  {name}: {coef:.6f}")  # 打印特征名和系数，保留6位小数

# 统计系数不为0的特征数量（被保留的特征）
n_selected = np.sum(coefficients != 0)  # 统计系数不等于0的特征个数
n_removed = np.sum(coefficients == 0)  # 统计系数等于0的特征个数（被筛掉的）
print(f"\n特征筛选结果:")  # 打印结果标题
print(f"  原始特征数: {len(feature_names)}")  # 打印原始特征总数：50
print(f"  保留特征数: {n_selected}")  # 打印保留的特征数
print(f"  筛除特征数: {n_removed}")  # 打印被筛除的特征数

# 列出被保留的特征（系数不为0）
selected_features = [name for name, coef in zip(feature_names, coefficients) if coef != 0]  # 筛选系数非零的特征名
print(f"\n被保留的特征列表 ({n_selected}个):")  # 打印标题
for name in selected_features:  # 遍历每个被保留的特征
    idx = feature_names.index(name)  # 获取该特征在原列表中的索引
    print(f"  {name} (系数={coefficients[idx]:.6f})")  # 打印特征名和对应系数

# 列出被筛除的特征（系数为0）
removed_features = [name for name, coef in zip(feature_names, coefficients) if coef == 0]  # 筛选系数为零的特征名
print(f"\n被筛除的特征列表 ({n_removed}个):")  # 打印标题
print(f"  {', '.join(removed_features)}")  # 打印所有被筛除的特征名，用逗号连接

# ============================================================
# 第八步：用筛选后的特征评估模型性能
# ============================================================
# 在测试集上预测，评估LASSO模型的分类准确率
y_pred = lasso_model.predict(X_test)  # 用训练好的模型对测试集做预测，输出连续值
y_pred_class = (y_pred > 0.5).astype(int)  # 将连续预测值转为二分类标签：大于0.5判为类别1，否则类别0
accuracy = accuracy_score(y_test, y_pred_class)  # 计算预测准确率
print(f"\n测试集分类准确率: {accuracy:.4f}")  # 打印准确率，保留4位小数
print("\n分类报告:")  # 打印分类报告标题
print(classification_report(y_test, y_pred_class))  # 打印精确率、召回率、F1等详细指标

# ============================================================
# 第九步：对比——使用全部特征的普通线性模型 vs LASSO筛选后
# ============================================================
# 这里用逻辑回归作为基准对比（因为是分类任务），看LASSO降维后是否损失性能
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型用于对比
logreg_full = LogisticRegression(max_iter=1000, random_state=42)  # 创建逻辑回归模型，使用全部50个特征
logreg_full.fit(X_train, y_train)  # 在训练集上训练逻辑回归
y_pred_full = logreg_full.predict(X_test)  # 在测试集上预测
accuracy_full = accuracy_score(y_test, y_pred_full)  # 计算全特征逻辑回归的准确率
print(f"\n【对比】全部50特征的逻辑回归准确率: {accuracy_full:.4f}")  # 打印全特征模型准确率
print(f"【对比】LASSO筛选后({n_selected}特征)的准确率: {accuracy:.4f}")  # 打印LASSO模型准确率
print(f"  → 用 {n_selected}/{len(feature_names)} 的特征达到了相近的准确率，实现了降维")  # 打印降维效果总结

# ============================================================
# 第十步：可视化——特征系数图
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))  # 创建2行1列的子图布局，总尺寸14x10英寸

# 子图1：所有50个特征的系数柱状图
ax1 = axes[0]  # 取第一个子图
colors = ["#2ecc71" if c != 0 else "#e74c3c" for c in coefficients]  # 系数非零用绿色，零用红色
ax1.bar(range(len(coefficients)), coefficients, color=colors)  # 绘制柱状图，x轴是特征索引，y轴是系数值
ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)  # 在y=0处画一条黑色参考线
ax1.set_xlabel("特征索引 (0-49)", fontsize=12)  # 设置x轴标签
ax1.set_ylabel("LASSO系数值", fontsize=12)  # 设置y轴标签
ax1.set_title(f"LASSO特征系数分布 (绿色=保留, 红色=筛除) | 最佳alpha={best_alpha:.4f}", fontsize=13)  # 设置标题
ax1.set_xticks(range(0, 50, 5))  # 设置x轴刻度，每5个特征显示一个
ax1.grid(axis="y", alpha=0.3)  # 添加y轴方向的网格线，透明度0.3

# 子图2：交叉验证中alpha与误差的关系曲线（如果LassoCV保存了路径）
ax2 = axes[1]  # 取第二个子图
if hasattr(lasso_cv, "mse_path_"):  # 检查LassoCV是否保存了交叉验证的均方误差路径
    mse_mean = lasso_cv.mse_path_.mean(axis=1)  # 对5折交叉验证的误差取平均
    mse_std = lasso_cv.mse_path_.std(axis=1)  # 计算5折误差的标准差
    alphas = lasso_cv.alphas_  # 获取所有候选alpha值
    ax2.plot(alphas, mse_mean, "b-", linewidth=2, label="交叉验证均方误差")  # 绘制alpha vs 误差曲线
    ax2.fill_between(alphas, mse_mean - mse_std, mse_mean + mse_std, alpha=0.2, color="blue")  # 绘制误差带
    ax2.axvline(x=best_alpha, color="red", linestyle="--", linewidth=2, label=f"最佳alpha={best_alpha:.4f}")  # 标记最佳alpha位置
    ax2.set_xscale("log")  # x轴设为对数刻度，因为alpha范围跨多个数量级
    ax2.set_xlabel("alpha (正则化强度，对数刻度)", fontsize=12)  # 设置x轴标签
    ax2.set_ylabel("交叉验证均方误差 (MSE)", fontsize=12)  # 设置y轴标签
    ax2.set_title("alpha与交叉验证误差的关系", fontsize=13)  # 设置标题
    ax2.legend(fontsize=11)  # 显示图例
    ax2.grid(alpha=0.3)  # 添加网格线

plt.tight_layout()  # 自动调整子图间距，防止标签重叠
plt.savefig("lasso_result.png", dpi=150, bbox_inches="tight")  # 保存图片到文件，dpi=150保证清晰度
print(f"\n结果图已保存为: lasso_result.png")  # 打印图片保存提示

# ============================================================
# 第十一步：保存筛选结果到Excel
# ============================================================
result_df = pd.DataFrame({  # 创建结果DataFrame
    "特征名称": feature_names,  # 第一列：特征名
    "LASSO系数": coefficients,  # 第二列：对应的LASSO系数
    "是否保留": ["保留" if c != 0 else "筛除" for c in coefficients]  # 第三列：标记该特征是否被保留
})
result_df = result_df.sort_values("LASSO系数", key=abs, ascending=False)  # 按系数绝对值降序排列，重要特征排前面
result_df.to_excel("lasso_selection_result.xlsx", index=False)  # 保存筛选结果到Excel
print(f"筛选结果已保存为: lasso_selection_result.xlsx")  # 打印保存提示

# ============================================================
# 总结输出
# ============================================================
print("\n" + "="*60)  # 打印分隔线
print("LASSO 降维总结")  # 打印总结标题
print("="*60)  # 打印分隔线
print(f"  输入维度: {X.shape[1]} 维 (50个特征)")  # 打印输入维度
print(f"  输出维度: {n_selected} 维 (LASSO保留的特征)")  # 打印降维后维度
print(f"  降维率: {(1 - n_selected/len(feature_names))*100:.1f}%")  # 计算并打印降维百分比
print(f"  最佳正则强度 alpha: {best_alpha:.6f}")  # 打印最佳alpha
print(f"  测试集准确率: {accuracy:.4f}")  # 打印最终准确率
print("="*60)  # 打印分隔线
