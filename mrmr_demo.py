# ============================================================
# mRMR (minimum Redundancy Maximum Relevance) 特征选择实现
# 数据集：dataset_300x50.xlsx（300瓶红酒 × 50项化学指标 + 1列评分）
# mRMR 原理：有监督的过滤式特征选择，贪心选出"与标签最相关、彼此之间最不冗余"的特征
#   - 最大相关 (Max-Relevance)：特征与标签的互信息越大越好
#   - 最小冗余 (Min-Redundancy)：已选特征之间的互信息越小越好
#   - MID准则：score = 相关性D - 冗余性R，每次选score最大的
# 与LASSO/PCA的区别：
#   LASSO = 嵌入式（训练模型时顺便选），PCA = 无监督提取，mRMR = 过滤式（先选特征再训练）
# ============================================================

import os  # 导入os模块，用于创建文件夹和拼接路径
import pandas as pd  # 导入pandas，用于读取Excel和数据处理
import numpy as np  # 导入numpy，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib，用于绘制可视化图表
from sklearn.preprocessing import StandardScaler  # 导入标准化工具
from sklearn.feature_selection import mutual_info_classif  # 导入互信息计算工具（特征与离散标签）
from sklearn.feature_selection import mutual_info_regression  # 导入互信息计算工具（连续特征之间）
from sklearn.model_selection import train_test_split  # 导入数据集划分工具
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归，用于分类对比
from sklearn.metrics import accuracy_score  # 导入准确率评估指标

# 设置matplotlib中文字体，防止图表中文显示为方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]  # 指定中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常问题

# ============================================================
# 第零步：创建结果保存文件夹（按指定的文件夹结构）
# ============================================================
result_dir = "./mRMR结果"  # 定义结果文件夹路径，和LASSO结果、PCA结果同级
os.makedirs(result_dir, exist_ok=True)  # 创建文件夹，exist_ok=True表示已存在不报错
print(f"结果将保存到文件夹: {result_dir}")  # 打印保存路径提示

# ============================================================
# 第一步：读取Excel数据
# ============================================================
df = pd.read_excel("wine_real_dataset_300x50.xlsx")  # 读取红酒数据集Excel文件
print(f"原始数据形状: {df.shape}")  # 打印数据维度：(300, 51)

# ============================================================
# 第二步：分离特征矩阵X和标签y
# ============================================================
X = df.drop("品鉴师评分", axis=1)  # 删除评分列，剩下50项化学指标作为特征矩阵X
y = df["品鉴师评分"]  # 提取品鉴师评分作为标签y
feature_names = X.columns.tolist()  # 获取50项指标的名称列表
n_features = X.shape[1]  # 获取特征总数50
print(f"特征矩阵X形状: {X.shape}")  # 打印特征矩阵形状：(300, 50)
print(f"标签向量y形状: {y.shape}")  # 打印标签形状：(300,)

# ============================================================
# 第三步：数据标准化
# ============================================================
# 注意：mRMR用互信息选特征，互信息对单调变换不敏感，标准化不影响互信息结果
# 但后续分类对比需要标准化，所以这里统一做
scaler = StandardScaler()  # 创建标准化对象
X_scaled = scaler.fit_transform(X)  # 执行标准化，均值0方差1
print(f"标准化完成，前5项指标均值: {X_scaled.mean(axis=0)[:5]}")  # 验证均值接近0

# ============================================================
# 第四步：计算每个特征与标签y的互信息（相关性D）
# ============================================================
# mutual_info_classif 计算每个连续特征与离散标签之间的互信息
# 互信息越大，说明这个特征包含越多关于标签的信息（越相关）
# random_state固定保证结果可复现
mi_with_y = mutual_info_classif(X_scaled, y, random_state=42)  # 计算50个特征各自与标签的互信息
print(f"\n各特征与标签的互信息（前10个）:")  # 打印标题
for i in range(10):  # 只打印前10个特征的互信息
    print(f"  {feature_names[i]}: {mi_with_y[i]:.6f}")  # 打印特征名和互信息值

# 找出与标签互信息最大的特征（mRMR第一个选的就是它）
best_first_idx = np.argmax(mi_with_y)  # 找到互信息最大的特征索引
print(f"\n与标签最相关的特征: {feature_names[best_first_idx]} (互信息={mi_with_y[best_first_idx]:.6f})")  # 打印结果

# ============================================================
# 第五步：预计算所有特征两两之间的互信息（冗余性R需要用到）
# ============================================================
# mRMR每次选特征时要算"该特征与所有已选特征的互信息均值"
# 50个特征两两组合有 50×49/2 = 1225 对，提前算好存起来避免重复计算
mi_matrix = np.zeros((n_features, n_features))  # 初始化50×50的互信息矩阵，全0
for i in range(n_features):  # 遍历每个特征i
    for j in range(i+1, n_features):  # 遍历i之后的每个特征j（避免重复计算）
        # mutual_info_regression 计算两个连续变量之间的互信息
        mi_ij = mutual_info_regression(
            X_scaled[:, i:i+1],  # 特征i的数据（需要二维输入，所以用i:i+1切片）
            X_scaled[:, j],      # 特征j的数据（一维）
            random_state=42      # 固定随机种子
        )[0]  # 返回数组，取第一个元素
        mi_matrix[i, j] = mi_ij  # 存入矩阵上三角
        mi_matrix[j, i] = mi_ij  # 互信息对称，下三角也填同样的值

print(f"\n特征两两互信息矩阵计算完成，形状: {mi_matrix.shape}")  # 打印矩阵形状

# ============================================================
# 第六步：mRMR贪心选择（MID准则：score = 相关性D - 冗余性R）
# ============================================================
# 选择保留的特征数量，这里选28个（和LASSO保留数一致，方便对比）
# 也可以根据mRMR得分拐点或交叉验证来选
n_select = 28  # 指定要选出的特征数量

selected_indices = []  # 存放已选特征的索引，初始为空
remaining_indices = list(range(n_features))  # 存放待选特征索引，初始是全部50个

# 第1个特征：直接选与标签互信息最大的（因为还没有已选特征，冗余性R=0）
first_idx = remaining_indices.pop(best_first_idx)  # 从待选列表中移除最相关的特征
selected_indices.append(first_idx)  # 加入已选列表
print(f"\nmRMR选择过程:")  # 打印标题
print(f"  第1个选: {feature_names[first_idx]} (D={mi_with_y[first_idx]:.4f}, R=0, score={mi_with_y[first_idx]:.4f})")  # 打印第一个选择

# 第2到第n_select个特征：贪心选 score = D - R 最大的
for step in range(2, n_select + 1):  # 从第2个选到第28个
    best_score = -np.inf  # 初始化最佳得分为负无穷
    best_idx = -1  # 初始化最佳特征索引为-1
    best_D = 0  # 记录最佳特征的相关性D
    best_R = 0  # 记录最佳特征的冗余性R

    for idx in remaining_indices:  # 遍历每个待选特征
        D = mi_with_y[idx]  # 相关性D：该特征与标签的互信息
        # 冗余性R：该特征与所有已选特征的互信息的平均值
        R = np.mean([mi_matrix[idx, s] for s in selected_indices])  # 计算平均互信息
        score = D - R  # MID准则：相关性减去冗余性
        if score > best_score:  # 如果当前得分比历史最佳高
            best_score = score  # 更新最佳得分
            best_idx = idx  # 更新最佳特征索引
            best_D = D  # 记录对应的D
            best_R = R  # 记录对应的R

    remaining_indices.remove(best_idx)  # 从待选列表中移除选中的特征
    selected_indices.append(best_idx)  # 加入已选列表
    print(f"  第{step}个选: {feature_names[best_idx]} (D={best_D:.4f}, R={best_R:.4f}, score={best_score:.4f})")  # 打印每步选择

# ============================================================
# 第七步：整理mRMR筛选结果
# ============================================================
selected_features = [feature_names[i] for i in selected_indices]  # 被选中的特征名列表
removed_features = [feature_names[i] for i in remaining_indices]  # 被筛除的特征名列表

print(f"\n===== mRMR 特征筛选结果 =====")  # 打印分隔标题
print(f"原始特征数: {n_features}")  # 打印原始特征数
print(f"保留特征数: {len(selected_features)}")  # 打印保留特征数
print(f"筛除特征数: {len(removed_features)}")  # 打印筛除特征数
print(f"降维率: {(1 - len(selected_features)/n_features)*100:.1f}%")  # 计算降维率
print(f"\n被保留的特征（按mRMR选择顺序）:")  # 打印标题
for i, name in enumerate(selected_features):  # 遍历每个被保留的特征
    print(f"  {i+1}. {name}")  # 打印序号和特征名

# ============================================================
# 第八步：用mRMR筛选后的特征做分类，对比效果
# ============================================================
# 提取被选中的特征列
X_mrmr = X_scaled[:, selected_indices]  # 从标准化后的矩阵中只取被选中的28列

# 划分训练集和测试集（和LASSO/PCA保持一致的随机种子和比例）
X_train_mrmr, X_test_mrmr, y_train, y_test = train_test_split(
    X_mrmr, y, test_size=0.2, random_state=42, stratify=y  # 20%测试集，分层抽样
)

# 用mRMR筛选后的特征训练逻辑回归
logreg_mrmr = LogisticRegression(max_iter=1000, random_state=42)  # 创建逻辑回归模型
logreg_mrmr.fit(X_train_mrmr, y_train)  # 在筛选后的训练集上训练
y_pred_mrmr = logreg_mrmr.predict(X_test_mrmr)  # 在测试集上预测
accuracy_mrmr = accuracy_score(y_test, y_pred_mrmr)  # 计算准确率

# 对比：用全部50个特征训练逻辑回归
X_train_full, X_test_full, _, _ = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y  # 同样的划分
)
logreg_full = LogisticRegression(max_iter=1000, random_state=42)  # 创建逻辑回归模型
logreg_full.fit(X_train_full, y_train)  # 在全特征训练集上训练
y_pred_full = logreg_full.predict(X_test_full)  # 在全特征测试集上预测
accuracy_full = accuracy_score(y_test, y_pred_full)  # 计算全特征准确率

print(f"\n===== 分类效果对比 =====")  # 打印分隔标题
print(f"全部50特征 + 逻辑回归准确率: {accuracy_full:.4f}")  # 打印全特征准确率
print(f"mRMR筛选后{len(selected_features)}特征 + 逻辑回归准确率: {accuracy_mrmr:.4f}")  # 打印mRMR准确率
print(f"维度从50降到{len(selected_features)}，准确率变化: {accuracy_mrmr - accuracy_full:+.4f}")  # 打印差值

# ============================================================
# 第九步：可视化
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))  # 创建2行1列子图，总尺寸14x10英寸

# 子图1：所有50个特征与标签的互信息柱状图（相关性排序）
ax1 = axes[0]  # 取第一个子图
# 按互信息从大到小排序
sorted_idx = np.argsort(mi_with_y)[::-1]  # 得到降序排列的索引
sorted_mi = mi_with_y[sorted_idx]  # 排序后的互信息值
sorted_names = [feature_names[i] for i in sorted_idx]  # 排序后的特征名
# 被mRMR选中的用绿色，没选中的用红色
colors = ["#2ecc71" if i in selected_indices else "#e74c3c" for i in sorted_idx]  # 设置颜色
ax1.bar(range(n_features), sorted_mi, color=colors)  # 绘制柱状图
ax1.set_xlabel("特征（按与标签互信息降序排列）", fontsize=12)  # x轴标签
ax1.set_ylabel("与标签的互信息（相关性D）", fontsize=12)  # y轴标签
ax1.set_title("各特征与标签的互信息（绿色=mRMR保留，红色=筛除）", fontsize=13)  # 标题
ax1.set_xticks(range(0, n_features, 5))  # 每5个特征显示一个刻度
ax1.set_xticklabels([sorted_names[i] for i in range(0, n_features, 5)], rotation=45, ha="right")  # 刻度标签旋转
ax1.grid(axis="y", alpha=0.3)  # y轴网格线

# 子图2：mRMR选择过程中每一步的score、D、R变化
ax2 = axes[1]  # 取第二个子图
steps = list(range(1, n_select + 1))  # 步骤编号1~28
# 重新计算每一步的D和R用于画图
d_values = [mi_with_y[selected_indices[0]]]  # 第一步的D
r_values = [0.0]  # 第一步的R=0
score_values = [mi_with_y[selected_indices[0]]]  # 第一步的score=D
for step in range(1, n_select):  # 从第二步开始
    idx = selected_indices[step]  # 当前步选中的特征索引
    D = mi_with_y[idx]  # 相关性
    R = np.mean([mi_matrix[idx, s] for s in selected_indices[:step]])  # 冗余性
    d_values.append(D)  # 记录D
    r_values.append(R)  # 记录R
    score_values.append(D - R)  # 记录score
ax2.plot(steps, d_values, "g-o", markersize=4, label="相关性D (与标签互信息)")  # 绘制D曲线
ax2.plot(steps, r_values, "r-s", markersize=4, label="冗余性R (与已选特征平均互信息)")  # 绘制R曲线
ax2.plot(steps, score_values, "b-^", markersize=4, label="mRMR得分 (D-R)")  # 绘制score曲线
ax2.set_xlabel("选择步骤（第几个被选中）", fontsize=12)  # x轴标签
ax2.set_ylabel("互信息值", fontsize=12)  # y轴标签
ax2.set_title("mRMR贪心选择过程：相关性、冗余性与得分变化", fontsize=13)  # 标题
ax2.legend(fontsize=11)  # 显示图例
ax2.grid(alpha=0.3)  # 网格线

plt.tight_layout()  # 自动调整子图间距
fig_path = os.path.join(result_dir, "mrmr_result.png")  # 拼接图片保存路径
plt.savefig(fig_path, dpi=150, bbox_inches="tight")  # 保存图片
print(f"\n结果图已保存为: {fig_path}")  # 打印保存提示

# ============================================================
# 第十步：保存筛选结果到Excel（存到mRMR结果文件夹）
# ============================================================
# 结果表1：每个特征的详细信息
result_df = pd.DataFrame({  # 创建结果DataFrame
    "特征名称": feature_names,  # 特征名
    "与标签互信息(D)": mi_with_y,  # 相关性D
    "是否被mRMR保留": ["保留" if i in selected_indices else "筛除" for i in range(n_features)],  # 保留状态
    "mRMR选择顺序": [selected_indices.index(i)+1 if i in selected_indices else "-" for i in range(n_features)]  # 被选中的顺序
})
result_df = result_df.sort_values("与标签互信息(D)", ascending=False)  # 按互信息降序排列
result_path = os.path.join(result_dir, "mrmr_selection_result.xlsx")  # 拼接保存路径
result_df.to_excel(result_path, index=False)  # 保存到Excel
print(f"筛选结果已保存为: {result_path}")  # 打印保存提示

# 结果表2：mRMR筛选后的数据（只保留选中的特征列 + 标签）
mrmr_data = pd.DataFrame(  # 创建降维后的数据DataFrame
    X_mrmr,  # 只取被选中的28列
    columns=selected_features  # 列名用被选中的特征名
)
mrmr_data["label"] = y.values  # 添加原始标签列
data_path = os.path.join(result_dir, "mrmr_reduced_data.xlsx")  # 拼接保存路径
mrmr_data.to_excel(data_path, index=False)  # 保存到Excel
print(f"降维后的数据已保存为: {data_path}")  # 打印保存提示

# ============================================================
# 最终总结
# ============================================================
print(f"\n" + "="*60)  # 打印分隔线
print(f"mRMR 特征选择总结")  # 打印总结标题
print(f"="*60)  # 打印分隔线
print(f"  输入维度: {n_features} 维 (50项化学指标)")  # 输入维度
print(f"  输出维度: {len(selected_features)} 维 (mRMR选出的特征)")  # 输出维度
print(f"  降维率: {(1 - len(selected_features)/n_features)*100:.1f}%")  # 降维率
print(f"  选择准则: MID (score = 相关性D - 冗余性R)")  # 使用的准则
print(f"  mRMR筛选后分类准确率: {accuracy_mrmr:.4f}")  # mRMR准确率
print(f"  全部50特征分类准确率: {accuracy_full:.4f}")  # 全特征准确率
print(f"  结果保存文件夹: {result_dir}")  # 保存路径
print(f"="*60)  # 打印分隔线
