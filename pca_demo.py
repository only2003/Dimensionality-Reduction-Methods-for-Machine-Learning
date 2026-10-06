# ============================================================
# PCA 主成分分析（无监督降维）完整实现
# 数据集：dataset_300x50.xlsx（300瓶红酒 × 50项化学指标 + 1列评分）
# PCA 原理：无监督线性降维，把50个相关指标线性组合成少数几个互不相关的"主成分"
# 与LASSO的区别：LASSO是"从50个里挑28个留下"（特征选择），
#               PCA是"把50个融合成10个新维度"（特征提取），新维度是原始指标的线性组合
# ============================================================

import pandas as pd  # 导入pandas，用于读取Excel和数据处理
import numpy as np  # 导入numpy，用于数值计算
import os  # 导入os，用于创建输出文件夹和处理路径
import matplotlib.pyplot as plt  # 导入matplotlib，用于绘制可视化图表
import seaborn as sns  # 导入seaborn，用于绘制协方差矩阵热力图
from sklearn.preprocessing import StandardScaler  # 导入标准化工具，PCA必须先标准化
from sklearn.decomposition import PCA  # 导入PCA主成分分析模型
from sklearn.model_selection import train_test_split  # 导入数据集划分工具（用于后续分类对比）
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归，用于对比降维前后的分类效果
from sklearn.metrics import accuracy_score  # 导入准确率评估指标

# 设置matplotlib中文字体，防止图表中文显示为方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]  # 指定中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常问题
# 创建输出文件夹，所有运行结果都存到"PCA结果"子文件夹里
# exist_ok=True表示如果文件夹已存在就不报错，避免重复创建时报错
os.makedirs("PCA结果", exist_ok=True)  # 创建PCA结果文件夹
output_dir = "PCA结果"  # 定义输出文件夹路径变量，后续保存文件时统一使用

# ============================================================
# 第一步：读取Excel数据
# ============================================================
df = pd.read_excel("wine_real_dataset_300x50.xlsx")  # 读取红酒数据集Excel文件
print(f"原始数据形状: {df.shape}")  # 打印数据维度：(300, 51)，即300瓶酒、51列
print(f"列名: {list(df.columns)}")  # 打印所有列名，确认50项指标+1列评分

# ============================================================
# 第二步：分离特征矩阵X和标签y
# ============================================================
# 注意：PCA是无监督的，训练时只用X（50项化学指标），完全不看y（品鉴师评分）
# y只是最后用来给散点图上色、以及做分类对比时才用到
X = df.drop("品鉴师评分", axis=1)  # 删除评分列，剩下50项化学指标作为特征矩阵X
y = df["品鉴师评分"]  # 提取品鉴师评分作为标签y（PCA训练时不用，仅用于可视化和对比）
feature_names = X.columns.tolist()  # 获取50项指标的名称列表
print(f"\n特征矩阵X形状: {X.shape}")  # 打印特征矩阵形状：(300, 50)
print(f"标签向量y形状: {y.shape}")  # 打印标签形状：(300,)

# ============================================================
# 第三步：数据标准化（PCA对特征尺度极其敏感，必须标准化）
# ============================================================
# 原因：PCA找的是"方差最大"的方向，如果酒精度(8~15)和花青素(0.1~0.5)不标准化，
# 酒精度的方差天然大，PCA会误以为酒精度最重要，实际只是量纲大
# 标准化后所有指标均值0、方差1，大家公平竞争
scaler = StandardScaler()  # 创建标准化对象
X_scaled = scaler.fit_transform(X)  # fit计算每项指标的均值和标准差，transform执行标准化转换
print(f"\n标准化后前5项指标均值(近似0): {X_scaled.mean(axis=0)[:5]}")  # 验证均值接近0
print(f"标准化后前5项指标标准差(近似1): {X_scaled.std(axis=0)[:5]}")  # 验证标准差为1
# ============================================================
# 第三步半：手动计算协方差矩阵（sklearn PCA内部隐藏的中间步骤）
# ============================================================
# 协方差矩阵 C = (1/(n-1)) * X^T @ X，形状为 50×50
# 因为数据已经标准化（均值为0），所以协方差矩阵 = 相关系数矩阵
# 对角线元素是各特征的方差（标准化后应为1），非对角线是两两特征的相关系数
n_samples = X_scaled.shape[0]  # 获取样本数=300
cov_matrix = (1 / (n_samples - 1)) * X_scaled.T @ X_scaled  # 计算协方差矩阵，X_scaled.T是(50,300)，@ X_scaled是(300,50)，结果(50,50)
print(f"\n===== 协方差矩阵 =====")  # 打印分隔标题
print(f"协方差矩阵形状: {cov_matrix.shape}")  # 打印形状：(50, 50)
print(f"对角线元素前10个（应为1）: {np.diag(cov_matrix)[:10]}")  # 打印对角线，标准化后方差≈1
print(f"非对角线元素绝对值均值: {np.mean(np.abs(cov_matrix - np.eye(50))):.6f}")  # 非对角线接近0说明特征间几乎不相关
# 将协方差矩阵保存为Excel，行名和列名都是特征名称
cov_df = pd.DataFrame(cov_matrix, index=feature_names, columns=feature_names)  # 转成DataFrame，行名列名用特征名
cov_df.to_excel(os.path.join(output_dir, "covariance_matrix.xlsx"))  # 保存协方差矩阵到PCA结果文件夹
print(f"协方差矩阵已保存为: {os.path.join(output_dir, 'covariance_matrix.xlsx')}")  # 打印保存提示
# 绘制协方差矩阵热力图（左右两个子图：全图 + 前10个特征放大图）
fig_cov, axes_cov = plt.subplots(1, 2, figsize=(16, 6))  # 创建1行2列子图，总尺寸16x6英寸
# 左图：完整50×50协方差矩阵热力图
ax_cov1 = axes_cov[0]  # 取第一个子图
sns.heatmap(cov_matrix, cmap="RdBu_r", center=0, vmin=-1, vmax=1,  # 红蓝配色，以0为中心，范围-1到1
            xticklabels=5, yticklabels=5, ax=ax_cov1,  # 每5个特征显示一个刻度标签，避免拥挤
            cbar_kws={"label": "相关系数"})  # 颜色条标签
ax_cov1.set_title("50×50 协方差矩阵热力图（=相关系数矩阵）", fontsize=13)  # 设置标题
ax_cov1.set_xlabel("特征编号", fontsize=11)  # 设置x轴标签
ax_cov1.set_ylabel("特征编号", fontsize=11)  # 设置y轴标签
# 右图：前10个特征的放大热力图，显示具体数值
ax_cov2 = axes_cov[1]  # 取第二个子图
sns.heatmap(cov_matrix[:10, :10], cmap="RdBu_r", center=0, vmin=-1, vmax=1,  # 只取前10行前10列
            xticklabels=[f"f{i+1}" for i in range(10)],  # x轴标签 f1~f10
            yticklabels=[f"f{i+1}" for i in range(10)],  # y轴标签 f1~f10
            annot=True, fmt=".2f", ax=ax_cov2,  # annot=True显示数值，fmt=".2f"保留两位小数
            cbar_kws={"label": "相关系数"})  # 颜色条标签
ax_cov2.set_title("前10个特征的协方差矩阵（放大，显示数值）", fontsize=13)  # 设置标题
ax_cov2.set_xlabel("特征", fontsize=11)  # 设置x轴标签
ax_cov2.set_ylabel("特征", fontsize=11)  # 设置y轴标签
plt.tight_layout()  # 自动调整子图间距，防止标签重叠
plt.savefig(os.path.join(output_dir, "covariance_matrix_heatmap.png"), dpi=150, bbox_inches="tight")  # 保存热力图到PCA结果文件夹
print(f"协方差矩阵热力图已保存为: {os.path.join(output_dir, 'covariance_matrix_heatmap.png')}")  # 打印保存提示
plt.close(fig_cov)  # 关闭该图，避免和后面的碎石图冲突

# ============================================================
# 第四步：先拟合全部50个主成分，查看每个主成分的方差解释率
# ============================================================
# 先不指定降维到几维，让PCA生成全部50个主成分，
# 然后看每个主成分能解释多少数据方差，据此决定保留几个
pca_full = PCA(n_components=None)  # n_components=None表示保留全部主成分（50个）
pca_full.fit(X_scaled)  # 在标准化后的数据上拟合PCA，计算所有主成分

# explained_variance_ratio_ 是每个主成分解释的方差占总方差的比例
# 比如第一个主成分解释了20%的方差，第二个解释了15%，以此类推
explained_variance = pca_full.explained_variance_ratio_  # 获取每个主成分的方差解释率
cumulative_variance = np.cumsum(explained_variance)  # 计算累计方差解释率（累加）

print(f"\n前10个主成分的方差解释率:")  # 打印标题
for i in range(10):  # 只打印前10个主成分的解释率
    print(f"  PC{i+1}: {explained_variance[i]*100:.2f}%  (累计: {cumulative_variance[i]*100:.2f}%)")  # 打印单个和累计

# 找到累计方差解释率达到85%时需要多少个主成分
# np.argmax返回第一个满足条件的索引，+1是因为索引从0开始
n_components_85 = np.argmax(cumulative_variance >= 0.85) + 1  # 找到累计≥85%的主成分个数
print(f"\n累计方差解释率达到85%需要 {n_components_85} 个主成分")  # 打印结果
print(f"此时累计方差解释率: {cumulative_variance[n_components_85-1]*100:.2f}%")  # 打印实际累计值

# ============================================================
# 第五步：绘制碎石图（Scree Plot），直观展示保留几个主成分合适
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 创建1行2列的子图，总尺寸16x6英寸

# 子图1：每个主成分的方差解释率（柱状图）+ 累计方差解释率（折线图）
ax1 = axes[0]  # 取第一个子图
ax1.bar(range(1, 51), explained_variance, color="steelblue", alpha=0.7, label="单个主成分方差解释率")  # 绘制50根柱子
ax1.plot(range(1, 51), cumulative_variance, color="red", marker="o", markersize=3, linewidth=2, label="累计方差解释率")  # 绘制累计折线
ax1.axhline(y=0.85, color="green", linestyle="--", linewidth=1.5, label="85%阈值线")  # 画85%参考线
ax1.axvline(x=n_components_85, color="orange", linestyle="--", linewidth=1.5, label=f"前{n_components_85}个主成分")  # 画最佳个数参考线
ax1.set_xlabel("主成分编号 (PC1 ~ PC50)", fontsize=12)  # 设置x轴标签
ax1.set_ylabel("方差解释率", fontsize=12)  # 设置y轴标签
ax1.set_title("碎石图：各主成分方差解释率与累计解释率", fontsize=13)  # 设置标题
ax1.legend(fontsize=10)  # 显示图例
ax1.grid(alpha=0.3)  # 添加网格线

# 子图2：前两个主成分的散点图（用真实标签上色，看降维后类别是否可分）
ax2 = axes[1]  # 取第二个子图
# 先用全部主成分模型把数据投影到前两个主成分上，用于可视化
X_pca_2d = pca_full.transform(X_scaled)[:, :2]  # transform得到50维主成分，取前2维用于画图
# 绘制类别0（普通酒）的散点，用蓝色
ax2.scatter(X_pca_2d[y == 0, 0], X_pca_2d[y == 0, 1], c="blue", alpha=0.6, label="普通酒(0)", s=40)  # 普通酒散点
# 绘制类别1（好酒）的散点，用红色
ax2.scatter(X_pca_2d[y == 1, 0], X_pca_2d[y == 1, 1], c="red", alpha=0.6, label="好酒(1)", s=40)  # 好酒散点
ax2.set_xlabel(f"PC1 (解释{explained_variance[0]*100:.1f}%方差)", fontsize=12)  # x轴是第一主成分
ax2.set_ylabel(f"PC2 (解释{explained_variance[1]*100:.1f}%方差)", fontsize=12)  # y轴是第二主成分
ax2.set_title("前两个主成分的散点图（按酒的类别着色）", fontsize=13)  # 设置标题
ax2.legend(fontsize=10)  # 显示图例
ax2.grid(alpha=0.3)  # 添加网格线

plt.tight_layout()  # 自动调整子图间距
plt.savefig(os.path.join(output_dir, "pca_scree_and_2d.png"), dpi=150, bbox_inches="tight")  # 保存图片到PCA结果文件夹
print(f"\n碎石图和2D散点图已保存为: {os.path.join(output_dir, 'pca_scree_and_2d.png')}")  # 打印保存提示

# ============================================================
# 第六步：用选定的主成分个数执行真正的降维
# ============================================================
# 这里选择累计方差解释率≥85%的主成分个数 n_components_85
# 也可以手动指定，比如 n_components=10
n_components = n_components_85  # 使用自动选出的主成分个数
pca = PCA(n_components=n_components, random_state=42)  # 创建PCA模型，指定保留n个主成分
X_pca = pca.fit_transform(X_scaled)  # fit计算主成分方向，transform把原始50维投影到n维

print(f"\n===== PCA 降维结果 =====")  # 打印分隔标题
print(f"原始维度: {X_scaled.shape[1]} 维 (50项化学指标)")  # 打印原始维度
print(f"降维后维度: {X_pca.shape[1]} 维 ({n_components}个主成分)")  # 打印降维后维度
print(f"降维率: {(1 - n_components/50)*100:.1f}%")  # 计算并打印降维百分比
print(f"累计方差解释率: {pca.explained_variance_ratio_.sum()*100:.2f}%")  # 打印实际保留的方差比例
print(f"降维后数据形状: {X_pca.shape}")  # 打印降维后的数据形状：(300, n_components)

# ============================================================
# 第七步：查看主成分载荷（每个主成分由哪些原始指标主导）
# ============================================================
# components_ 是主成分的载荷矩阵，形状为 (n_components, 50)
# 每一行是一个主成分，每个元素是该主成分对应原始指标的权重（系数）
# 权重绝对值大，说明这个原始指标对该主成分贡献大
loadings = pd.DataFrame(  # 将载荷矩阵转成DataFrame，方便查看
    pca.components_.T,  # 转置后每行是一个原始指标，每列是一个主成分
    index=feature_names,  # 行名用原始指标名称
    columns=[f"PC{i+1}" for i in range(n_components)]  # 列名用PC1, PC2...
)
print(f"\n前3个主成分的载荷（权重绝对值最大的5项指标）:")  # 打印标题
for pc in [f"PC{i+1}" for i in range(min(3, n_components))]:  # 只看前3个主成分
    top_features = loadings[pc].abs().sort_values(ascending=False).head(5)  # 按权重绝对值降序，取前5
    print(f"\n  {pc} (解释{pca.explained_variance_ratio_[int(pc[2:])-1]*100:.1f}%方差):")  # 打印主成分编号和解释率
    for feat, weight in top_features.items():  # 遍历前5项指标
        sign = "+" if loadings.loc[feat, pc] > 0 else "-"  # 判断权重正负号
        print(f"    {feat}: {sign}{abs(weight):.4f}")  # 打印指标名和权重（带正负号）

# ============================================================
# 第八步：用降维后的数据做分类，对比降维前后的效果
# ============================================================
# 划分训练集和测试集（和LASSO代码保持一致的划分方式）
X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y  # 20%测试集，分层抽样，固定随机种子
)

# 用降维后的主成分训练逻辑回归分类器
logreg_pca = LogisticRegression(max_iter=1000, random_state=42)  # 创建逻辑回归模型
logreg_pca.fit(X_train_pca, y_train)  # 在降维后的训练集上训练
y_pred_pca = logreg_pca.predict(X_test_pca)  # 在降维后的测试集上预测
accuracy_pca = accuracy_score(y_test, y_pred_pca)  # 计算准确率

# 对比：用原始50维数据训练逻辑回归（不降维）
X_train_full, X_test_full, _, _ = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y  # 同样的划分
)
logreg_full = LogisticRegression(max_iter=1000, random_state=42)  # 创建逻辑回归模型
logreg_full.fit(X_train_full, y_train)  # 在原始50维训练集上训练
y_pred_full = logreg_full.predict(X_test_full)  # 在原始50维测试集上预测
accuracy_full = accuracy_score(y_test, y_pred_full)  # 计算准确率

print(f"\n===== 分类效果对比 =====")  # 打印分隔标题
print(f"原始50维数据 + 逻辑回归准确率: {accuracy_full:.4f}")  # 打印全维度准确率
print(f"PCA降维后{n_components}维 + 逻辑回归准确率: {accuracy_pca:.4f}")  # 打印降维后准确率
print(f"维度从50降到{n_components}，准确率变化: {accuracy_pca - accuracy_full:+.4f}")  # 打印准确率差值

# ============================================================
# 第九步：PCA vs LASSO 的本质区别总结
# ============================================================
print(f"\n===== PCA vs LASSO 本质区别 =====")  # 打印分隔标题
print(f"LASSO: 特征选择 —— 从50项指标里挑出28项留下，扔掉22项，留下的还是原始指标")  # LASSO特点
print(f"PCA:   特征提取 —— 把50项指标线性融合成{n_components}个新维度(主成分)，")  # PCA特点
print(f"       每个主成分是所有原始指标的加权和，不再是某一项具体指标")  # 主成分含义
print(f"       比如PC1 = 0.3×酒精度 + 0.25×单宁 - 0.2×酸度 + ...")  # 举例说明主成分是组合
print(f"LASSO有监督(看标签)，PCA无监督(不看标签，只看数据本身的方差结构)")  # 监督方式区别

# ============================================================
# 第十步：保存降维后的数据到Excel
# ============================================================
# 把降维后的主成分数据和原始标签合并保存
pca_result = pd.DataFrame(  # 创建结果DataFrame
    X_pca,  # 降维后的数据，300行 × n_components列
    columns=[f"PC{i+1}" for i in range(n_components)]  # 列名为PC1, PC2...
)
pca_result["label"] = y.values  # 添加原始标签列
pca_result.to_excel(os.path.join(output_dir, "pca_reduced_data.xlsx"), index=False)  # 保存到PCA结果文件夹，不保存行索引
print(f"\n降维后的数据已保存为: {os.path.join(output_dir, 'pca_reduced_data.xlsx')}")  # 打印保存提示

# 保存主成分载荷矩阵（每个主成分由哪些原始指标构成）
loadings.to_excel(os.path.join(output_dir, "pca_loadings.xlsx"))  # 保存载荷矩阵到PCA结果文件夹
print(f"主成分载荷矩阵已保存为: {os.path.join(output_dir, 'pca_loadings.xlsx')}")  # 打印保存提示

# ============================================================
# 最终总结
# ============================================================
print(f"\n" + "="*60)  # 打印分隔线
print(f"PCA 降维总结")  # 打印总结标题
print(f"="*60)  # 打印分隔线
print(f"  输入维度: 50 维 (50项化学指标)")  # 输入维度
print(f"  输出维度: {n_components} 维 ({n_components}个主成分)")  # 输出维度
print(f"  降维率: {(1 - n_components/50)*100:.1f}%")  # 降维率
print(f"  保留方差: {pca.explained_variance_ratio_.sum()*100:.2f}%")  # 保留的方差比例
print(f"  降维后分类准确率: {accuracy_pca:.4f}")  # 降维后分类准确率
print(f"  原始50维分类准确率: {accuracy_full:.4f}")  # 原始维度准确率
print(f"="*60)  # 打印分隔线
