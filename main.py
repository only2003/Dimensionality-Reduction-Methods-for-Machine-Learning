# ============================================================
# mRMR第五步演示：特征两两互信息矩阵的计算过程
# 从红酒真实数据集(300×50)中选6个特征做演示，避免50×50太庞大
# 选的6个特征：总酚、总单宁、pH值、酒精度、挥发酸、钾
# 这6个特征涵盖了强相关(总酚-总单宁)、弱相关、不相关等各种情况
# ============================================================

import numpy as np  # 导入numpy用于数值计算
import pandas as pd  # 导入pandas用于读取Excel和数据处理
import matplotlib.pyplot as plt  # 导入matplotlib用于可视化
from sklearn.feature_selection import mutual_info_regression  # 导入连续变量互信息计算工具

# 设置中文字体，防止图表中文显示为方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]  # 指定中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常

# ============================================================
# 第一步：读取数据，选6个特征做演示
# ============================================================
df = pd.read_excel("wine_real_dataset_300x50.xlsx")  # 读取红酒数据集

# 选6个有代表性的特征用于演示
# 总酚和总单宁 → 强正相关(都来自葡萄皮籽)
# pH和总酸 → 强负相关(酸多则pH低)，这里用pH代替
# 酒精度 → 和多酚中等相关
# 挥发酸 → 和大部分指标弱相关
# 钾 → 矿物质，和多酚弱相关
selected_features = [
    "总酚(mg/L)",      # 特征0：多酚总量
    "总单宁(mg/L)",    # 特征1：单宁含量
    "pH值",            # 特征2：酸碱度
    "酒精度(%vol)",    # 特征3：酒精浓度
    "挥发酸(g/L醋酸)", # 特征4：挥发酸
    "钾(mg/L)"         # 特征5：钾离子
]
n_feat = len(selected_features)  # 特征数量=6

X = df[selected_features].values  # 提取这6列的数据，形状(300, 6)
print(f"选中的{n_feat}个特征:")  # 打印标题
for i, name in enumerate(selected_features):  # 遍历每个特征
    print(f"  特征{i}: {name}  范围=[{X[:,i].min():.2f}, {X[:,i].max():.2f}]  均值={X[:,i].mean():.2f}")  # 打印特征名和范围

# ============================================================
# 第二步：互信息是什么？(原理讲解)
# ============================================================
print(f"\n{'='*60}")  # 打印分隔线
print(f"互信息(Mutual Information)原理")  # 打印标题
print(f"{'='*60}")  # 打印分隔线
print(f"互信息 MI(X,Y) 衡量两个变量之间的相互依赖程度：")  # 解释定义
print(f"  MI = 0    → X和Y完全独立，知道X对预测Y毫无帮助")  # MI=0的含义
print(f"  MI 越大   → X和Y关系越强(可以是线性也可以是非线性)")  # MI大的含义
print(f"  MI ≥ 0    → 互信息永远非负")  # 非负性
print(f"\n和相关系数的区别：")  # 对比相关系数
print(f"  皮尔逊相关系数 → 只衡量线性关系，范围[-1, 1]")  # 相关系数特点
print(f"  互信息         → 衡量任何关系(线性+非线性)，范围[0, +∞)")  # 互信息特点
print(f"  两个变量如果是U型关系，相关系数≈0但互信息>0")  # 关键区别

# ============================================================
# 第三步：手动演示计算一对特征的互信息
# ============================================================
print(f"\n{'='*60}")  # 打印分隔线
print(f"演示：计算 总酚 和 总单宁 的互信息")  # 打印标题
print(f"{'='*60}")  # 打印分隔线

x0 = X[:, 0].reshape(-1, 1)  # 总酚，需要二维输入(n_samples, n_features)，所以reshape
y0 = X[:, 1]  # 总单宁，一维输入(n_samples,)

# mutual_info_regression 的输入要求：
#   X: 二维数组 (n_samples, n_features)，自变量
#   y: 一维数组 (n_samples,)，因变量
# 返回值: 每个自变量和y的互信息数组，长度=n_features
mi_01 = mutual_info_regression(x0, y0, random_state=42)  # 计算总酚和总单宁的互信息
print(f"总酚 和 总单宁 的互信息 = {mi_01[0]:.4f}")  # 打印结果
print(f"  → 互信息较大，说明总酚和总单宁强相关(都来自葡萄皮籽)")  # 解释结果

# 再算一对弱相关的：总酚 和 钾
x5 = X[:, 5].reshape(-1, 1)  # 钾
mi_05 = mutual_info_regression(x0, X[:, 5], random_state=42)  # 总酚和钾的互信息
print(f"\n总酚 和 钾 的互信息 = {mi_05[0]:.4f}")  # 打印结果
print(f"  → 互信息较小，说明总酚和钾关系弱(一个来自葡萄皮，一个是土壤矿物质)")  # 解释

# ============================================================
# 第四步：计算所有两两互信息(6×5/2=15对)
# ============================================================
print(f"\n{'='*60}")  # 打印分隔线
print(f"计算所有两两互信息：{n_feat}×{n_feat-1}/2 = {n_feat*(n_feat-1)//2}对")  # 打印对数
print(f"{'='*60}")  # 打印分隔线

# 初始化6×6的全零矩阵
mi_matrix = np.zeros((n_feat, n_feat))  # 创建6×6全零矩阵

# 只计算上三角(i<j)，避免重复计算，然后对称填充
pair_count = 0  # 计数器，记录已经算了多少对
for i in range(n_feat):  # 外层循环：第i个特征
    for j in range(i + 1, n_feat):  # 内层循环：第j个特征(j>i，只算上三角)
        # 提取第i列和第j列
        xi = X[:, i].reshape(-1, 1)  # 第i个特征，二维输入
        yj = X[:, j]  # 第j个特征，一维因变量
        # 计算互信息
        mi_ij = mutual_info_regression(xi, yj, random_state=42)[0]  # 取第一个(也是唯一一个)结果
        # 存入矩阵，对称填充
        mi_matrix[i, j] = mi_ij  # 上三角位置
        mi_matrix[j, i] = mi_ij  # 下三角位置，互信息对称：MI(i,j)=MI(j,i)
        pair_count += 1  # 计数器+1
        print(f"  第{pair_count:2d}对: {selected_features[i]:<15s} ↔ {selected_features[j]:<15s}  MI = {mi_ij:.4f}")  # 打印每对结果

print(f"\n共计算了 {pair_count} 对互信息")  # 打印总数

# ============================================================
# 第五步：展示得到的6×6互信息矩阵
# ============================================================
print(f"\n{'='*60}")  # 打印分隔线
print(f"得到的 {n_feat}×{n_feat} 互信息矩阵")  # 打印标题
print(f"{'='*60}")  # 打印分隔线

# 打印矩阵，带行名和列名
header = f"{'':<15}" + "".join([f"{name[:6]:>8}" for name in selected_features])  # 表头
print(header)  # 打印表头
print("-" * len(header))  # 打印分隔线
for i in range(n_feat):  # 遍历每行
    row_str = f"{selected_features[i][:8]:<15}"  # 行名
    for j in range(n_feat):  # 遍历每列
        row_str += f"{mi_matrix[i, j]:>8.4f}"  # 矩阵元素
    print(row_str)  # 打印行

print(f"\n矩阵特点：")  # 打印矩阵特点
print(f"  ① 对角线全为0：MI(X,X)理论上是无穷大(自己和自己完全相关)，但我们没算对角线")  # 对角线
print(f"  ② 对称矩阵：mi_matrix[i][j] = mi_matrix[j][i]，因为互信息对称")  # 对称性
print(f"  ③ 数值越大说明两个特征越相似(冗余越高)")  # 数值含义

# ============================================================
# 第六步：热力图可视化
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))  # 创建画布，尺寸8×6英寸
im = ax.imshow(mi_matrix, cmap="YlOrRd", aspect="auto")  # 绘制热力图，黄橙红配色

# 设置坐标轴标签
ax.set_xticks(range(n_feat))  # x轴刻度位置
ax.set_yticks(range(n_feat))  # y轴刻度位置
ax.set_xticklabels([name[:8] for name in selected_features], rotation=45, ha="right")  # x轴标签，旋转45度
ax.set_yticklabels([name[:8] for name in selected_features])  # y轴标签

# 在每个格子里写数值
for i in range(n_feat):  # 遍历行
    for j in range(n_feat):  # 遍历列
        text = ax.text(j, i, f"{mi_matrix[i, j]:.3f}",  # 在(i,j)位置写数值
                       ha="center", va="center",  # 居中对齐
                       color="black" if mi_matrix[i, j] < 0.15 else "white")  # 数值大的用白字，小的用黑字

plt.colorbar(im, label="互信息值")  # 添加颜色条
ax.set_title("6个特征的两两互信息矩阵热力图", fontsize=14)  # 设置标题
plt.tight_layout()  # 自动调整布局
plt.savefig("mi_matrix_demo.png", dpi=150, bbox_inches="tight")  # 保存图片
print(f"\n热力图已保存为: mi_matrix_demo.png")  # 打印保存提示

# ============================================================
# 第七步：这个矩阵在mRMR中怎么用？(核心！)
# ============================================================
print(f"\n{'='*60}")  # 打印分隔线
print(f"这个矩阵在mRMR贪心选择中怎么用？")  # 打印标题
print(f"{'='*60}")  # 打印分隔线

# 模拟mRMR的选择过程，演示如何查R值
print(f"mRMR每轮选特征时，冗余性R = 该特征与所有已选特征的互信息平均值")  # R的定义
print(f"这个平均值就是从 mi_matrix 里查出来的！\n")  # 数据来源

# 模拟：假设已选了{总酚(特征0), pH值(特征2)}，现在考虑选不选总单宁(特征1)
selected_demo = [0, 2]  # 假设已选：总酚(0)、pH值(2)
candidate = 1  # 候选：总单宁(1)

print(f"假设已选特征: {[selected_features[i] for i in selected_demo]}")  # 打印已选
print(f"候选特征: {selected_features[candidate]}")  # 打印候选
print(f"\n查 mi_matrix 计算冗余性R：")  # 打印标题
r_values = []  # 存储每个已选特征与候选的互信息
for s in selected_demo:  # 遍历每个已选特征
    mi_val = mi_matrix[candidate, s]  # 从矩阵里查候选和已选特征的互信息
    r_values.append(mi_val)  # 存入列表
    print(f"  {selected_features[candidate]} ↔ {selected_features[s]}: MI = {mi_val:.4f}")  # 打印
R = np.mean(r_values)  # 取平均，得到冗余性R
print(f"  冗余性 R = 平均值 = ({' + '.join([f'{v:.4f}' for v in r_values])}) / {len(r_values)} = {R:.4f}")  # 打印R的计算

# 再算一个候选：钾(特征5)
candidate2 = 5  # 候选：钾(5)
print(f"\n另一个候选: {selected_features[candidate2]}")  # 打印候选
r_values2 = [mi_matrix[candidate2, s] for s in selected_demo]  # 查矩阵
R2 = np.mean(r_values2)  # 冗余性
print(f"  冗余性 R = {R2:.4f}")  # 打印

print(f"\n对比：")  # 打印对比
print(f"  总单宁 R = {R:.4f} (和总酚强相关，冗余高)")  # 总单宁
print(f"  钾     R = {R2:.4f} (和已选特征都不相关，冗余低)")  # 钾
print(f"  → 如果两者D(与标签互信息)差不多，mRMR会选钾，因为冗余低")  # 结论

# ============================================================
# 第八步：对比50特征的完整矩阵
# ============================================================
print(f"\n{'='*60}")  # 打印分隔线
print(f"扩展到完整50特征矩阵")  # 打印标题
print(f"{'='*60}")  # 打印分隔线
print(f"演示用了6个特征，矩阵是6×6，算15对")  # 演示规模
print(f"真实mRMR用50个特征，矩阵是50×50，算50×49/2=1225对")  # 真实规模
print(f"计算逻辑完全一样，只是循环次数多了")  # 逻辑相同
print(f"\n为什么要提前算好存成矩阵？")  # 预计算的原因
print(f"  mRMR贪心选择要跑28轮，每轮要遍历剩余特征，每个特征要算和已选特征的平均互信息")  # 使用频率
print(f"  如果不预计算，每轮都要重复算互信息，非常慢")  # 不预计算的问题
print(f"  预计算成矩阵后，查R值就是 O(1) 的矩阵查表，速度快很多")  # 预计算的好处

print(f"\n演示完成！")  # 结束提示
