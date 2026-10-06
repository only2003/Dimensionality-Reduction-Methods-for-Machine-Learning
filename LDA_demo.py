# ============================================================
# LDA 线性判别分析（有监督降维）完整实现
# 数据集：wine_real_dataset_300x50.xlsx（300个样本 × 50个特征 + 1列标签）
#
# LDA 原理：利用类别标签寻找投影方向，使“类间距离尽量大、类内距离尽量小”。
# 与 PCA 不同：PCA 不看标签、尽量保留数据方差；LDA 会使用标签、尽量分开类别。
# LDA 最多可降到 min(原始特征数, 类别数 - 1) 维：
#   二分类最多得到 1 个判别轴，多分类才可能得到 2 个或更多判别轴。
# ============================================================

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 参数设置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_NAME = "wine_real_dataset_300x50.xlsx"
DEFAULT_LABEL_COLUMN = "品鉴师评分"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def parse_args():
    """读取命令行参数；不传参数时直接使用与原示例相同的数据集和标签列。"""
    parser = argparse.ArgumentParser(description="使用 LDA 对带标签的 Excel 数据集进行监督降维")
    parser.add_argument(
        "--data",
        type=Path,
        default=SCRIPT_DIR / DEFAULT_DATA_NAME,
        help=f"Excel 数据集路径（默认：脚本同目录下的 {DEFAULT_DATA_NAME}）",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL_COLUMN,
        help=f"标签列名（默认：{DEFAULT_LABEL_COLUMN}）",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=None,
        help="希望保留的 LDA 维数；默认自动取允许的最大维数",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="结果保存目录（默认：数据集所在目录/LDA结果）",
    )
    return parser.parse_args()


def validate_and_prepare_data(df, label_column):
    """检查标签、缺失值和特征类型，返回可用于 LDA 的 X、y。"""
    if label_column not in df.columns:
        raise ValueError(
            f"找不到标签列“{label_column}”。\n"
            f"当前列名为：{list(df.columns)}"
        )

    if df[label_column].isna().any():
        missing_count = int(df[label_column].isna().sum())
        raise ValueError(f"标签列存在 {missing_count} 个缺失值，请先补充或删除这些样本。")

    X = df.drop(columns=[label_column]).copy()
    y = df[label_column].copy()

    # LDA 需要数值特征。这里把能够转换的内容转为数值，不能转换的内容会变为 NaN。
    X = X.apply(pd.to_numeric, errors="coerce")
    all_missing_columns = X.columns[X.isna().all()].tolist()
    if all_missing_columns:
        raise ValueError(
            "以下特征列无法转换为数值，请先编码或删除："
            + "、".join(map(str, all_missing_columns))
        )

    # 用每一列的中位数填补少量缺失值，避免 LDA 因 NaN 无法训练。
    missing_count = int(X.isna().sum().sum())
    if missing_count > 0:
        print(f"检测到特征矩阵中有 {missing_count} 个缺失值，使用各列中位数填补。")
        X = X.fillna(X.median(numeric_only=True))

    if X.shape[1] == 0:
        raise ValueError("数据集中没有可用于降维的特征列。")
    if y.nunique() < 2:
        raise ValueError("LDA 至少需要两个类别，当前标签列不足两个类别。")

    return X, y


def choose_component_count(requested_components, n_features, n_classes):
    """根据 LDA 数学限制确定最终保留的维数。"""
    max_components = min(n_features, n_classes - 1)
    if requested_components is None:
        return max_components, max_components
    if requested_components < 1:
        raise ValueError("--components 必须是大于等于 1 的整数。")
    if requested_components > max_components:
        print(
            f"请求保留 {requested_components} 维，但当前 {n_classes} 个类别的 LDA "
            f"最多只能保留 {max_components} 维，已自动调整为 {max_components} 维。"
        )
        return max_components, max_components
    return requested_components, max_components


def save_visualization(
    X_reduced,
    y,
    classes,
    explained_ratio,
    loadings,
    cm,
    accuracy_lda,
    output_path,
):
    """保存 LDA 投影、解释率、混淆矩阵和主要载荷图。"""
    n_components = X_reduced.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(classes), 2)))

    # 1. 降维后的样本分布：一维时画直方图，二维及以上画前两个判别轴散点图。
    ax1 = axes[0, 0]
    if n_components == 1:
        for color, class_value in zip(colors, classes):
            class_mask = np.asarray(y == class_value)
            ax1.hist(
                X_reduced[class_mask, 0],
                bins=20,
                alpha=0.55,
                color=color,
                label=f"类别 {class_value}",
            )
        ax1.set_xlabel("LD1（第一判别轴）")
        ax1.set_ylabel("样本数")
        ax1.set_title("LDA 一维投影后的类别分布")
    else:
        for color, class_value in zip(colors, classes):
            class_mask = np.asarray(y == class_value)
            ax1.scatter(
                X_reduced[class_mask, 0],
                X_reduced[class_mask, 1],
                s=42,
                alpha=0.7,
                color=color,
                label=f"类别 {class_value}",
            )
        ax1.set_xlabel("LD1（第一判别轴）")
        ax1.set_ylabel("LD2（第二判别轴）")
        ax1.set_title("LDA 前两个判别轴上的样本分布")
    ax1.legend()
    ax1.grid(alpha=0.25)

    # 2. 每个判别轴的判别信息解释率。
    ax2 = axes[0, 1]
    component_labels = [f"LD{i + 1}" for i in range(len(explained_ratio))]
    ax2.bar(component_labels, explained_ratio * 100, color="steelblue", alpha=0.8)
    ax2.set_xlabel("线性判别轴")
    ax2.set_ylabel("解释率（%）")
    ax2.set_title("各判别轴的判别信息解释率")
    ax2.grid(axis="y", alpha=0.25)
    for index, ratio in enumerate(explained_ratio):
        ax2.text(index, ratio * 100, f"{ratio * 100:.1f}%", ha="center", va="bottom")

    # 3. LDA 分类结果混淆矩阵。
    ax3 = axes[1, 0]
    image = ax3.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_xticks(range(len(classes)), labels=[str(value) for value in classes])
    ax3.set_yticks(range(len(classes)), labels=[str(value) for value in classes])
    ax3.set_xlabel("预测类别")
    ax3.set_ylabel("真实类别")
    ax3.set_title(f"测试集混淆矩阵（准确率={accuracy_lda:.4f}）")
    threshold = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            text_color = "white" if cm[row, column] > threshold else "black"
            ax3.text(column, row, str(cm[row, column]), ha="center", va="center", color=text_color)

    # 4. 第一判别轴绝对载荷最大的 15 个原始特征。
    ax4 = axes[1, 1]
    top_count = min(15, len(loadings))
    top_loadings = loadings["LD1"].abs().nlargest(top_count).sort_values()
    ax4.barh(top_loadings.index.astype(str), top_loadings.values, color="seagreen", alpha=0.8)
    ax4.set_xlabel("|LD1 载荷|（绝对值）")
    ax4.set_title(f"LD1 贡献最大的 {top_count} 个原始特征")
    ax4.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else data_path.parent / "LDA结果"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"找不到数据集：{data_path}\n"
            "请把脚本放到数据集同一目录，或使用 --data 指定 Excel 文件路径。"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 第一步：读取并检查 Excel 数据
    # ============================================================
    df = pd.read_excel(data_path)
    X, y = validate_and_prepare_data(df, args.label)
    feature_names = X.columns.tolist()
    classes = np.sort(y.unique())
    n_classes = len(classes)
    n_components, max_components = choose_component_count(
        args.components, X.shape[1], n_classes
    )

    print(f"数据集路径: {data_path}")
    print(f"原始数据形状: {df.shape}")
    print(f"特征矩阵 X 形状: {X.shape}")
    print(f"标签向量 y 形状: {y.shape}")
    print(f"类别及样本数:\n{y.value_counts().sort_index()}")
    print(f"LDA 理论最大维数: min({X.shape[1]}, {n_classes}-1) = {max_components}")
    print(f"本次实际输出维数: {n_components}")

    # ============================================================
    # 第二步：先划分训练集和测试集，避免测试集信息泄漏到模型训练
    # ============================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"训练集大小: {X_train.shape[0]} 个样本")
    print(f"测试集大小: {X_test.shape[0]} 个样本")

    # ============================================================
    # 第三步：标准化；只在训练集上 fit，再转换测试集
    # ============================================================
    scaler_eval = StandardScaler()
    X_train_scaled = scaler_eval.fit_transform(X_train)
    X_test_scaled = scaler_eval.transform(X_test)

    # ============================================================
    # 第四步：在训练集拟合 LDA，并将训练集、测试集降到低维空间
    # ============================================================
    lda_eval = LinearDiscriminantAnalysis(
        n_components=n_components,
        solver="svd",
    )
    X_train_lda = lda_eval.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda_eval.transform(X_test_scaled)
    y_pred_lda = lda_eval.predict(X_test_scaled)
    accuracy_lda = accuracy_score(y_test, y_pred_lda)

    print("\n===== LDA 测试集分类结果 =====")
    print(f"LDA 自身分类准确率: {accuracy_lda:.4f}")
    print(classification_report(y_test, y_pred_lda, zero_division=0))

    # ============================================================
    # 第五步：公平比较原始特征与 LDA 降维特征的分类效果
    # ============================================================
    baseline_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    baseline_model.fit(X_train_scaled, y_train)
    y_pred_full = baseline_model.predict(X_test_scaled)
    accuracy_full = accuracy_score(y_test, y_pred_full)

    reduced_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    reduced_model.fit(X_train_lda, y_train)
    y_pred_reduced = reduced_model.predict(X_test_lda)
    accuracy_reduced = accuracy_score(y_test, y_pred_reduced)

    print("===== 降维前后分类效果对比 =====")
    print(f"原始 {X.shape[1]} 维 + 逻辑回归准确率: {accuracy_full:.4f}")
    print(f"LDA {n_components} 维 + 逻辑回归准确率: {accuracy_reduced:.4f}")
    print(f"LDA 自身分类准确率: {accuracy_lda:.4f}")

    # ============================================================
    # 第六步：在完整数据上重新拟合最终模型，生成完整降维数据集
    # 说明：训练/测试评估使用上面的训练集模型；完整模型只用于最终导出。
    # ============================================================
    scaler_final = StandardScaler()
    X_scaled_all = scaler_final.fit_transform(X)
    lda_final = LinearDiscriminantAnalysis(
        n_components=n_components,
        solver="svd",
    )
    X_lda_all = lda_final.fit_transform(X_scaled_all, y)
    component_names = [f"LD{i + 1}" for i in range(n_components)]

    reduced_df = pd.DataFrame(X_lda_all, columns=component_names, index=df.index)
    reduced_df[args.label] = y.values
    reduced_path = output_dir / "lda_reduced_data.xlsx"
    reduced_df.to_excel(reduced_path, index=False)

    # scalings_ 表示每个原始特征在各判别轴中的权重。
    loadings = pd.DataFrame(
        lda_final.scalings_[:, :n_components],
        index=feature_names,
        columns=component_names,
    )
    loadings.index.name = "特征名称"
    loadings_path = output_dir / "lda_loadings.xlsx"
    loadings.to_excel(loadings_path)

    class_means = pd.DataFrame(
        lda_final.means_,
        index=[f"类别_{value}" for value in lda_final.classes_],
        columns=feature_names,
    )
    class_means.index.name = "类别（标准化空间）"
    class_means_path = output_dir / "lda_class_means.xlsx"
    class_means.to_excel(class_means_path)

    metrics_df = pd.DataFrame(
        {
            "方案": [
                f"原始{X.shape[1]}维 + 逻辑回归",
                f"LDA降至{n_components}维 + 逻辑回归",
                "LDA自身分类器",
            ],
            "测试集准确率": [accuracy_full, accuracy_reduced, accuracy_lda],
        }
    )
    metrics_path = output_dir / "lda_model_comparison.xlsx"
    metrics_df.to_excel(metrics_path, index=False)

    # ============================================================
    # 第七步：可视化并保存结果图
    # ============================================================
    explained_ratio = lda_final.explained_variance_ratio_[:n_components]
    cm = confusion_matrix(y_test, y_pred_lda, labels=classes)
    figure_path = output_dir / "lda_result.png"
    save_visualization(
        X_lda_all,
        y,
        classes,
        explained_ratio,
        loadings,
        cm,
        accuracy_lda,
        figure_path,
    )

    # ============================================================
    # 最终总结
    # ============================================================
    print("\n" + "=" * 60)
    print("LDA 降维总结")
    print("=" * 60)
    print(f"输入维度: {X.shape[1]} 维")
    print(f"输出维度: {n_components} 维")
    print(f"降维率: {(1 - n_components / X.shape[1]) * 100:.1f}%")
    print(f"判别信息累计解释率: {explained_ratio.sum() * 100:.2f}%")
    print(f"LDA 测试集分类准确率: {accuracy_lda:.4f}")
    print(f"降维后的完整数据: {reduced_path}")
    print(f"判别轴载荷矩阵: {loadings_path}")
    print(f"类别均值: {class_means_path}")
    print(f"模型对比结果: {metrics_path}")
    print(f"可视化结果: {figure_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
