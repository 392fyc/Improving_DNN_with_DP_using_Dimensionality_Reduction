import dask.dataframe as dd
from dask_ml.decomposition import PCA as daskPCA
from dask_ml.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import numpy as np


def load_data(file_path):
    """读取本地CSV文件"""
    return dd.read_csv(file_path)


def preprocess_data(df):
    """转换非数值列为数值列并处理缺失值"""
    # 将所有对象类型的列转换为类别类型
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype('category')

    # 将类别类型编码为数值
    df = df.categorize()

    # 将类别列转换为数值
    for col in df.select_dtypes(include=['category']):
        df[col] = df[col].cat.codes

    # 用列的平均值填充 NaN
    df = df.fillna(df.mean())

    return df


def cumulative_explained_variance(X, target_variance=0.95):
    """累计解释方差法选择最优维度"""
    X = X.to_dask_array(lengths=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = daskPCA(n_components=min(X.shape[1], 100))  # 限制最大组件数
    pca.fit(X)

    # 检查 explained_variance_ratio_ 的类型
    if hasattr(pca.explained_variance_ratio_, 'compute'):
        explained_variance = pca.explained_variance_ratio_.compute()
    else:
        explained_variance = pca.explained_variance_ratio_

    cumulative_variance = np.cumsum(explained_variance)
    n_components = np.argmax(cumulative_variance >= target_variance) + 1
    return n_components, cumulative_variance


def bic_selection(X, max_components=43):

    bic_values = []

    # 检查 X 是否为 Dask DataFrame 或 Dask Array
    if hasattr(X, 'compute'):
        X = X.compute()  # Convert Dask DataFrame or Array to NumPy array

    for n in range(1, min(max_components, X.shape[1]) + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=0)
        gmm.fit(X)
        bic_values.append(gmm.bic(X))

    optimal_bic = np.argmin(bic_values) + 1

    return optimal_bic, bic_values


file_path = 'data/diabetic_data.csv'
data = load_data(file_path)

# 预处理数据，转换非数值列为数值列
data_numeric = preprocess_data(data)

# 累计解释方差法选择最优维度
# optimal_k_cumulative, cumulative_variance = cumulative_explained_variance(data_numeric)

# BIC选择最优维度
optimal_k_bic, bic_values = bic_selection(data_numeric)

# 打印结果
# print(f"累计解释方差法选择的最优维度: {optimal_k_cumulative}")
print(f"BIC选择的最优维度: {optimal_k_bic}")