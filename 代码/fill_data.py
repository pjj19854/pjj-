from sklearn import linear_model

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd




#-------------------1只考虑包含完整行的数据--------------#
def cca_train_fill(train_data,train_label):
        data = pd.concat([train_data,train_label],axis=1)
        data = data.reset_index(drop=True)
        df_filled = data.dropna()
        # dropna 后行号是散的，必须再重置一次，否则后面和 x 按索引拼接会对不上
        df_filled = df_filled.reset_index(drop=True)
        return df_filled.drop('矿物类型',axis=1),df_filled.矿物类型


def cca_test_fill(train_data, train_label,test_data,test_label):
    # 完整行法：测试集同样只保留没有缺失值的行
    # （train_data/train_label 参数用不上，保留是为了和其他 test_fill 的接口一致）
    data = pd.concat([test_data, test_label], axis=1)
    data = data.reset_index(drop=True)
    df_filled = data.dropna()
    df_filled = df_filled.reset_index(drop=True)
    return df_filled.drop('矿物类型', axis=1), df_filled.矿物类型

#-------------------2均值--------------#
def mean_method(data):
    fill_value = data.mean()
    return data.fillna(fill_value)

def mean_train_fill(train_data, train_label):
    data = pd.concat([train_data, train_label], axis=1)
    data = data.reset_index(drop=True)
    A = data[data['矿物类型'] == 0]
    B = data[data['矿物类型'] == 1]
    C = data[data['矿物类型'] == 2]
    D = data[data['矿物类型'] == 3]

    A = mean_method(A)
    B = mean_method(B)
    C = mean_method(C)
    D = mean_method(D)

    df_filled = pd.concat([A, B, C, D])
    df_filled = df_filled.reset_index(drop=True)
    return df_filled.drop('矿物类型', axis=1), df_filled.矿物类型

def mean_test_method(train_data, test_data):
    fill_value = train_data.mean()
    return test_data.fillna(fill_value)

def mean_test_fill(train_data, train_label, test_data, test_label):
    train_data_all = pd.concat([train_data, train_label], axis=1)
    train_data_all = train_data_all.reset_index(drop=True)
    test_data_all = pd.concat([test_data, test_label], axis=1)
    test_data_all = test_data_all.reset_index(drop=True)

    A_train = train_data_all[train_data_all['矿物类型'] == 0]
    B_train = train_data_all[train_data_all['矿物类型'] == 1]
    C_train = train_data_all[train_data_all['矿物类型'] == 2]
    D_train = train_data_all[train_data_all['矿物类型'] == 3]

    A_test = test_data_all[test_data_all['矿物类型'] == 0]
    B_test = test_data_all[test_data_all['矿物类型'] == 1]
    C_test = test_data_all[test_data_all['矿物类型'] == 2]
    D_test = test_data_all[test_data_all['矿物类型'] == 3]

    A = mean_test_method(A_train, A_test)
    B = mean_test_method(B_train, B_test)
    C = mean_test_method(C_train, C_test)
    D = mean_test_method(D_train, D_test)

    df_filled = pd.concat([A, B, C, D])
    df_filled = df_filled.reset_index(drop=True)
    return df_filled.drop('矿物类型', axis=1), df_filled.矿物类型

#-------------------3中位数--------------#
def median_method(data):
    fill_value = data.median()
    return data.fillna(fill_value)

def median_train_fill(train_data, train_label):
    data = pd.concat([train_data, train_label], axis=1)
    data = data.reset_index(drop=True)
    A = data[data['矿物类型'] == 0]
    B = data[data['矿物类型'] == 1]
    C = data[data['矿物类型'] == 2]
    D = data[data['矿物类型'] == 3]

    A = median_method(A)
    B = median_method(B)
    C = median_method(C)
    D = median_method(D)

    df_filled = pd.concat([A, B, C, D])
    df_filled = df_filled.reset_index(drop=True)
    return df_filled.drop('矿物类型', axis=1), df_filled.矿物类型

def median_test_method(train_data, test_data):
    fill_value = train_data.median()
    return test_data.fillna(fill_value)

def median_test_fill(train_data, train_label, test_data, test_label):
    train_data_all = pd.concat([train_data, train_label], axis=1)
    train_data_all = train_data_all.reset_index(drop=True)
    test_data_all = pd.concat([test_data, test_label], axis=1)
    test_data_all = test_data_all.reset_index(drop=True)

    A_train = train_data_all[train_data_all['矿物类型'] == 0]
    B_train = train_data_all[train_data_all['矿物类型'] == 1]
    C_train = train_data_all[train_data_all['矿物类型'] == 2]
    D_train = train_data_all[train_data_all['矿物类型'] == 3]

    A_test = test_data_all[test_data_all['矿物类型'] == 0]
    B_test = test_data_all[test_data_all['矿物类型'] == 1]
    C_test = test_data_all[test_data_all['矿物类型'] == 2]
    D_test = test_data_all[test_data_all['矿物类型'] == 3]

    A = median_test_method(A_train, A_test)
    B = median_test_method(B_train, B_test)
    C = median_test_method(C_train, C_test)
    D = median_test_method(D_train, D_test)

    df_filled = pd.concat([A, B, C, D])
    df_filled = df_filled.reset_index(drop=True)
    return df_filled.drop('矿物类型', axis=1), df_filled.矿物类型

#-------------------4众数--------------#
def mode_method(data):
    fill_value = data.apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
    a = data.fillna(fill_value)
    return data.fillna(fill_value)

def mode_train_fill(train_data, train_label):
    data = pd.concat([train_data, train_label], axis=1)
    data = data.reset_index(drop=True)
    A = data[data['矿物类型'] == 0]
    B = data[data['矿物类型'] == 1]
    C = data[data['矿物类型'] == 2]
    D = data[data['矿物类型'] == 3]

    A = mode_method(A)
    B = mode_method(B)
    C = mode_method(C)
    D = mode_method(D)

    df_filled = pd.concat([A,B,C,D])
    df_filled = df_filled.reset_index(drop=True)
    return df_filled.drop('矿物类型',axis=1),df_filled.矿物类型

def mode_test_method(train_data, test_data):
    fill_value = train_data.apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
    return test_data.fillna(fill_value)

def mode_test_fill(train_data,train_label, test_data,test_label):
    train_data_all = pd.concat([train_data, train_label], axis=1)
    train_data_all = train_data_all.reset_index(drop=True)
    test_data_all = pd.concat([test_data, test_label], axis=1)
    test_data_all = test_data_all.reset_index(drop=True)

    A_train = train_data_all[train_data_all['矿物类型'] == 0]
    B_train = train_data_all[train_data_all['矿物类型'] == 1]
    C_train = train_data_all[train_data_all['矿物类型'] == 2]
    D_train = train_data_all[train_data_all['矿物类型'] == 3]


    A_test = test_data_all[test_data_all['矿物类型'] == 0]
    B_test = test_data_all[test_data_all['矿物类型'] == 1]
    C_test = test_data_all[test_data_all['矿物类型'] == 2]
    D_test = test_data_all[test_data_all['矿物类型'] == 3]

    A = mode_test_method(A_train,A_test)
    B = mode_test_method(B_train,B_test)
    C = mode_test_method(C_train,C_test)
    D= mode_test_method(D_train,D_test)


    df_filled = pd.concat([A, B, C, D])
    df_filled = df_filled.reset_index(drop=True)
    return df_filled.drop('矿物类型',axis = 1), df_filled.矿物类型


# -------------------5线性回归 / 6随机森林（回归模型填充）--------------#
# 两种方法共用下面这套逻辑，区别只是每列用线性回归还是随机森林来填：
#   1) 按缺失值从少到多依次处理每一列：缺失少的列先填，填好后可作为其它列的“已知信息”
#   2) 填某一列时，用已填好的其它列当 x、该列当 y，在该列不缺失的行上训练回归模型，
#      再去预测该列的缺失行
#   3) 某一列缺失超过一半时，模型学不到可靠信息，直接用训练集该列的中位数填
#   4) 测试集只允许用训练集的信息：先把训练集填完，再拿填好的训练集训练模型去填测试集

def _regress_fill(train_x, test_x, make_model, null_counts, verbose):
    """按列回归填充缺失值，train_x / test_x 就地修改；test_x 为 None 表示只填训练集。
    null_counts 必须用“原始训练集”的缺失数：处理顺序和中位数兜底阈值都由它决定，
    填测试集时也复用同一份，保证训练集、测试集走同一套规则。
    make_model：无参函数，每次填一列时新建一个回归模型（fit/predict 接口）。
    """
    order = null_counts.sort_values(ascending=True).index  # 缺失少的列先填
    n_rows = len(train_x)
    filled_features = []  # 已填完的列，作为后面列的预测依据

    for i in order:
        filled_features.append(i)
        train_null = null_counts[i]
        test_null = 0 if test_x is None else test_x[i].isnull().sum()

        # 训练集和测试集这一列都完整，无需处理
        if train_null == 0 and test_null == 0:
            continue

        # 缺失超过一半：模型学不到东西，直接用训练集该列的中位数填
        if train_null > n_rows * 0.5:
            median_v = train_x[i].median()
            train_x[i] = train_x[i].fillna(median_v)
            if test_x is not None:
                test_x[i] = test_x[i].fillna(median_v)
            continue

        # 还没有任何已填好的列可当 x（比如第一列就缺失），也只能用中位数
        x_cols = [c for c in filled_features if c != i]
        if len(x_cols) == 0:
            median_v = train_x[i].median()
            train_x[i] = train_x[i].fillna(median_v)
            if test_x is not None:
                test_x[i] = test_x[i].fillna(median_v)
            continue

        # 用训练集该列“不缺失”的行训练回归模型
        mask = train_x[i].notnull()
        model = make_model()
        model.fit(train_x.loc[mask, x_cols], train_x.loc[mask, i])

        # 预测值裁剪到训练集该列的观测范围，防止线性回归外推出负浓度之类的怪值
        y_min, y_max = train_x.loc[mask, i].min(), train_x.loc[mask, i].max()

        # 填训练集该列的缺失行
        train_missing = train_x[train_x[i].isnull()].index
        if len(train_missing) > 0:
            pred = model.predict(train_x.loc[train_missing, x_cols])
            train_x.loc[train_missing, i] = np.clip(pred, y_min, y_max)
            if verbose:
                print(f'完成训练集填充 {i}')

        # 填测试集该列的缺失行
        if test_x is not None and test_null > 0:
            test_missing = test_x[test_x[i].isnull()].index
            pred = model.predict(test_x.loc[test_missing, x_cols])
            test_x.loc[test_missing, i] = np.clip(pred, y_min, y_max)
            if verbose:
                print(f'完成测试集填充 {i}')


# --------5线性回归填充--------#
def lt_train_fill(train_data, train_label):
    data = pd.concat([train_data, train_label], axis=1).reset_index(drop=True)
    train_x = data.drop('矿物类型', axis=1)
    null_counts = train_x.isnull().sum()
    _regress_fill(train_x, None, lambda: LinearRegression(), null_counts, verbose=True)
    train_x = train_x.fillna(train_x.median())  # 兜底，理论上不会再有缺失
    return train_x, data.矿物类型


def lt_test_fill(train_data, train_label, test_data, test_label):
    # 这里 train_data 直接传原始(未填充)的训练集即可，函数内部会先把训练集填好
    train_all = pd.concat([train_data, train_label], axis=1).reset_index(drop=True)
    test_all = pd.concat([test_data, test_label], axis=1).reset_index(drop=True)
    train_x = train_all.drop('矿物类型', axis=1)
    test_x = test_all.drop('矿物类型', axis=1)
    null_counts = train_x.isnull().sum()  # 原始训练集的缺失数，两轮填充共用

    _regress_fill(train_x, None, lambda: LinearRegression(), null_counts, verbose=False)
    _regress_fill(train_x, test_x, lambda: LinearRegression(), null_counts, verbose=True)

    test_x = test_x.fillna(train_x.median())  # 兜底，用训练集的中位数填剩余空缺
    return test_x, test_all.矿物类型


# --------6随机森林填充--------#
def rt_train_fill(train_data, train_label):
    data = pd.concat([train_data, train_label], axis=1).reset_index(drop=True)
    train_x = data.drop('矿物类型', axis=1)
    null_counts = train_x.isnull().sum()
    _regress_fill(train_x, None, lambda: RandomForestRegressor(n_estimators=100, random_state=42),
                  null_counts, verbose=True)
    train_x = train_x.fillna(train_x.median())
    return train_x, data.矿物类型


def rt_test_fill(train_data, train_label, test_data, test_label):
    # 同样：train_data 传原始(未填充)的训练集即可，函数内部会先把训练集填好
    train_all = pd.concat([train_data, train_label], axis=1).reset_index(drop=True)
    test_all = pd.concat([test_data, test_label], axis=1).reset_index(drop=True)
    train_x = train_all.drop('矿物类型', axis=1)
    test_x = test_all.drop('矿物类型', axis=1)
    null_counts = train_x.isnull().sum()

    _regress_fill(train_x, None, lambda: RandomForestRegressor(n_estimators=100, random_state=42),
                  null_counts, verbose=False)
    _regress_fill(train_x, test_x, lambda: RandomForestRegressor(n_estimators=100, random_state=42),
                  null_counts, verbose=True)

    test_x = test_x.fillna(train_x.median())
    return test_x, test_all.矿物类型








