import pandas as pd
import numpy as np
import time
import warnings
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score


warnings.filterwarnings('ignore')

# 1. 读取数据
train_path = r'E:\daily study\机器学习\day08+day09\temp_data\训练数据集[随机森林].xlsx'
test_path = r'E:\daily study\机器学习\day08+day09\temp_data\测试数据集[随机森林].xlsx'

train_data = pd.read_excel(train_path)
test_data = pd.read_excel(test_path)

X_train = train_data.drop(['矿物类型'], axis=1)
y_train = train_data['矿物类型']
X_test = test_data.drop(['矿物类型'], axis=1)
y_test = test_data['矿物类型']
# 检查是否有重叠
overlap = X_train.merge(X_test, how='inner')
print(f"训练集和测试集完全相同的行数: {len(overlap)}")
# 2. 定义交叉验证策略（3折，分层抽样，保证每折中各类别比例均衡）
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# 3. 定义各个模型的参数网格（基于你之前图片中使用的参数）
param_grids = {
    'Logistic Regression': {
        'model': LogisticRegression(solver='lbfgs', max_iter=1000),
        'param_grid': {
            'C': [0.001, 0.01, 0.1, 1]  # 搜 C 值（正则化强度）
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(bootstrap=False, max_features='log2', random_state=487),
        'param_grid': {
            'n_estimators': [50, 100],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=100),
        # SVM非常耗时，网格精简为4种组合
        'param_grid': {
            'C': [0.1, 1],
            'gamma': [0.1, 1],
            'kernel': ['rbf', 'poly'],
            'degree': [3, 4]  # 仅对 poly 核生效
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), random_state=0),
        # 注意：新版sklearn必须用estimator，且algorithm参数已移除
        'param_grid': {
            'n_estimators': [50, 200],
            'learning_rate': [0.5, 1.0]
        }
    },
    'Gaussian Naive Bayes': {
        'model': GaussianNB(),
        'param_grid': {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(objective='multi:softmax', num_class=5, seed=0),
        'param_grid': {
            'learning_rate': [0.01, 0.05],
            'n_estimators': [100, 200],
            'max_depth': [5, 7],
            'subsample': [0.6, 0.8]
        }
    }
}

print_acc = []
# 4. 循环进行网格搜索训练
print("========== 开始网格搜索交叉验证 ==========\n")
for name, config in param_grids.items():
    print(f"正在对 [{name}] 进行网格搜索...")
    start_time = time.time()

    # 实例化 GridSearchCV
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['param_grid'],
        cv=cv,
        scoring='accuracy',  # 多分类一般用准确率，也可以用 'f1_macro'
        n_jobs=-1,  # 开启全部CPU核心并行计算，大大提速
        verbose=1  # 打印进度条
    )

    # 训练
    grid_search.fit(X_train, y_train)

    # 获取最优模型
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    # 用最优模型在测试集上预测
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print_acc.append(test_acc)
    # 输出结果
    print(f"\n✅ [{name}] 训练完成，耗时 {time.time() - start_time:.2f} 秒")
    print(f"最佳参数组合: {best_params}")
    print(f"交叉验证平均准确率: {best_cv_score:.4f}")
    print(f"最终测试集准确率: {test_acc:.4f}")
    print("分类报告:")
    print(classification_report(y_test, y_pred,digits=6))
    print("=" * 50 + "\n")
print(f'在中位数填充下？得到最好的结果，acc={max(print_acc):.4f}')
print("所有模型网格搜索完成！")