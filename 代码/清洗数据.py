from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import fill_data  # 假设该模块包含 cca_train_fill
from sklearn.model_selection import train_test_split

data = pd.read_excel("矿物数据.xls")
data = data[data['矿物类型'] != 'E']

# 2. 分离特征和标签
x_whole = data.drop(['矿物类型', '序号'], axis=1)
y_whole = data['矿物类型']

# 3. 标签编码
label_dict = {'A':0, 'B':1, 'C':2, 'D':3}
y_encoded = y_whole.map(label_dict)

# 4. 强制数值类型（非数值转为NaN）
for col in x_whole.columns:
    x_whole[col] = pd.to_numeric(x_whole[col], errors='coerce')

# 注意：这里要用 y_encoded（数字0~3），不能用 y_whole（字符串A~D），
# 因为 fill_data 里的众数函数是按 0/1/2/3 来过滤数据的
x_train_w,x_test_w,y_train_w,y_test_w=\
    train_test_split(x_whole,y_encoded,test_size=0.3,random_state=50000)


#1,考虑包含完整行的数据
# x_train_fill, y_train_fill = fill_data.cca_train_fill(x_train_w, y_train_w)
# x_test_fill, y_test_fill = fill_data.cca_test_fill(x_train_fill, y_train_fill, x_test_w, y_test_w)

# #2,均值
# x_train_fill, y_train_fill = fill_data.mean_train_fill(x_train_w, y_train_w)
# x_test_fill, y_test_fill = fill_data.mean_test_fill(
#     x_train_w, y_train_w, x_test_w, y_test_w
# )


# # 3，中位数
# x_train_fill, y_train_fill = fill_data.median_train_fill(x_train_w, y_train_w)
# x_test_fill, y_test_fill = fill_data.median_test_fill(
#     x_train_w, y_train_w, x_test_w, y_test_w
# )



# #4众数
# x_train_fill, y_train_fill = fill_data.mode_train_fill(x_train_w, y_train_w)
# x_test_fill, y_test_fill = fill_data.mode_test_fill(
#     x_train_w, y_train_w, x_test_w, y_test_w
# )


# #5线性回归
# x_train_fill, y_train_fill = fill_data.lt_train_fill(x_train_w, y_train_w)
# x_test_fill, y_test_fill = fill_data.lt_test_fill(x_train_w, y_train_w, x_test_w, y_test_w)


#6随机森林
x_train_fill, y_train_fill = fill_data.rt_train_fill(x_train_w, y_train_w)
x_test_fill, y_test_fill = fill_data.rt_test_fill(x_train_w, y_train_w, x_test_w, y_test_w)


# # 填充完成后再标准化：只 fit 训练集，再用它 transform 测试集（避免数据泄露）
scaler = StandardScaler()
x_train_z = scaler.fit_transform(x_train_fill)
x_test_z = scaler.transform(x_test_fill)
x_train_fill = pd.DataFrame(x_train_z, columns=x_train_fill.columns)
x_test_fill = pd.DataFrame(x_test_z, columns=x_test_fill.columns)

#数据保存为excel文件
data_train = pd.concat([y_train_fill,x_train_fill],axis=1).sample(frac=1,random_state=5)
data_test = pd.concat([y_test_fill,x_test_fill],axis=1)

data_train.to_excel(r'.//temp_data//训练数据集[随机森林].xlsx',index=False)
data_test.to_excel(r'.//temp_data//测试数据集[随机森林].xlsx',index=False)


