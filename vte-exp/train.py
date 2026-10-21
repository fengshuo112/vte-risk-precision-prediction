import mlflow
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
mlflow.tensorflow.autolog()

data_dir='./data'
train = pd.read_excel(f'{data_dir}/train.xlsx')

# 空值特征统计
train['null_count'] = train.isnull().sum(axis=1)


# clear data
def clear_char(x):
    x = str(x).replace('<', '').replace('>', '').replace(',', '').replace('kg', '').replace('/', '')
    try:
        x = x.split()[0]
        x = float(x.split('(')[0])
    except:
        # pass
        x = -1
    return x


process_cols = [col for col in train.columns if
                col not in ['唯一标识', '性别', '民族', '3P试验', '住院号', 'flag', '总评分', '诊断']]
for col in process_cols:
    train[col] = train[col].apply(lambda x: clear_char(x))

# 重要特征
select_columns = ['体重',
                  '心率',
                  '收缩压',
                  '住院次数',
                  '舒张压',
                  '身高',
                  'D-二聚体',
                  '呼吸',
                  '丙氨酸氨基转移酶',
                  '凝血酶时间',
                  '低密度脂蛋白胆固醇',
                  '乳酸脱氢酶',
                  '尿素',
                  '凝血酶原时间',
                  'γ谷氨酰转移酶',
                  '天门冬氨酸氨基转移酶',
                  '凝血酶原时间活动度',
                  '国际标准化比率',
                  '凝血酶原时间比率',
                  'γ-谷氨酰基转移酶']
train["imp_sum"] = train[select_columns].sum(axis=1)

categorical_columns = train.select_dtypes(include=['object']).columns.tolist()
numerical_columns = train.select_dtypes(exclude=['object']).columns.tolist()
numerical_columns = [col for col in numerical_columns if col not in ['唯一标识', 'flag', '总评分']]

numerical_column_names = [f'num_{idx}' for idx in range(len(numerical_columns))]
for idx, num_col in enumerate(numerical_column_names):
    # sd=StandardScaler()
    train[f'num_{idx}'] = train[numerical_columns[idx]]
    # train[f'num_{idx}']=sd.fit_transform(train[f'num_{idx}'].values.reshape(-1,1))
    del train[numerical_columns[idx]]
numerical_columns = numerical_column_names

indicator_column_names = [f'cate_{idx}' for idx in range(len(categorical_columns))]
for idx, num_col in enumerate(indicator_column_names):
    train[f'cate_{idx}'] = train[categorical_columns[idx]]
    del train[categorical_columns[idx]]
categorical_columns = indicator_column_names

cat = ['住院号', '性别', '民族', '诊断', '3P试验']
no_feas = ['唯一标识', 'flag', '总评分']
features = [col for col in train.columns if col not in no_feas]

tmp_data = pd.concat([train[categorical_columns]], axis=0)
for col in categorical_columns:
    lbl = LabelEncoder()
    lbl.fit(tmp_data[col].astype(str).fillna('-1'))
    train[col] = lbl.transform(train[col].astype(str).fillna('-1'))

dataframe = train[features+['flag']].copy()
dataframe = dataframe.fillna(0)
print(dataframe)
train, test = train_test_split(dataframe, test_size=0.1)
train, val = train_test_split(train, test_size=0.1)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('flag')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5  # A small batch sized is used for demonstration purposes
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['num_2'])
    print('A batch of targets:', label_batch)

feature_columns = []
# numeric cols
for header in numerical_columns:
    feature_columns.append(feature_column.numeric_column(header))
# indicator_columns
for idx, col_name in enumerate(categorical_columns):
    categorical_column = feature_column.categorical_column_with_vocabulary_list(
        col_name, dataframe[col_name].unique())
    indicator_column = feature_column.indicator_column(categorical_column)
    feature_columns.append(indicator_column)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
batch_size = 8
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dropout(.2),
    layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=10)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
mlflow.tensorflow.log_model(model, artifact_path="model")
