# ✅ [라이브러리 임포트]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate, Flatten, RepeatVector
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
import tensorflow as tf

# ✅ [0] 데이터 로드 및 전처리 --------------------------------------------------
df = pd.read_parquet('prepro_data.parquet')  # parquet 파일에서 데이터 로드

# 범주형 문자열 변수를 숫자로 변환 (Label Encoding)
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        df[col] = df[col].astype(str)
        df[col] = LabelEncoder().fit_transform(df[col])

# 시간 관련 변수 주기 변환 (주기성을 반영하여 sin, cos 값으로 변환)
df['시_sin'] = np.sin(2 * np.pi * df['시'] / 24)
df['시_cos'] = np.cos(2 * np.pi * df['시'] / 24)
df['요일_sin'] = np.sin(2 * np.pi * df['요일'] / 7)
df['요일_cos'] = np.cos(2 * np.pi * df['요일'] / 7)

# 월, 일 컬럼도 범주형 처리
df['월'] = LabelEncoder().fit_transform(df['월'].astype(str))
df['일'] = LabelEncoder().fit_transform(df['일'].astype(str))

# 데이터 타입 최적화 (메모리 사용 절감)
float_cols = df.select_dtypes(include=['float64']).columns
int_cols = df.select_dtypes(include=['int64']).columns
df[float_cols] = df[float_cols].astype('float32')
df[int_cols] = df[int_cols].astype('int32')

# 사용하지 않는 열 제거
df = df.drop(['시', '요일'], axis=1)

# ✅ [1] 피처 구성 --------------------------------------------------------
# Embedding을 적용할 범주형 변수 설정
embed_cols = ['역명', 'AWS지점코드']
label_encoders = {}
for col in embed_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 각 입력 데이터 정의
numeric_cols = df.drop(columns=['혼잡도', '시간', '역명', 'AWS지점코드', '호선']).columns
X_numeric = df[numeric_cols].values.astype(np.float32)        # 수치형 데이터
X_station = df['역명'].values.astype(np.int32)                # 역명 (카테고리형)
X_aws     = df['AWS지점코드'].values.astype(np.int32)         # 기상 지점 코드
X_line    = df['호선'].values.astype(np.int32)                # 호선 (카테고리형)

y = df['혼잡도'].values.astype(np.float32)                    # 타겟 변수

# 스케일링 (입력: MinMax, 타겟: Standard)
mm = MinMaxScaler()
X_numeric_scaled = mm.fit_transform(X_numeric).astype(np.float32)
ss = StandardScaler()
y_scaled = ss.fit_transform(y.reshape(-1, 1)).flatten().astype(np.float32)

# ✅ [2] 시계열 데이터 생성기 정의 ----------------------------------------
# tf.data.Dataset을 사용하여 시계열 학습용 배치 생성 함수 정의
def create_tf_dataset(X_numeric, X_station, X_aws, X_line, y, sequence_length, batch_size):
    def generator():
        for i in range(sequence_length, len(X_numeric)):
            yield (
                {
                    "station": np.int32(X_station[i]),
                    "aws": np.int32(X_aws[i]),
                    "line": np.int32(X_line[i]),
                    "numeric": X_numeric[i-sequence_length:i]  # 이전 21개 시간 구간 입력
                },
                np.float32(y[i])  # 현재 시간의 타겟값 예측
            )

    output_signature = (
        {
            "station": tf.TensorSpec(shape=(), dtype=tf.int32),
            "aws": tf.TensorSpec(shape=(), dtype=tf.int32),
            "line": tf.TensorSpec(shape=(), dtype=tf.int32),
            "numeric": tf.TensorSpec(shape=(sequence_length, X_numeric.shape[1]), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ✅ [3] 학습/검증 데이터 분할 및 Dataset 생성 ------------------------------
sequence_length = 21
batch_size = 128
split_index = int(len(X_numeric_scaled) * 0.8)  # 80% 학습, 20% 검증

# 학습/검증 데이터셋 생성
train_ds = create_tf_dataset(X_numeric_scaled[:split_index], X_station[:split_index], X_aws[:split_index], X_line[:split_index], y_scaled[:split_index], sequence_length, batch_size)
val_ds   = create_tf_dataset(X_numeric_scaled[split_index:], X_station[split_index:], X_aws[split_index:], X_line[split_index:], y_scaled[split_index:], sequence_length, batch_size)

# ✅ [4] 모델 정의 및 학습 ----------------------------------------------------
# Embedding에 필요한 차원 수 계산
n_station = df['역명'].nunique()
n_aws     = df['AWS지점코드'].nunique()
n_line    = df['호선'].nunique()
num_numeric_features = X_numeric.shape[1]

# 입력 정의
inp_station = Input(shape=(), dtype='int32', name="station")
inp_aws     = Input(shape=(), dtype='int32', name="aws")
inp_line    = Input(shape=(), dtype='int32', name="line")
inp_numeric = Input(shape=(sequence_length, num_numeric_features), dtype='float32', name="numeric")

# 임베딩 처리 (범주형 변수를 dense 벡터로 변환)
emb_station = Embedding(input_dim=n_station + 1, output_dim=8)(inp_station)
emb_aws     = Embedding(input_dim=n_aws + 1, output_dim=4)(inp_aws)
emb_line    = Embedding(input_dim=n_line + 1, output_dim=2)(inp_line)

# Flatten 후 병합
flat_station = Flatten()(emb_station)
flat_aws     = Flatten()(emb_aws)
flat_line    = Flatten()(emb_line)

emb_concat = Concatenate()([flat_station, flat_aws, flat_line])
emb_dense = Dense(16, activation='relu')(emb_concat)
emb_repeated = RepeatVector(sequence_length)(emb_dense)  # 시계열 길이에 맞게 반복

# 수치형 입력과 반복된 임베딩을 concat
merged_input = Concatenate(axis=-1)([inp_numeric, emb_repeated])

# LSTM 구조 정의
x = LSTM(64, return_sequences=True)(merged_input)
x = Dropout(0.2)(x)
x = LSTM(32)(x)
x = Dropout(0.2)(x)
out = Dense(1)(x)  # 회귀 출력

# 모델 정의 및 컴파일
model = Model(inputs=[inp_station, inp_aws, inp_line, inp_numeric], outputs=out)
model.compile(loss='mse', optimizer=Adam(0.001))
model.summary()

# 콜백 정의 (조기 종료, 모델 저장, 학습률 감소)
callbacks = [
    ModelCheckpoint(filepath='bm.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)
]

# 모델 학습
history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=callbacks)

# ✅ [5] 학습 시각화 및 평가 ----------------------------------------------------
# 학습 손실 시각화
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# 최적 모델 가중치 불러오기 및 예측
model.load_weights('bm.weights.h5')
pred_scaled = model.predict(val_ds)

# 예측값, 실제값 역변환
pred = ss.inverse_transform(pred_scaled)
true = ss.inverse_transform(y_scaled[split_index + sequence_length:].reshape(-1, 1))

# 성능 평가 (RMSE, R²)
from sklearn.metrics import root_mean_squared_error, r2_score
rmse = root_mean_squared_error(true, pred)
r2 = r2_score(true, pred)
print(f"✅ 검증 RMSE: {rmse:.4f}")
print(f"✅ R² Score: {r2:.4f}")

###############################################################################
###############################################################################
###############################################################################

# Epoch 1/50
# Epoch 1: val_loss improved from inf to 1.22784, saving model to bm.weights.h5
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 5220s 55ms/step - loss: 0.2131 - val_loss: 1.2278 - learning_rate: 0.0010
# Epoch 2/50
# 94921/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step - loss: 0.1058     
# Epoch 2: val_loss improved from 1.22784 to 1.06985, saving model to bm.weights.h5
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 5241s 55ms/step - loss: 0.1058 - val_loss: 1.0698 - learning_rate: 0.0010
# Epoch 3/50
# 94922/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - loss: 0.1042      
# Epoch 3: val_loss improved from 1.06985 to 1.02362, saving model to bm.weights.h5
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 4912s 52ms/step - loss: 0.1042 - val_loss: 1.0236 - learning_rate: 0.0010
# Epoch 4/50
# 94922/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 70ms/step - loss: 0.0936      
# Epoch 4: val_loss did not improve from 1.02362
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 8088s 85ms/step - loss: 0.0936 - val_loss: 1.3577 - learning_rate: 0.0010
# Epoch 5/50
# 94922/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 94ms/step - loss: 0.0884      
# Epoch 5: val_loss did not improve from 1.02362
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 10813s 114ms/step - loss: 0.0884 - val_loss: 1.3642 - learning_rate: 0.0010
# Epoch 6/50
# 94922/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 86ms/step - loss: 0.0990      
# Epoch 6: val_loss improved from 1.02362 to 0.84439, saving model to bm.weights.h5
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 9907s 104ms/step - loss: 0.0990 - val_loss: 0.8444 - learning_rate: 5.0000e-04
# Epoch 7/50
# 94922/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 86ms/step - loss: 0.0850      
# Epoch 7: val_loss improved from 0.84439 to 0.80229, saving model to bm.weights.h5
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 9758s 103ms/step - loss: 0.0850 - val_loss: 0.8023 - learning_rate: 5.0000e-04
# Epoch 8/50
# 94922/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step - loss: 0.0804      
# Epoch 8: val_loss did not improve from 0.80229
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 4699s 49ms/step - loss: 0.0804 - val_loss: 0.8154 - learning_rate: 5.0000e-04
# Epoch 9/50
# 94921/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step - loss: 0.0733     
# Epoch 9: val_loss improved from 0.80229 to 0.76390, saving model to bm.weights.h5
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 4825s 51ms/step - loss: 0.0733 - val_loss: 0.7639 - learning_rate: 5.0000e-04
# Epoch 10/50
# 94922/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step - loss: 0.0694     
# Epoch 10: val_loss improved from 0.76390 to 0.72749, saving model to bm.weights.h5
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 4863s 51ms/step - loss: 0.0694 - val_loss: 0.7275 - learning_rate: 5.0000e-04
# Epoch 11/50
# 94921/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - loss: 0.0652     
# Epoch 11: val_loss improved from 0.72749 to 0.68035, saving model to bm.weights.h5
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 6197s 65ms/step - loss: 0.0652 - val_loss: 0.6804 - learning_rate: 5.0000e-04
# Epoch 12/50
# 94921/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step - loss: 0.0616     
# Epoch 12: val_loss improved from 0.68035 to 0.62071, saving model to bm.weights.h5
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 4844s 51ms/step - loss: 0.0616 - val_loss: 0.6207 - learning_rate: 5.0000e-04
# Epoch 13/50
# 94922/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step - loss: 0.0587     
# Epoch 13: val_loss did not improve from 0.62071
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 5430s 57ms/step - loss: 0.0587 - val_loss: 0.6366 - learning_rate: 5.0000e-04
# Epoch 14/50
# 94921/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - loss: 0.0573     
# Epoch 14: val_loss did not improve from 0.62071
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 5001s 53ms/step - loss: 0.0573 - val_loss: 0.6323 - learning_rate: 5.0000e-04
# Epoch 15/50
# 94922/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 0.0640     
# Epoch 15: val_loss improved from 0.62071 to 0.47647, saving model to bm.weights.h5
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 5322s 56ms/step - loss: 0.0640 - val_loss: 0.4765 - learning_rate: 2.5000e-04
# Epoch 16/50
# 94921/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 47ms/step - loss: 0.0598     
# Epoch 16: val_loss improved from 0.47647 to 0.41735, saving model to bm.weights.h5
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 5302s 56ms/step - loss: 0.0598 - val_loss: 0.4173 - learning_rate: 2.5000e-04
# Epoch 17/50
# 94922/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 70ms/step - loss: 0.0581     
# Epoch 17: val_loss improved from 0.41735 to 0.40943, saving model to bm.weights.h5
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 7467s 79ms/step - loss: 0.0581 - val_loss: 0.4094 - learning_rate: 2.5000e-04
# Epoch 18/50
# 94922/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step - loss: 0.0568     
# Epoch 18: val_loss improved from 0.40943 to 0.40864, saving model to bm.weights.h5
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 4969s 52ms/step - loss: 0.0568 - val_loss: 0.4086 - learning_rate: 2.5000e-04
# Epoch 19/50
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 0.0557     
# Epoch 19: val_loss improved from 0.40864 to 0.40087, saving model to bm.weights.h5
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 5212s 55ms/step - loss: 0.0557 - val_loss: 0.4009 - learning_rate: 2.5000e-04
# Epoch 20/50
# 94922/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 0.0546     
# Epoch 20: val_loss did not improve from 0.40087
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 5280s 56ms/step - loss: 0.0546 - val_loss: 0.4040 - learning_rate: 2.5000e-04
# Epoch 21/50
# 94922/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - loss: 0.0544     
# Epoch 21: val_loss improved from 0.40087 to 0.39560, saving model to bm.weights.h5
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 5182s 55ms/step - loss: 0.0544 - val_loss: 0.3956 - learning_rate: 2.5000e-04
# Epoch 22/50
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 0.0533     
# Epoch 22: val_loss improved from 0.39560 to 0.39214, saving model to bm.weights.h5
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 5262s 55ms/step - loss: 0.0533 - val_loss: 0.3921 - learning_rate: 2.5000e-04
# Epoch 23/50
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 0.0529     
# Epoch 23: val_loss did not improve from 0.39214
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 5245s 55ms/step - loss: 0.0529 - val_loss: 0.4062 - learning_rate: 2.5000e-04
# Epoch 24/50
# 94922/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 0.0523     
# Epoch 24: val_loss did not improve from 0.39214
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 5243s 55ms/step - loss: 0.0523 - val_loss: 0.3964 - learning_rate: 2.5000e-04
# Epoch 25/50
# 94922/94923 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - loss: 0.0603     
# Epoch 25: val_loss did not improve from 0.39214
# 94923/94923 ━━━━━━━━━━━━━━━━━━━━ 5208s 55ms/step - loss: 0.0603 - val_loss: 0.4073 - learning_rate: 1.2500e-04

# ✅ 검증 RMSE: 12.1630
# ✅ R² Score: 0.6589




































