
###############################################################################

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

###############################################################################

# ✅ 기본 모델 가중치 한 번만 불러오기
model.load_weights('bm.weights.h5')
model.compile(loss='mse', optimizer=Adam(0.0005))

# 성능 저장 리스트
rmse_list, r2_list = [], []

for line_num in range(1, 9):
    print(f"\n🚇 {line_num}호선 전이학습 (누적) 시작")

    # 1. 해당 호선만 필터링
    df_line = df[df['호선'] == line_num].copy()

    # 2. 피처 재정의
    X_numeric_line = df_line[numeric_cols].values.astype(np.float32)
    X_station_line = df_line['역명'].values.astype(np.int32)
    X_aws_line     = df_line['AWS지점코드'].values.astype(np.int32)
    X_line_code    = df_line['호선'].values.astype(np.int32)
    y_line         = df_line['혼잡도'].values.astype(np.float32)

    # 3. 스케일링
    X_numeric_scaled_line = mm.transform(X_numeric_line)
    y_scaled_line = ss.transform(y_line.reshape(-1, 1)).flatten()

    # 4. 시계열 Dataset 생성
    split_idx = int(len(X_numeric_scaled_line) * 0.8)
    train_ds_line = create_tf_dataset(
        X_numeric_scaled_line[:split_idx],
        X_station_line[:split_idx],
        X_aws_line[:split_idx],
        X_line_code[:split_idx],
        y_scaled_line[:split_idx],
        sequence_length, batch_size
    )
    val_ds_line = create_tf_dataset(
        X_numeric_scaled_line[split_idx:],
        X_station_line[split_idx:],
        X_aws_line[split_idx:],
        X_line_code[split_idx:],
        y_scaled_line[split_idx:],
        sequence_length, batch_size
    )

    # 5. 콜백 정의 (호선별로 저장)
    callbacks = [
        ModelCheckpoint(filepath=f"finetuned_accum_line{line_num}.weights.h5",
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=1),
        EarlyStopping(monitor='val_loss',
                      patience=3,
                      restore_best_weights=True,
                      verbose=1)
    ]

    # 6. 누적 파인튜닝 (이전 학습 내용 유지)
    model.fit(train_ds_line,
              validation_data=val_ds_line,
              epochs=5,
              callbacks=callbacks,
              verbose=1)

    # 7. 예측 및 평가
    pred_line_scaled = model.predict(val_ds_line)
    y_true_scaled = np.concatenate([y.numpy() for _, y in val_ds_line])
    y_pred_line = ss.inverse_transform(pred_line_scaled)
    y_true_line = ss.inverse_transform(y_true_scaled.reshape(-1, 1))

    rmse = root_mean_squared_error(y_true_line, y_pred_line)
    r2 = r2_score(y_true_line, y_pred_line)
    print(f"✅ {line_num}호선 RMSE: {rmse:.4f}, R² Score: {r2:.4f}")

    rmse_list.append(rmse)
    r2_list.append(r2)

###############################################################################


# ✅ 1호선 RMSE: 13.1835, R² Score: 0.6855
# ✅ 2호선 RMSE: 11.5994, R² Score: 0.7601
# ✅ 3호선 RMSE: 11.5806, R² Score: 0.4214
# ✅ 4호선 RMSE: 11.2613, R² Score: 0.6844
# ✅ 5호선 RMSE: 6.7375, R² Score: 0.8787
# ✅ 6호선 RMSE: 6.4536, R² Score: 0.8268
# ✅ 7호선 RMSE: 10.1789, R² Score: 0.8137
# ✅ 8호선 RMSE: 5.0346, R² Score: 0.9427

###############################################################################

# ✅ 최종 누적 모델 가중치 불러오기 (예: 8호선까지 학습 완료된 모델)
model.load_weights('finetuned_accum_line8.weights.h5')

# ✅ 전체 검증 데이터셋 구성 (앞에서 쓴 val_ds 그대로 활용 가능)
# (주의: TimeseriesGenerator 또는 tf.data.Dataset은 이미 sequence_length 반영되어 생성됨)

# ✅ 전체 검증 데이터 예측
pred_scaled_all = model.predict(val_ds)

# ✅ 예측값 역변환
pred_all = ss.inverse_transform(pred_scaled_all)

# ✅ 실제값 역변환
true_all = ss.inverse_transform(y_scaled[split_index + sequence_length:].reshape(-1, 1))

# ✅ 최종 성능 평가
from sklearn.metrics import root_mean_squared_error, r2_score

rmse_final = root_mean_squared_error(true_all, pred_all)
r2_final = r2_score(true_all, pred_all)

print("\n📊 [최종 누적 모델 성능 - 전체 검증 데이터]")
print(f"✅ RMSE: {rmse_final:.4f}")
print(f"✅ R² Score: {r2_final:.4f}")

# 📊 [최종 누적 모델 성능 - 전체 검증 데이터]
# ✅ RMSE: 13.5906
# ✅ R² Score: 0.5741


###############################################################################



