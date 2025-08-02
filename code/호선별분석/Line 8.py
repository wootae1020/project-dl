# 📦 라이브러리 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# 📁 데이터 불러오기
df = pd.read_parquet('prepro_data.parquet')
df1 = df[df['호선'] == 8].copy()
cols = ['연도', '호선','상하구분','AWS지점코드']
for col in cols:
    df1[col] = df1[col].astype('category')

# 시 (0~23 기준)
df1['시_sin'] = np.sin(2 * np.pi * df1['시'] / 24)
df1['시_cos'] = np.cos(2 * np.pi * df1['시'] / 24)

# 요일 (0~6 기준, 월~일)
df1['요일_sin'] = np.sin(2 * np.pi * df1['요일'] / 7)
df1['요일_cos'] = np.cos(2 * np.pi * df1['요일'] / 7)

# 월, 일 (1~12, 1~31 기준)
le = LabelEncoder()
df['월'] = le.fit_transform(df['월'])
df['일'] = le.fit_transform(df['일'])

# 1. float64 → float32
float_cols = df1.select_dtypes(include=['float64']).columns
df1[float_cols] = df1[float_cols].astype('float32')

# 2. int64 → int32
int_cols = df1.select_dtypes(include=['int64']).columns
df1[int_cols] = df1[int_cols].astype('int32')

# 3. object(문자열) → category
obj_cols = df1.select_dtypes(include=['object']).columns
df1[obj_cols] = df1[obj_cols].astype('category')    
    
df1 = df1.drop(['시', '요일'], axis=1)
df1.info()

# ✅ [1] 피처 구성 및 전처리 --------------------------------------------------------

# 🎯 타겟 변수 '혼잡도' 제거하여 X 생성
X = df1.drop(['혼잡도', '시간'], axis=1)

# 🔤 범주형 변수 원-핫 인코딩 (첫 번째 카테고리는 drop)
o_h_e_x = pd.get_dummies(X, columns=['연도', '역명', '호선', '상하구분', 'AWS지점코드'], drop_first=True)

# 🔢 X 정규화 (0~1 범위로 스케일링)
mm = MinMaxScaler()
mm_x = mm.fit_transform(o_h_e_x)

# 🎯 y 정규화 (표준화: 평균 0, 표준편차 1)
y = df1[['혼잡도']]  # 2D로 유지
ss = StandardScaler()
ss_y = ss.fit_transform(y)

# ✅ [2] 시계열 학습/검증 데이터 분할 --------------------------------------------------

# 전체의 80%를 학습용, 나머지 20%를 검증용으로 사용
split_index = int(len(mm_x) * 0.8)

train_x = mm_x[:split_index]
train_y = ss_y[:split_index]

val_x = mm_x[split_index:]
val_y = ss_y[split_index:]

# ✅ [3] 시계열 데이터 생성 -----------------------------------------------------------

# 과거 21시간 데이터를 기반으로 예측
sequence_length = 21
batch_size = 128

# TimeseriesGenerator를 통해 시계열 학습/검증셋 생성
train_gen = TimeseriesGenerator(train_x, train_y, length=sequence_length, batch_size=batch_size)
val_gen   = TimeseriesGenerator(val_x, val_y, length=sequence_length, batch_size=batch_size)

# ✅ [4] LSTM 모델 정의 및 학습 --------------------------------------------------------

# 딥러닝 모델 구조 정의
model = Sequential([
    Input(shape=(sequence_length, train_x.shape[1])),  # (timesteps, features)
    LSTM(64, return_sequences=True),   # 첫 번째 LSTM: 시퀀스 출력 유지
    Dropout(0.2),
    LSTM(32),                          # 두 번째 LSTM: 최종 출력만 반환
    Dropout(0.2),
    Dense(1)                           # 예측값 하나 출력 (혼잡도)
])

# 모델 컴파일
model.compile(loss='mse', optimizer=Adam(0.001))

# 콜백 설정
callbacks = [
    ModelCheckpoint(
        filepath='Line8_best_model.weights.h5',      # 저장할 경로
        monitor='val_loss',            # 기준 지표
        save_best_only=True,           # 가장 성능 좋은 모델만 저장
        save_weights_only=True,        # 전체 모델이 아니라 가중치만 저장
        verbose=1
    ),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)
]

# 모델 학습
history = model.fit(train_gen, validation_data=val_gen, epochs=50, callbacks=callbacks)

# Epoch 1/50
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step - loss: 0.3367   
# Epoch 1: val_loss improved from inf to 0.06764, saving model to Line8_best_model.weights.h5
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 140s 29ms/step - loss: 0.3366 - val_loss: 0.0676 - learning_rate: 0.0010
# Epoch 2/50
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0895  
# Epoch 2: val_loss improved from 0.06764 to 0.05913, saving model to Line8_best_model.weights.h5
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 138s 29ms/step - loss: 0.0895 - val_loss: 0.0591 - learning_rate: 0.0010
# Epoch 3/50
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0702  
# Epoch 3: val_loss improved from 0.05913 to 0.04694, saving model to Line8_best_model.weights.h5
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 142s 29ms/step - loss: 0.0702 - val_loss: 0.0469 - learning_rate: 0.0010
# Epoch 4/50
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0627  
# Epoch 4: val_loss did not improve from 0.04694
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 141s 29ms/step - loss: 0.0627 - val_loss: 0.0658 - learning_rate: 0.0010
# Epoch 5/50
# 4824/4825 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0558  
# Epoch 5: val_loss improved from 0.04694 to 0.03534, saving model to Line8_best_model.weights.h5
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 140s 29ms/step - loss: 0.0558 - val_loss: 0.0353 - learning_rate: 0.0010
# Epoch 6/50
# 4824/4825 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - loss: 0.0502  
# Epoch 6: val_loss improved from 0.03534 to 0.03121, saving model to Line8_best_model.weights.h5
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 145s 30ms/step - loss: 0.0502 - val_loss: 0.0312 - learning_rate: 0.0010
# Epoch 7/50
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0455  
# Epoch 7: val_loss improved from 0.03121 to 0.03004, saving model to Line8_best_model.weights.h5
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 141s 29ms/step - loss: 0.0455 - val_loss: 0.0300 - learning_rate: 0.0010
# Epoch 8/50
# 4824/4825 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - loss: 0.0437  
# Epoch 8: val_loss improved from 0.03004 to 0.02668, saving model to Line8_best_model.weights.h5
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 149s 31ms/step - loss: 0.0437 - val_loss: 0.0267 - learning_rate: 0.0010
# Epoch 9/50
# 4824/4825 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step - loss: 0.0415  
# Epoch 9: val_loss did not improve from 0.02668
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 149s 31ms/step - loss: 0.0415 - val_loss: 0.0274 - learning_rate: 0.0010
# Epoch 10/50
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0412  
# Epoch 10: val_loss improved from 0.02668 to 0.02581, saving model to Line8_best_model.weights.h5
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 143s 30ms/step - loss: 0.0412 - val_loss: 0.0258 - learning_rate: 0.0010
# Epoch 11/50
# 4824/4825 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0378  
# Epoch 11: val_loss improved from 0.02581 to 0.02359, saving model to Line8_best_model.weights.h5
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 144s 30ms/step - loss: 0.0378 - val_loss: 0.0236 - learning_rate: 0.0010
# Epoch 12/50
# 4824/4825 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0368  
# Epoch 12: val_loss did not improve from 0.02359
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 143s 30ms/step - loss: 0.0368 - val_loss: 0.0289 - learning_rate: 0.0010
# Epoch 13/50
# 4824/4825 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0359  
# Epoch 13: val_loss improved from 0.02359 to 0.02091, saving model to Line8_best_model.weights.h5
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 142s 29ms/step - loss: 0.0359 - val_loss: 0.0209 - learning_rate: 0.0010
# Epoch 14/50
# 4824/4825 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - loss: 0.0368  
# Epoch 14: val_loss did not improve from 0.02091
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 151s 31ms/step - loss: 0.0368 - val_loss: 0.0234 - learning_rate: 0.0010
# Epoch 15/50
# 4824/4825 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - loss: 0.0344  
# Epoch 15: val_loss did not improve from 0.02091
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 143s 30ms/step - loss: 0.0344 - val_loss: 0.0241 - learning_rate: 0.0010
# Epoch 16/50
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - loss: 0.0318  
# Epoch 16: val_loss did not improve from 0.02091
# 4825/4825 ━━━━━━━━━━━━━━━━━━━━ 147s 30ms/step - loss: 0.0318 - val_loss: 0.0281 - learning_rate: 5.0000e-04

# 📊 학습 및 검증 손실 시각화
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# ✅ [5] 예측 및 평가 ---------------------------------------------------------------

# 최고 성능의 가중치 로드
model.load_weights('Line8_best_model.weights.h5')

# 검증셋에 대해 예측 수행
pred_scaled = model.predict(val_gen)

# 정규화 해제 (역정규화) → 실제 혼잡도 단위로 복원
pred = ss.inverse_transform(pred_scaled)

# 실제 y도 시계열 슬라이싱 기준에 맞게 자르기
true = ss.inverse_transform(val_y[sequence_length:])

# 평가 지표 계산
rmse = mean_squared_error(true, pred, squared=False)  # RMSE
r2 = r2_score(true, pred)                             # R² Score

# 결과 출력
print(f"✅ 검증 RMSE: {rmse:.4f}")
print(f"✅ R² Score: {r2:.4f}")
# ✅ 검증 RMSE: 3.1449
# ✅ R² Score: 0.9776
















