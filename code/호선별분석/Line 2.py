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
df1 = df[df['호선'] == 2].copy()
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
        filepath='Line2_best_model.weights.h5',      # 저장할 경로
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
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step - loss: 0.3004   
# Epoch 1: val_loss improved from inf to 0.13276, saving model to Line2_best_model.weights.h5
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 318s 27ms/step - loss: 0.3004 - val_loss: 0.1328 - learning_rate: 0.0010
# Epoch 2/50
# 11678/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 23ms/step - loss: 0.0960  
# Epoch 2: val_loss improved from 0.13276 to 0.09544, saving model to Line2_best_model.weights.h5
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 308s 26ms/step - loss: 0.0960 - val_loss: 0.0954 - learning_rate: 0.0010
# Epoch 3/50
# 11679/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step - loss: 0.0751   
# Epoch 3: val_loss improved from 0.09544 to 0.07526, saving model to Line2_best_model.weights.h5
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 316s 27ms/step - loss: 0.0751 - val_loss: 0.0753 - learning_rate: 0.0010
# Epoch 4/50
# 11678/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step - loss: 0.0641  
# Epoch 4: val_loss improved from 0.07526 to 0.07221, saving model to Line2_best_model.weights.h5
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 319s 27ms/step - loss: 0.0641 - val_loss: 0.0722 - learning_rate: 0.0010
# Epoch 5/50
# 11679/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step - loss: 0.0585   
# Epoch 5: val_loss improved from 0.07221 to 0.06597, saving model to Line2_best_model.weights.h5
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 332s 28ms/step - loss: 0.0585 - val_loss: 0.0660 - learning_rate: 0.0010
# Epoch 6/50
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step - loss: 0.0536   
# Epoch 6: val_loss did not improve from 0.06597
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 335s 29ms/step - loss: 0.0536 - val_loss: 0.0685 - learning_rate: 0.0010
# Epoch 7/50
# 11679/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0501   
# Epoch 7: val_loss did not improve from 0.06597
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 343s 29ms/step - loss: 0.0501 - val_loss: 0.0661 - learning_rate: 0.0010
# Epoch 8/50
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - loss: 0.0438  
# Epoch 8: val_loss improved from 0.06597 to 0.05998, saving model to Line2_best_model.weights.h5
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 343s 29ms/step - loss: 0.0438 - val_loss: 0.0600 - learning_rate: 5.0000e-04
# Epoch 9/50
# 11679/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - loss: 0.0408   
# Epoch 9: val_loss did not improve from 0.05998
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 347s 30ms/step - loss: 0.0408 - val_loss: 0.0628 - learning_rate: 5.0000e-04
# Epoch 10/50
# 11679/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0394   
# Epoch 10: val_loss improved from 0.05998 to 0.05658, saving model to Line2_best_model.weights.h5
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 346s 30ms/step - loss: 0.0394 - val_loss: 0.0566 - learning_rate: 5.0000e-04
# Epoch 11/50
# 11679/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0398   
# Epoch 11: val_loss improved from 0.05658 to 0.05188, saving model to Line2_best_model.weights.h5
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 333s 29ms/step - loss: 0.0398 - val_loss: 0.0519 - learning_rate: 5.0000e-04
# Epoch 12/50
# 11679/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step - loss: 0.0366   
# Epoch 12: val_loss did not improve from 0.05188
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 357s 31ms/step - loss: 0.0366 - val_loss: 0.0584 - learning_rate: 5.0000e-04
# Epoch 13/50
# 11679/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0369   
# Epoch 13: val_loss did not improve from 0.05188
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 338s 29ms/step - loss: 0.0369 - val_loss: 0.0522 - learning_rate: 5.0000e-04
# Epoch 14/50
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 25ms/step - loss: 0.0337   
# Epoch 14: val_loss improved from 0.05188 to 0.04894, saving model to Line2_best_model.weights.h5
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 330s 28ms/step - loss: 0.0337 - val_loss: 0.0489 - learning_rate: 2.5000e-04
# Epoch 15/50
# 11679/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0329   
# Epoch 15: val_loss did not improve from 0.04894
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 333s 29ms/step - loss: 0.0329 - val_loss: 0.0509 - learning_rate: 2.5000e-04
# Epoch 16/50
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0325   
# Epoch 16: val_loss did not improve from 0.04894
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 343s 29ms/step - loss: 0.0325 - val_loss: 0.0522 - learning_rate: 2.5000e-04
# Epoch 17/50
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - loss: 0.0314   
# Epoch 17: val_loss improved from 0.04894 to 0.04752, saving model to Line2_best_model.weights.h5
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 348s 30ms/step - loss: 0.0314 - val_loss: 0.0475 - learning_rate: 1.2500e-04
# Epoch 18/50
# 11679/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 26ms/step - loss: 0.0315  
# Epoch 18: val_loss did not improve from 0.04752
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 346s 30ms/step - loss: 0.0315 - val_loss: 0.0491 - learning_rate: 1.2500e-04
# Epoch 19/50
# 11679/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - loss: 0.0317   
# Epoch 19: val_loss did not improve from 0.04752
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 351s 30ms/step - loss: 0.0317 - val_loss: 0.0493 - learning_rate: 1.2500e-04
# Epoch 20/50
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - loss: 0.0305   
# Epoch 20: val_loss improved from 0.04752 to 0.04691, saving model to Line2_best_model.weights.h5
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 351s 30ms/step - loss: 0.0305 - val_loss: 0.0469 - learning_rate: 6.2500e-05
# Epoch 21/50
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - loss: 0.0307  
# Epoch 21: val_loss did not improve from 0.04691
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 352s 30ms/step - loss: 0.0307 - val_loss: 0.0475 - learning_rate: 6.2500e-05
# Epoch 22/50
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - loss: 0.0304   
# Epoch 22: val_loss did not improve from 0.04691
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 350s 30ms/step - loss: 0.0304 - val_loss: 0.0474 - learning_rate: 6.2500e-05
# Epoch 23/50
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step - loss: 0.0301   
# Epoch 23: val_loss did not improve from 0.04691
# 11680/11680 ━━━━━━━━━━━━━━━━━━━━ 354s 30ms/step - loss: 0.0301 - val_loss: 0.0470 - learning_rate: 3.1250e-05

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
model.load_weights('Line2_best_model.weights.h5')

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
# ✅ 검증 RMSE: 4.4045
# ✅ R² Score: 0.9654
















