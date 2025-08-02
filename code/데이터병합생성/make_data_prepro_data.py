# 📦 라이브러리 불러오기
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# 📁 데이터 불러오기
df = pd.read_csv('data.csv', encoding='cp949')
df.info()

# 시간 변수 추가
df['시간'] = pd.to_datetime(df['시간'])
df['연도'] = df['시간'].dt.year
df['월'] = df['시간'].dt.month
df['일'] = df['시간'].dt.day
df['시'] = df['시간'].dt.hour
df['요일'] = df['시간'].dt.weekday           # 0=월, 6=일
df['주말'] = df['요일'].isin([5, 6]).astype(int)
public_holidays = [
    # 2021
    '2021-01-01', '2021-02-11', '2021-02-12', '2021-02-13', '2021-03-01',
    '2021-05-05', '2021-05-19', '2021-06-06', '2021-08-15',
    '2021-09-20', '2021-09-21', '2021-09-22',
    '2021-10-03', '2021-10-09', '2021-12-25',

    # 2022
    '2022-01-01', '2022-01-31', '2022-02-01', '2022-02-02', '2022-03-01',
    '2022-03-09', '2022-05-05', '2022-05-08', '2022-06-01', '2022-06-06',
    '2022-08-15', '2022-09-09', '2022-09-10', '2022-09-11', '2022-09-12',
    '2022-10-03', '2022-10-09', '2022-10-10', '2022-12-25',

    # 2023
    '2023-01-01', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24',
    '2023-03-01', '2023-05-05', '2023-05-27', '2023-05-29', '2023-06-06',
    '2023-08-15', '2023-09-28', '2023-09-29', '2023-09-30',
    '2023-10-02', '2023-10-03', '2023-10-09', '2023-12-25'
]
public_holidays = pd.to_datetime(public_holidays)
df['공휴일'] = df['시간'].dt.normalize().isin(public_holidays).astype(int)

df['풍향'] = df['풍향'].where(df['풍향'] >= 0, np.nan)
df['일강수량'] = df['일강수량'].replace(-99.0, np.nan)
df['시간강수량'] = df['시간강수량'].replace(-99.0, np.nan)
df['기온'] = df['기온'].replace(-99.0, np.nan)
df['체감온도'] = df['체감온도'].replace(-99.0, np.nan)
df = df.dropna()

cols = ['호선','상하구분','AWS지점코드']
for col in cols:
    df[col] = df[col].astype('category')
    
df['일사량'] = df['일사량'].replace(-99.0, np.nan)
df['일사량_측정여부'] = df['일사량'].notna().astype(int)
df = df.drop(columns='일사량')

df['상대습도'] = df['상대습도'].replace(-99.0, np.nan)
df['상대습도'] = df['상대습도'].interpolate().fillna(method='ffill').fillna(method='bfill')
print('보간 후 남은 결측값:', df['상대습도'].isna().sum())

df.info()

# 1. float64 → float32
float_cols = df.select_dtypes(include=['float64']).columns
df[float_cols] = df[float_cols].astype('float32')

# 2. int64 → int32
int_cols = df.select_dtypes(include=['int64']).columns
df[int_cols] = df[int_cols].astype('int32')

# 3. object(문자열) → category
obj_cols = df.select_dtypes(include=['object']).columns
df[obj_cols] = df[obj_cols].astype('category')

df.info(memory_usage='deep')  # 메모리 변화 확인

# 저장
df.to_parquet('prepro_data.parquet')

###############################################################################
# 상관관계 시각화 코드 ---------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 윈도우인 경우 'Malgun Gothic' 사용
plt.rcParams['font.family'] = 'Malgun Gothic'

# 음수 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import seaborn as sns

# 1. 수치형 변수만 선택
numeric_df = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64'])

# 2. 상관관계 계산
corr = numeric_df.corr()

# 3. 히트맵 시각화
plt.figure(figsize=(14, 12))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5)
plt.title("📊 수치형 변수 간 상관관계 히트맵", fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
df = pd.read_parquet('prepro_data.parquet')
df.info()
np.unique(df['역번호'], return_counts=True)
np.unique(df['기온'], return_counts=True)
np.unique(df['풍향'], return_counts=True)
np.unique(df['풍속'], return_counts=True)
np.unique(df['일강수량'], return_counts=True)
np.unique(df['시간강수량'], return_counts=True)
np.unique(df['상대습도'], return_counts=True)
np.unique(df['체감온도'], return_counts=True)
np.unique(df['혼잡도'], return_counts=True)
np.unique(df['미세먼지'], return_counts=True)
np.unique(df['연도'], return_counts=True)
np.unique(df['월'], return_counts=True)
np.unique(df['일'], return_counts=True)
np.unique(df['시'], return_counts=True)
np.unique(df['요일'], return_counts=True)
np.unique(df['주말'], return_counts=True)
np.unique(df['공휴일'], return_counts=True)
np.unique(df['일사량_측정여부'], return_counts=True)

# 시간에 따른 혼잡도 -----------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

# 1. 시간(datetime 컬럼)에서 시(hour)만 추출
df['hour'] = pd.to_datetime(df['시간']).dt.hour

# 2. 시간대별 평균 혼잡도 계산
hourly_congestion = df.groupby('hour')['혼잡도'].mean()

# 3. 시각화
plt.figure(figsize=(10, 5))
sns.lineplot(x=hourly_congestion.index, y=hourly_congestion.values, marker='o')
plt.title('시간대별 평균 혼잡도')
plt.xlabel('시간 (Hour)')
plt.ylabel('평균 혼잡도')
plt.grid(True)
plt.xticks(range(0, 24))
plt.show()











