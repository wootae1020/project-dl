###############################################################################
# train_subway 데이터 병합 -----------------------------------------------------------
import pandas as pd
df21 = pd.read_csv('train_subway21.csv', encoding='cp949')
df22 = pd.read_csv('train_subway22.csv', encoding='cp949')
df23 = pd.read_csv('train_subway23.csv', encoding='cp949')

df21.columns = ['시간', '호선', '역번호', '역명', '상하구분', 'AWS지점코드', '기온', '풍향', '풍속', '일강수량', '시간강수량', '상대습도', '일사량', '체감온도', '혼잡도']
df22.columns = ['시간', '호선', '역번호', '역명', '상하구분', 'AWS지점코드', '기온', '풍향', '풍속', '일강수량', '시간강수량', '상대습도', '일사량', '체감온도', '혼잡도']
df23.columns = ['시간', '호선', '역번호', '역명', '상하구분', 'AWS지점코드', '기온', '풍향', '풍속', '일강수량', '시간강수량', '상대습도', '일사량', '체감온도', '혼잡도']

df_merged = pd.concat([df21, df22, df23], ignore_index=True)

df_merged.to_csv('train_subway.csv', encoding='cp949', index=False)

###############################################################################
# 승하차 인원수 데이터 병합 -----------------------------------------------------
import pandas as pd
df21 = pd.read_csv('CARD_SUBWAY_MONTH_2021.csv', encoding='cp949', index_col=False)
df22 = pd.read_csv('CARD_SUBWAY_MONTH_2022.csv', encoding='cp949', index_col=False)

monthly_data = []
for month in range(1, 13):
    file_name = f'CARD_SUBWAY_MONTH_2023{month:02d}.csv'
    try:
        df = pd.read_csv(file_name, encoding='utf-8', index_col=False)
        monthly_data.append(df)
        print(f"{file_name} 불러오기 성공")
    except Exception as e:
        print(f"{file_name} 불러오기 실패: {e}")

df23 = pd.concat(monthly_data, ignore_index=True)

df21 = df21.drop('등록일자', axis=1)
df22 = df22.drop('등록일자', axis=1)
df23 = df23.drop('등록일자', axis=1)

df_merged = pd.concat([df21,df22,df23], ignore_index=True)
df_merged.to_csv('people.csv', encoding='cp949', index=False)

###############################################################################
# 미세먼지 데이터 병합 ----------------------------------------------------------
## 2021년도 --------------------------------------------------------------------
import pandas as pd

df_list = []

for i in range(12):
    dfi = pd.read_excel(f'day ({i}).xlsx', header=4)
    dfi = dfi.iloc[[0]]  # 첫 번째 행만 선택
    dfi = dfi.drop(columns=['Unnamed: 1'])  # 불필요한 열 제거
    df_list.append(dfi)

# 하나의 DataFrame으로 병합
df21 = pd.concat(df_list, ignore_index=True)
df21 = df21.drop(columns=['구분'])
df21 = df21.T
df21.columns = [f'{i}월' for i in range(1, 13)]

# 1. 인덱스를 숫자만 추출하여 '일'로 변환
df21.index = df21.index.str.replace('일', '').astype(int)

# 2. melt()로 long format으로 변환
df_long = df21.reset_index().melt(id_vars='index', var_name='월', value_name='미세먼지')
df_long = df_long.rename(columns={'index': '일'})

# 3. '시간' 컬럼 만들기 (2021년 기준 예시)
df_long['시간'] = pd.to_datetime('2021-' + df_long['월'].str.replace('월', '').str.zfill(2) + '-' + df_long['일'].astype(str).str.zfill(2), errors='coerce')

# 4. 정렬 및 컬럼 정리
df_long = df_long[['시간', '미세먼지']].dropna(subset=['시간'])  # 유효한 날짜만 남김
df_long = df_long.sort_values(by='시간').reset_index(drop=True)

# 결과 저장
df_long.to_csv('미세먼지21.csv', index=False)

## 2022년도 --------------------------------------------------------------------
import pandas as pd

df_list = []

for i in range(12):
    dfi = pd.read_excel(f'day ({i}).xlsx', header=4)
    dfi = dfi.iloc[[0]]  # 첫 번째 행만 선택
    dfi = dfi.drop(columns=['Unnamed: 1'])  # 불필요한 열 제거
    df_list.append(dfi)

# 하나의 DataFrame으로 병합
df22 = pd.concat(df_list, ignore_index=True)
df22 = df22.drop(columns=['구분'])
df22 = df22.T
df22.columns = [f'{i}월' for i in range(1, 13)]

# 1. 인덱스를 숫자만 추출하여 '일'로 변환
df22.index = df22.index.str.replace('일', '').astype(int)

# 2. melt()로 long format으로 변환
df_long = df22.reset_index().melt(id_vars='index', var_name='월', value_name='미세먼지')
df_long = df_long.rename(columns={'index': '일'})

# 3. '시간' 컬럼 만들기 (2022년 기준 예시)
df_long['시간'] = pd.to_datetime('2022-' + df_long['월'].str.replace('월', '').str.zfill(2) + '-' + df_long['일'].astype(str).str.zfill(2), errors='coerce')

# 4. 정렬 및 컬럼 정리
df_long = df_long[['시간', '미세먼지']].dropna(subset=['시간'])  # 유효한 날짜만 남김
df_long = df_long.sort_values(by='시간').reset_index(drop=True)

# 결과 저장
df_long.to_csv('미세먼지22.csv', index=False)

## 2023년도 --------------------------------------------------------------------
import pandas as pd

df_list = []

for i in range(12):
    dfi = pd.read_excel(f'day ({i}).xlsx', header=4)
    dfi = dfi.iloc[[0]]  # 첫 번째 행만 선택
    dfi = dfi.drop(columns=['Unnamed: 1'])  # 불필요한 열 제거
    df_list.append(dfi)

# 하나의 DataFrame으로 병합
df23 = pd.concat(df_list, ignore_index=True)
df23 = df23.drop(columns=['구분'])
df23 = df23.T
df23.columns = [f'{i}월' for i in range(1, 13)]

# 1. 인덱스를 숫자만 추출하여 '일'로 변환
df23.index = df23.index.str.replace('일', '').astype(int)

# 2. melt()로 long format으로 변환
df_long = df23.reset_index().melt(id_vars='index', var_name='월', value_name='미세먼지')
df_long = df_long.rename(columns={'index': '일'})

# 3. '시간' 컬럼 만들기 (2023년 기준 예시)
df_long['시간'] = pd.to_datetime('2023-' + df_long['월'].str.replace('월', '').str.zfill(2) + '-' + df_long['일'].astype(str).str.zfill(2), errors='coerce')

# 4. 정렬 및 컬럼 정리
df_long = df_long[['시간', '미세먼지']].dropna(subset=['시간'])  # 유효한 날짜만 남김
df_long = df_long.sort_values(by='시간').reset_index(drop=True)

# 결과 저장
df_long.to_csv('미세먼지23.csv', index=False)

## 2021 + 2022 + 2023 ---------------------------------------------------------
import pandas as pd
df1 = pd.read_csv('미세먼지21.csv')
df2 = pd.read_csv('미세먼지22.csv')
df3 = pd.read_csv('미세먼지23.csv')
df = pd.concat([df1,df2,df3])
df.to_csv('미세먼지.csv', index=False)

###############################################################################
# train_subway 데이터 + 승하차 데이터 + 미세먼지 데이터 ---------------------------
## train_subway 데이터 + 승하차 데이터 ------------------------------------------
import pandas as pd
df1 = pd.read_csv('train_subway.csv', encoding='cp949', index_col=False)
df2 = pd.read_csv('people.csv', encoding='cp949', index_col=False)
df1['날짜'] = pd.to_datetime(df1['시간']).dt.date.astype(str)

import re
# 괄호 안의 모든 내용 제거 (예: '강변(동서울터미널)' → '강변')
df2['역명'] = df2['역명'].str.replace(r'\(.*?\)', '', regex=True).str.strip()

df2.duplicated(subset=['사용일자', '역명']).sum()

df2[df2.duplicated(subset=['사용일자', '역명'], keep=False)] \
    .sort_values(['사용일자', '역명']) \
    .head(20)

df2 = df2.drop('노선명', axis=1)

df2_grouped = df2.groupby(['사용일자', '역명'], as_index=False)[['승차총승객수', '하차총승객수']].sum()

df2_grouped.duplicated(subset=['사용일자', '역명']).sum()

df_merged = pd.merge(
    df1,
    df2_grouped,
    left_on=['날짜', '역명'],
    right_on=['사용일자', '역명'],
    how='left'
)

df_merged.isna().mean().sort_values(ascending=False)

df_missing = df_merged[df_merged.isna().any(axis=1)]
df_missing

df = df_merged.dropna()
df.isna().mean().sort_values(ascending=False)

df.to_csv('subway.csv', encoding='cp949', index=False)

## train_subway 데이터 + 승하차 데이터 + 미세먼지 데이터 --------------------------
import pandas as pd
df1 = pd.read_csv('subway.csv', encoding='cp949')
df2 = pd.read_csv('미세먼지.csv')

# 1. df1에서 '일자' 추출 (datetime 형식으로 처리)
df1['일자'] = pd.to_datetime(df1['시간']).dt.date

# 2. df2의 컬럼 이름을 '일자'로 바꾸고, datetime.date 타입으로 변환
df2 = df2.rename(columns={'시간': '일자'})
df2['일자'] = pd.to_datetime(df2['일자']).dt.date

# 3. 병합
df1 = df1.merge(df2, on='일자', how='left')  # left join으로 df1 기준 유지
df1.isnull().sum()

# 4. 컬럼 제거
df1 = df1.drop(columns=['일자', '날짜', '사용일자'])

df1.to_csv('data.csv', encoding='cp949', index=False)

###############################################################################
# 전처리 시작 ------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 음수 부호 깨짐 방지

df = pd.read_csv('data.csv', encoding='cp949')

# 시간을 데이트타임 형태로 변경 후 정렬
df['시간'] = pd.to_datetime(df['시간'])
df = df.sort_values('시간')

# 결측치 확인
missing_counts = df.isna().sum()
print(missing_counts / len(df) * 100)

# 수치형이 아닌 컬럼을 카테고리 타입으로 변경
cols = ['호선', '역번호', '역명', '상하구분', 'AWS지점코드']
for col in cols:
    df[col] = df[col].astype('category')

# 음수값 확인
numeric_cols = df.select_dtypes(include=[np.number]).columns

# 음수가 존재하는지 확인
for col in numeric_cols:
    min_val = df[col].min()
    if min_val < 0:
        print(f"🔴 {col} → 음수 값 존재 (최솟값: {min_val})")
    else:
        print(f"🟢 {col} → 음수 없음 (최솟값: {min_val})")
        
# -99.0 값의 비율 계산
for col in numeric_cols:
    total_count = df[col].count()  # NaN 제외한 값 개수
    minus99_count = (df[col] == -99.0).sum()
    ratio = minus99_count / total_count * 100

    if minus99_count > 0:
        print(f"🔴 {col} → -99.0 개수: {minus99_count}개 ({ratio:.2f}%)")       

# -99.0을 제외한 음수의 비율
for col in numeric_cols:
    total_count = df[col].count()  # NaN 제외
    true_negative_count = ((df[col] < 0) & (df[col] != -99.0)).sum()

    if true_negative_count > 0:
        ratio = true_negative_count / total_count * 100
        print(f"🔴 {col} → 진짜 음수 개수: {true_negative_count}개 ({ratio:.4f}%)")
    else:
        print(f"🟢 {col} → 진짜 음수 없음")      
        
# 풍향, 풍속은 음수가 물리적으로 존재할 수 없음
df['풍향'] = df['풍향'].where(df['풍향'] >= 0, np.nan)
df['풍속'] = df['풍속'].where(df['풍속'] >= 0, np.nan)
  
# 일강수량, 시간강수량 에서 0 확인
total = df.shape[0]
일강수량_0비율 = (df['일강수량'] == 0).sum() / total * 100
시간강수량_0비율 = (df['시간강수량'] == 0).sum() / total * 100
print(f'일강수량 = 0 비율: {일강수량_0비율:.2f}%')
print(f'시간강수량 = 0 비율: {시간강수량_0비율:.2f}%')

# 기온, 일강수량, 시간강수량, 체감온도의 -99를 NaN으로 대체(비율이 적음)
df['기온'] = df['기온'].replace(-99.0, np.nan)
df['일강수량'] = df['일강수량'].replace(-99.0, np.nan)
df['시간강수량'] = df['시간강수량'].replace(-99.0, np.nan)
df['체감온도'] = df['체감온도'].replace(-99.0, np.nan)
df = df.dropna()

# 일사량에서 -99.0이 언제 많이 발생하는지 시각화
df['시'] = df['시간'].dt.hour
sns.countplot(x='시', data=df[df['일사량'] == -99.0])
plt.title('일사량 -99.0 발생 시간대 분포')
plt.show()

# 연도별로 일사량에서 -99.0이 언제 많이 발생하는지 시각화
df['연도'] = df['시간'].dt.year
df['시'] = df['시간'].dt.hour
df_99 = df[df['일사량'] == -99.0]
plt.figure(figsize=(12, 6))
sns.countplot(data=df_99, x='시', hue='연도')
plt.title('연도별 일사량 -99.0 발생 시간대 분포')
plt.xlabel('시')
plt.ylabel('count')
plt.legend(title='연도')
plt.tight_layout()
plt.show()

# 일사량 1(미측정), 0(측정)으로 처리
df['일사량'] = df['일사량'].replace(-99.0, np.nan)
df['일사량'] = np.where(df['일사량'].isna(), 1, 0)

# 상대습도 확인
# 상대습도가 -99.0인 행 필터링
df_missing_hum = df[df['상대습도'] == -99.0].copy()

# 연도, 시간대 파생
df_missing_hum['연도'] = df_missing_hum['시간'].dt.year
df_missing_hum['시'] = df_missing_hum['시간'].dt.hour

# 시간대별 분포 확인
plt.figure(figsize=(10, 5))
sns.countplot(data=df_missing_hum, x='시')
plt.title('상대습도 -99.0 발생 시간대 분포')
plt.xlabel('시간대 (시)')
plt.ylabel('결측값 개수')
plt.tight_layout()
plt.show()

# 연도별 + 시간대별 분포 (hue로 구분)
plt.figure(figsize=(12, 6))
sns.countplot(data=df_missing_hum, x='시', hue='연도')
plt.title('연도별 상대습도 -99.0 발생 시간대 분포')
plt.xlabel('시간대 (시)')
plt.ylabel('결측값 개수')
plt.legend(title='연도')
plt.tight_layout()
plt.show()

# 연도별 상대습도 비율 확인
year_total = df['연도'].value_counts().sort_index()
missing_by_year = df_missing_hum['연도'].value_counts().sort_index()
humidity_missing_ratio = (missing_by_year / year_total * 100).round(2)
print(humidity_missing_ratio)

# 상대습도 선형 보간으로 결측값 채우기
df['상대습도'] = df['상대습도'].replace(-99.0, np.nan)
df['상대습도'] = (
    df['상대습도']
    .interpolate(method='linear')
    .fillna(method='ffill')
    .fillna(method='bfill')
)
print('최종 결측값 개수:', df['상대습도'].isna().sum())  # ➜ 0이 되어야 함

df = df.drop(['시','연도'], axis=1)

# 추가 파생 변수 생성
df['요일'] = df['시간'].dt.dayofweek
df['주말여부'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)
public_holidays = [
    # 🎉 2021년 공휴일
    '2021-01-01', '2021-02-11', '2021-02-12', '2021-02-13', '2021-03-01',
    '2021-05-05', '2021-05-19', '2021-06-06', '2021-08-15',
    '2021-09-20', '2021-09-21', '2021-09-22',
    '2021-10-03', '2021-10-09', '2021-12-25',

    # 🎉 2022년 공휴일
    '2022-01-01', '2022-01-31', '2022-02-01', '2022-02-02', '2022-03-01',
    '2022-03-09', '2022-05-05', '2022-05-08', '2022-06-01', '2022-06-06',
    '2022-08-15', '2022-09-09', '2022-09-10', '2022-09-11', '2022-09-12',
    '2022-10-03', '2022-10-09', '2022-10-10', '2022-12-25',

    # 🎉 2023년 공휴일
    '2023-01-01', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24',
    '2023-03-01', '2023-05-05', '2023-05-27', '2023-05-29', '2023-06-06',
    '2023-08-15', '2023-09-28', '2023-09-29', '2023-09-30',
    '2023-10-02', '2023-10-03', '2023-10-09', '2023-12-25'
]
public_holidays = pd.to_datetime(public_holidays)

# 2. 공휴일 여부 컬럼 생성
df['공휴일'] = df['시간'].dt.normalize().isin(public_holidays).astype(int)

# 지하철 요금 인상 가격 반영을 위한 코드
기준일 = pd.to_datetime('2023-10-06')
df['요금'] = df['시간'].apply(lambda x: 1300 if x <= 기준일 else 1450)

# 미세먼지 범주 컬럼 추가
df['pm10_level'] = np.where(df['미세먼지'] >= 151, '매우나쁨',
                    np.where(df['미세먼지'] >= 81, '나쁨',
                    np.where(df['미세먼지'] >= 31, '보통',
                    np.where(df['미세먼지'] >= 0, '좋음', None))))

label_map = {'좋음': 0, '보통': 1, '나쁨': 2, '매우나쁨': 3}
df['pm10_level'] = df['pm10_level'].map(label_map)

df.info()
# 최종 저장
df.to_csv('prepro_data.csv', encoding='cp949', index=False)

###############################################################################
# 변수간 상관관계 파악 ----------------------------------------------------------
import pandas as pd
df = pd.read_csv('prepro_data.csv', encoding='cp949')
df.info()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 음수 부호 깨짐 방지

# 수치형 컬럼만 추출
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# 상관계수 계산
corr = df[numeric_cols].corr()

# 상관관계 히트맵 시각화
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, mask=np.triu(corr))
plt.title("Correlation between Numeric Features", fontsize=14)
plt.tight_layout()
plt.show()

# 0.00 호선
# 0.01 일강수량, 시간강수량
# 0.02 pm10_level, 미세먼지

# ['역번호', '역명', '상하구분', 'AWS지점코드', '혼잡도'] 상관관계 계산
df['역명'] = df['역명'].astype('category').cat.codes
df['상하구분'] = df['상하구분'].astype('category').cat.codes
cols = ['역번호', '역명', '상하구분', 'AWS지점코드', '혼잡도']
corr = df[cols].corr()
congestion_corr = corr['혼잡도'].drop('혼잡도').sort_values(ascending=False)
print("📊 혼잡도와의 상관계수:")
print(congestion_corr)

# 저장
df = df.drop(['호선','일강수량','시간강수량','pm10_level','미세먼지'], axis=1)
df.to_csv('modeling_data.csv', encoding='cp949', index=False)

###############################################################################
# 모델링1 ----------------------------------------------------------------------
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

df = pd.read_csv('modeling_data.csv', encoding='cp949')
df.info()

df['시간'] = pd.to_datetime(df['시간'])

cols = ['역번호', '역명', '상하구분', 'AWS지점코드']
for col in cols:
    df[col] = df[col].astype('category')

float_cols = df.select_dtypes(include='float64').columns
df[float_cols] = df[float_cols].astype('float32')

int_cols = df.select_dtypes(include='int64').columns
df[int_cols] = df[int_cols].astype('int32')


df.info()

# ✅ [1] 피처 구성 및 전처리

# 🎯 타겟 변수 '혼잡도' 제거하여 X 생성
X = df.drop(['혼잡도', '시간', '역번호', '역명', '상하구분'], axis=1)


# 🔤 범주형 변수 원-핫 인코딩 (첫 번째 카테고리는 drop)
o_h_e_x = pd.get_dummies(X, columns=['AWS지점코드'], drop_first=True)

# 🔢 X 정규화 (0~1 범위로 스케일링)
mm = MinMaxScaler()
mm_x = mm.fit_transform(o_h_e_x)

# 🎯 y 정규화 (표준화: 평균 0, 표준편차 1)
y = np.log1p(df[['혼잡도']])
ss = StandardScaler()
ss_y = ss.fit_transform(y)

# ✅ [2] 시계열 학습/검증 데이터 분할

# 전체의 80%를 학습용, 나머지 20%를 검증용으로 사용
split_index = int(len(mm_x) * 0.8)

train_x = mm_x[:split_index]
train_y = ss_y[:split_index]

val_x = mm_x[split_index:]
val_y = ss_y[split_index:]

# ✅ [3] 시계열 데이터 생성

# 과거 24시간 데이터를 기반으로 예측
sequence_length = 24
batch_size = 128

# TimeseriesGenerator를 통해 시계열 학습/검증셋 생성
train_gen = TimeseriesGenerator(train_x, train_y, length=sequence_length, batch_size=batch_size)
val_gen   = TimeseriesGenerator(val_x, val_y, length=sequence_length, batch_size=batch_size)

# ✅ [4] LSTM 모델 정의 및 학습

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
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)

# 모델 학습
history = model.fit(train_gen, validation_data=val_gen,
                    epochs=10, callbacks=[early_stop, reduce_lr])

# ✅ [5] 예측 및 평가

# 검증셋에 대해 예측 수행
pred_scaled = model.predict(val_gen)

# 정규화 해제 (역정규화) → 실제 혼잡도 단위로 복원
pred = ss.inverse_transform(pred_scaled)
pred = np.expm1(pred)

# 실제 y도 시계열 슬라이싱 기준에 맞게 자르기
true = ss.inverse_transform(val_y[sequence_length:])
true = np.expm1(true)

# 평가 지표 계산
rmse = mean_squared_error(true, pred, squared=False)  # RMSE
r2 = r2_score(true, pred)                             # R² Score

# 결과 출력
print(f"✅ 검증 RMSE: {rmse:.4f}")
print(f"✅ R² Score: {r2:.4f}")
# ✅ 검증 RMSE: 22.0393
# ✅ R² Score: -0.0812
###############################################################################
# 모델링2 ----------------------------------------------------------------------
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

df = pd.read_csv('modeling_data.csv', encoding='cp949')
df.info()

df['시간'] = pd.to_datetime(df['시간'])

df['hour'] = df['시간'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['weekday_sin'] = np.sin(2 * np.pi * df['요일'] / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['요일'] / 7)


cols = ['역번호', '역명', '상하구분', 'AWS지점코드']
for col in cols:
    df[col] = df[col].astype('category')

float_cols = df.select_dtypes(include='float64').columns
df[float_cols] = df[float_cols].astype('float32')

int_cols = df.select_dtypes(include='int64').columns
df[int_cols] = df[int_cols].astype('int32')


df.info()

# ✅ [1] 피처 구성 및 전처리

# 🎯 타겟 변수 '혼잡도' 제거하여 X 생성
X = df.drop(['혼잡도', '시간', '역번호', '역명'], axis=1)


# 🔤 범주형 변수 원-핫 인코딩 (첫 번째 카테고리는 drop)
o_h_e_x = pd.get_dummies(X, columns=['상하구분', 'AWS지점코드'], drop_first=True)

# 🔢 X 정규화 (0~1 범위로 스케일링)
mm = MinMaxScaler()
mm_x = mm.fit_transform(o_h_e_x)

# 🎯 y 정규화 (표준화: 평균 0, 표준편차 1)
y = np.log1p(df[['혼잡도']])
ss = StandardScaler()
ss_y = ss.fit_transform(y)

# ✅ [2] 시계열 학습/검증 데이터 분할

# 전체의 80%를 학습용, 나머지 20%를 검증용으로 사용
split_index = int(len(mm_x) * 0.8)

train_x = mm_x[:split_index]
train_y = ss_y[:split_index]

val_x = mm_x[split_index:]
val_y = ss_y[split_index:]

# ✅ [3] 시계열 데이터 생성

# 과거 24시간 데이터를 기반으로 예측
sequence_length = 24
batch_size = 128

# TimeseriesGenerator를 통해 시계열 학습/검증셋 생성
train_gen = TimeseriesGenerator(train_x, train_y, length=sequence_length, batch_size=batch_size)
val_gen   = TimeseriesGenerator(val_x, val_y, length=sequence_length, batch_size=batch_size)

# ✅ [4] LSTM 모델 정의 및 학습

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
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)

# 모델 학습
history = model.fit(train_gen, validation_data=val_gen,
                    epochs=10, callbacks=[early_stop, reduce_lr])

# ✅ [5] 예측 및 평가

# 검증셋에 대해 예측 수행
pred_scaled = model.predict(val_gen)

# 정규화 해제 (역정규화) → 실제 혼잡도 단위로 복원
pred = ss.inverse_transform(pred_scaled)
pred = np.expm1(pred)

# 실제 y도 시계열 슬라이싱 기준에 맞게 자르기
true = ss.inverse_transform(val_y[sequence_length:])
true = np.expm1(true)

# 평가 지표 계산
rmse = mean_squared_error(true, pred, squared=False)  # RMSE
r2 = r2_score(true, pred)                             # R² Score

# 결과 출력
print(f"✅ 검증 RMSE: {rmse:.4f}")
print(f"✅ R² Score: {r2:.4f}")

###############################################################################
# 모델링3 ----------------------------------------------------------------------
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

df = pd.read_csv('modeling_data.csv', encoding='cp949')
df.info()

df['시간'] = pd.to_datetime(df['시간'])

df['hour'] = df['시간'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['weekday_sin'] = np.sin(2 * np.pi * df['요일'] / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['요일'] / 7)


cols = ['역번호', '역명', '상하구분', 'AWS지점코드']
for col in cols:
    df[col] = df[col].astype('category')

float_cols = df.select_dtypes(include='float64').columns
df[float_cols] = df[float_cols].astype('float32')

int_cols = df.select_dtypes(include='int64').columns
df[int_cols] = df[int_cols].astype('int32')


df.info()

# ✅ [1] 피처 구성 및 전처리

# 🎯 타겟 변수 '혼잡도' 제거하여 X 생성
X = df.drop(['혼잡도', '시간', '역번호'], axis=1)


# 🔤 범주형 변수 원-핫 인코딩 (첫 번째 카테고리는 drop)
o_h_e_x = pd.get_dummies(X, columns=['역명', '상하구분', 'AWS지점코드'], drop_first=True)

# 🔢 X 정규화 (0~1 범위로 스케일링)
mm = MinMaxScaler()
mm_x = mm.fit_transform(o_h_e_x)

# 🎯 y 정규화 (표준화: 평균 0, 표준편차 1)
y = np.log1p(df[['혼잡도']])
ss = StandardScaler()
ss_y = ss.fit_transform(y)

# ✅ [2] 시계열 학습/검증 데이터 분할

# 전체의 80%를 학습용, 나머지 20%를 검증용으로 사용
split_index = int(len(mm_x) * 0.8)

train_x = mm_x[:split_index]
train_y = ss_y[:split_index]

val_x = mm_x[split_index:]
val_y = ss_y[split_index:]

# ✅ [3] 시계열 데이터 생성

# 과거 24시간 데이터를 기반으로 예측
sequence_length = 24
batch_size = 128

# TimeseriesGenerator를 통해 시계열 학습/검증셋 생성
train_gen = TimeseriesGenerator(train_x, train_y, length=sequence_length, batch_size=batch_size)
val_gen   = TimeseriesGenerator(val_x, val_y, length=sequence_length, batch_size=batch_size)

# ✅ [4] LSTM 모델 정의 및 학습

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
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5)

# 모델 학습
history = model.fit(train_gen, validation_data=val_gen,
                    epochs=10, callbacks=[early_stop, reduce_lr])

# ✅ [5] 예측 및 평가

# 검증셋에 대해 예측 수행
pred_scaled = model.predict(val_gen)

# 정규화 해제 (역정규화) → 실제 혼잡도 단위로 복원
pred = ss.inverse_transform(pred_scaled)
pred = np.expm1(pred)

# 실제 y도 시계열 슬라이싱 기준에 맞게 자르기
true = ss.inverse_transform(val_y[sequence_length:])
true = np.expm1(true)

# 평가 지표 계산
rmse = mean_squared_error(true, pred, squared=False)  # RMSE
r2 = r2_score(true, pred)                             # R² Score

# 결과 출력
print(f"✅ 검증 RMSE: {rmse:.4f}")
print(f"✅ R² Score: {r2:.4f}")






































