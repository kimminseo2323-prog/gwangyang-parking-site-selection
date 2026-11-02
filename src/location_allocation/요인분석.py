###########################################################
# 과 제 명: A10_광양시_주차장입지분석
# 파 일 명: 3.요인분석
# 작 성 자: 김민서
###########################################################

import os, importlib, warnings;
warnings.filterwarnings(action='ignore')
from config import config_A10 as config;
importlib.reload(config)
from utils import functions_utils as fu;
importlib.reload(fu)
from utils import functions_log as fl;
importlib.reload(fl)
from utils import functions_lime as flime;
importlib.reload(flime)
from utils import functions_utils_db as fud;
importlib.reload(fud)
# ==================================================================================================
# [0-1] log 기록 정의
log = fl.Logger(config.PATH_LOG, config.LOG_FILE_NM);
log.set_log_level("info")
log.info("0-1 log 기록 정의")

# ==================================================================================================
# [0-2] 실행 변수 정의
log.info("0.2.실행 변수 정의")
config_info = fu.f_get_config_info(config.config_run, config.ENV_DEV)

# ==================================================================================================
# [0-3] 실행 변수
log.info("0.3.실행 변수 추가");
fu.f_memory_check()
# 전역 변수 선언 부분
# config_info["TODAY"]           = "20230630"

# [분석 처리 시작]
fl.f_log_line("1.전처리_현황.py")

# 1.실행 변수 정의
log.info("1.실행 변수 정의")
path_base = "2024-06-13"
path_base2 = "2024-08-27"
path_data_out = os.path.join(config_info["PATH_DATA_COLLECT"], "01.외부")
path_data_in = os.path.join(config_info["PATH_DATA_COLLECT"], "02.내부",path_base)
path_data_in2 = os.path.join(config_info["PATH_DATA_COLLECT"], "02.내부",path_base2)
path_data_analyze = os.path.join(config_info["PATH_DATA_ANALYZE"])
path_data_output = os.path.join(config_info["PATH_DATA_OUTPUT"])
path_database = os.path.join(config_info["PATH_DATA_OUTPUT_DATEBASE"], path_base)
path_pretreat = os.path.join(config_info["PATH_DATA_OUTPUT_PRETREAT"])
path_middle = os.path.join(config_info["PATH_DATA_OUTPUT_MIDDLE"])
path_result = os.path.join(config_info["PATH_DATA_OUTPUT_RESULT"])
PATH_RESULT3 = os.path.join(config_info["PATH_RESULT"],'QGIS_DATA')
###################################################################################################
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

###################################################################################################
'''
factor_result_geo = gpd.read_file(PATH_RESULT3 + '//영향인자//격자_최종영향인자SET.gpkg', driver='GPKG')
factor_result = pd.read_pickle(path_result + '//영향인자//factor_result.pkl')
#factor_result.info()
#factor_result.columns

#범주형 변수를 수치형으로 변환
var = factor_result.copy()
columns_to_convert = [col for col in var.columns if col.endswith('여부')]
var[columns_to_convert] = var[columns_to_convert].applymap(lambda x: 1 if x == 'O' else 0)
var[columns_to_convert].value_counts()
var.info()
var.to_pickle(path_result + '//영향인자//factor_result2.pkl')
'''

#로지스틱 회귀모델
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

factor_result3 = pd.read_pickle(path_result + '//영향인자//factor_result2.pkl')
var = factor_result3.copy()
var_list = ['공영주차장_주차면수_300M_여부', '유동인구_2시/시간평균', '버스정류장_300M',
            '공원_300M_여부', '유동인구_18시/시간평균', '최근접_주민센터직선거리',
            '버스정류장_300M_여부', '버스정류장_300M', '전체주요시설_300M',
            '복합_총주차면수/거주인구', '공원_300M', '버스정류장_300M']

# 클래스 분포 확인
print(var['공영주차장_여부'].value_counts())
class_0 = var[var['공영주차장_여부'] == 0]
class_1 = var[var['공영주차장_여부'] == 1]
# under-sampling: class 1의 개수에 맞춰 class 0에서 랜덤 샘플링 (4배수)
class_0_sampled = class_0.sample(n=len(class_1) * 4, random_state=42) #585
balanced_df = pd.concat([class_0_sampled, class_1])
balanced_df = balanced_df.replace([np.inf, -np.inf], np.nan)  # 무한대를 NaN으로 변환
balanced_df = balanced_df.fillna(0.0)  # NaN 값 제거

X = balanced_df[var_list]
y = balanced_df['공영주차장_여부']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()  # 데이터 표준화
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)  # 예측
accuracy = accuracy_score(y_test, y_pred)  # 정확도 평가
print("모델 정확도:", accuracy) #모델 정확도: 0.9658119658119658

# 샘플링된 데이터에 대해 점수 산출
balanced_df['Score'] = model.predict_proba(scaler.transform(balanced_df[X.columns]))[:, 1]  # 입지 점수 산출
#balanced_df.describe()
# 원본 데이터에 대한 점수 산출
factor_result4 = factor_result3.replace([np.inf, -np.inf], np.nan)  # 무한대를 NaN으로 변환
factor_result4 = factor_result4.fillna(0.0)  # NaN 값 제거
factor_result4_scaled = scaler.transform(factor_result4[var_list])  # 원본 데이터 표준화
factor_result4['Score'] = model.predict_proba(factor_result4_scaled)[:, 1]  # 원본 데이터의 Score 추가

sorted_df = factor_result4.sort_values(by='Score', ascending=False)


bins = [i / 10 for i in range(11)]  # [0.0, 0.1, ..., 1.0] # 점수를 0.1 단위로 나누기
labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 1)]
sorted_df['Score_Bin'] = pd.cut(sorted_df['Score'], bins=bins, labels=labels, include_lowest=True) # 각 Score가 속한 구간을 구분
sorted_df.groupby('Score_Bin').size()
# 각 구간에 대한 타겟 비율 계산
target_ratio = sorted_df.groupby('Score_Bin')['공영주차장_여부'].mean()
print(target_ratio)

# 1의 비율이 낮은 구간 찾기
low_ratio_bins = target_ratio[target_ratio <=  0.1].index.tolist()
print("1의 비율이 낮은 구간:", low_ratio_bins)

# 1의 비율이 낮은 구간 제거
filtered_df = sorted_df[~sorted_df['Score_Bin'].isin(low_ratio_bins)]
filtered_df.columns
filtered_df2 = filtered_df[['gid', '공영주차장_여부','Score', 'geometry']]
filtered_df2.to_file('C://2024_project//project//A10_광양시_주차장입지분석//P2_Module//1.분석모델//입지배분모델//입지_데이터//분석대상격자_socre(0.6이상구간).gpkg', driver='GPKG')

target_grid = filtered_df[['gid','geometry']]

########################################################################################################################
r_parking = gpd.read_file(PATH_RESULT3 + '//영향인자//주차장//주차장_격자.gpkg', driver='GPKG')
r_parking['공영주차장_여부'] = r_parking['공영주차장_여부'].replace({'O': 1, 'X': 0})
r_parking_r = r_parking[['gid', '행정동명','공영주차장_여부', '총주차면수', 'geometry']]
flowpop_time = pd.read_pickle('C://2024_project//project//A10_광양시_주차장입지분석//P2_Module//1.분석모델//입지배분모델//입지_데이터//9.old//유동인구//flowpop_time.pkl')
flowpop_time_r = flowpop_time[['gid','TMST_12', 'TMST_20']]


data_1 = pd.merge(r_parking_r,flowpop_time_r, on='gid', how='left')
data_1.fillna(0.0, inplace=True)
data_1 = data_1[['gid', '행정동명','공영주차장_여부', '총주차면수', 'TMST_12', 'TMST_20']]
data_1.rename(columns={'TMST_12' : '유동인구12시','TMST_20':'유동인구20시'}, inplace=True)

data_1.to_pickle('C://2024_project//project//A10_광양시_주차장입지분석//P2_Module//1.분석모델//입지배분모델//입지_데이터//전체격자_유동인구.pkl')

total_grid = factor_result3.copy()
total_grid = total_grid[['gid', '거주인구','등록차량수','geometry']]
total_grid = pd.merge(total_grid,data_1, on='gid', how='left')
total_grid2 = total_grid[['gid', '행정동명','공영주차장_여부', '유동인구12시','유동인구20시', '거주인구', '등록차량수', '총주차면수', 'geometry']]
total_grid2['거주인구/전체주차장주차면수'] = (total_grid2['거주인구']/total_grid2['총주차면수']).round(2)
total_grid2['유동인구12시/전체주차장주차면수'] = (total_grid2['유동인구12시']/total_grid2['총주차면수']).round(2)
total_grid2['유동인구20시/전체주차장주차면수'] = (total_grid2['유동인구20시']/total_grid2['총주차면수']).round(2)
total_grid2['등록차량수/전체주차장주차면수'] = (total_grid2['등록차량수']/total_grid2['총주차면수']).round(2)
total_grid2.fillna(0.0, inplace=True)
total_grid2 = total_grid2[['gid', '행정동명','공영주차장_여부', '유동인구12시', '유동인구20시', '거주인구', '등록차량수', '총주차면수',
                           '거주인구/전체주차장주차면수', '유동인구12시/전체주차장주차면수', '유동인구20시/전체주차장주차면수','등록차량수/전체주차장주차면수']]

total_grid2 = total_grid2.replace([np.inf, -np.inf], np.nan)  # 무한대를 NaN으로 변환
total_grid2 = total_grid2.fillna(0.0)  #
total_grid_gy = total_grid2[total_grid2['행정동명'] == '광양읍']
total_grid_jm = total_grid2[total_grid2['행정동명'] == '중마동']

total_grid_gy.to_csv('C://2024_project//project//A10_광양시_주차장입지분석//P2_Module//1.분석모델//입지배분모델//입지_데이터//최종영향인자_광양읍.csv', encoding='cp949')
total_grid_jm.to_csv('C://2024_project//project//A10_광양시_주차장입지분석//P2_Module//1.분석모델//입지배분모델//입지_데이터//최종영향인자_중마동.csv', encoding='cp949')

########################################################################################################################

target_grid = pd.merge(target_grid, total_grid2[['gid','행정동명']], on='gid', how='left')
target_grid = target_grid[['gid', '행정동명', 'geometry']]
target_grid
target_grid_gy = target_grid[target_grid['행정동명'] == '광양읍']
target_grid_jm = target_grid[target_grid['행정동명'] == '중마동']

target_grid_gy.to_file('C://2024_project//project//A10_광양시_주차장입지분석//P2_Module//1.분석모델//입지배분모델//입지_데이터//분석대상격자_광양읍.gpkg', driver='GPKG')
target_grid_jm.to_file('C://2024_project//project//A10_광양시_주차장입지분석//P2_Module//1.분석모델//입지배분모델//입지_데이터//분석대상격자_중마동.gpkg', driver='GPKG')




















'''
factor_result2 = pd.read_pickle(path_result + '//영향인자//factor_result2.pkl')
factor_result2


var_list = ['거주인구', '거주인구_300M', '거주인구_70대이상/거주인구',
       '거주인구_70대이상_300M/거주인구', '거주인구_거주인구/거주인구_300M', '유동인구', '유동인구_300M',
       '유동인구_유동인구/유동인구_300M', '유동인구_2시/시간평균', '유동인구_18시/시간평균', '등록차량수',
       '등록차량수_300M', '등록차량수_등록차량/등록차량_300M', '부설주차장_주차면수', '부설주차장_주차면수_300M',
       '부설주차장/부설주차장_300M', '주택주차장_주차면수', '주택주차장_주차면수_300M', '주택주차장/주택주차장_300M',
       '주택주차장/부설주차장', '주택주차장_300M/부설주차장_300M', '공영주차장_주차면수_300M_여부',
       '공영주차장_주차면수_300M', '최근접_공영주차장직선거리', '공원', '버스정류장', '은행/저축기관', '의원/병원',
       '주민센터', '버스정류장_여부', '공원_여부', '의원/병원_여부', '은행/저축기관_여부', '주민센터_여부',
       '버스정류장_300M_여부', '공원_300M_여부', '은행/저축기관_300M_여부', '의원/병원_300M_여부',
       '주민센터_300M_여부', '버스정류장_300M', '공원_300M', '은행/저축기관_300M', '의원/병원_300M',
       '주민센터_300M', '전체주요시설_300M', '최근접_주민센터직선거리', '복합_유동인구/거주인구',
       '복합_등록차량수/거주인구', '복합_등록차량수/유동인구', '복합_총주차면수/거주인구', '복합_총주차면수/유동인구',
       '복합_총주차면수/등록차량수', '복합_유동인구/거주인구_300M', '복합_등록차량수/거주인구_300M',
       '복합_등록차량수/유동인구_300M', '복합_총주차면수/거주인구_300M', '복합_총주차면수/유동인구_300M',
       '복합_총주차면수/등록차량수_300M' ]
var = factor_result2.copy()
var['공영주차장_여부'].value_counts()
공영주차장_여부
0    6900
1     117
Name: count, dtype: int64
'''

'''
# NaN 값을 포함한 행이 있는지 확인
print(var.isnull().sum())
##### 영향인자 로지스틱 회귀분석 : Under-sampling
# 공영주차장_여부가 0인 데이터에서 under-sampling 진행
# 공영주차장이 있는 데이터와 없는 데이터 분리
park_1 = var[var["공영주차장_여부"] == 1]
park_0 = var[var["공영주차장_여부"] == 0]

# 공영주차장 있는 데이터 수
n_park_1 = len(park_1)
# 공영주차장 없는 데이터에서 가능한 최대한의 샘플을 추출(랜덤추출법)
park_0_sample = park_0.sample(n=n_park_1, random_state=42)
# 샘플링된 데이터와 공영주차장 있는 데이터 결합하여 최종 샘플링 데이터셋 생성
samdb_0 = pd.concat([park_0_sample, park_1], axis=0)
# samdb에 있는 무한대값을 NaN으로 변환
samdb_0.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
# NaN 값을 0으로 대체
samdb_0.fillna(0, inplace=True)
# 최종 샘플데이터셋 출력
samdb = samdb_0

# 데이터셋에서 target과 features 분리
target = samdb[['공영주차장_여부']]
# StandardScaler를 사용하여 변수들을 표준화
scaler = StandardScaler()

for var in var_list:
       features = samdb[[var]]
       # feature scaling

       train_features = scaler.fit_transform(features)

       # 로지스틱 회귀 모델 생성 및 학습
       sm_model = sm.Logit(target, train_features).fit()

       # 결과 출력
       print(f"변수: {var}")
       print(sm_model.summary())


# 다운샘플링 결과로 변수별 r2
y = samdb['공영주차장_여부']

# var_list에 해당되는 컬럼들만 독립 변수로 설정
independent_vars = samdb[var_list]

# 각 변수별로 R² 값을 저장할 딕셔너리
r2_values = {}

# 회귀 모델 객체 생성
model = LinearRegression()

# var_list의 각 변수에 대해 회귀 분석을 진행하고 R² 값 계산
for column in independent_vars.columns:
       # 해당 변수를 독립 변수로 설정 (2D 배열로 변환)
       X = independent_vars[[column]]

       # 모델 학습
       model.fit(X, y)

       # 예측값 계산
       y_pred = model.predict(X)

       # R² 값 계산
       r2 = r2_score(y, y_pred)

       # R² 값을 딕셔너리에 저장
       r2_values[column] = r2

# R² 값을 정렬하여 데이터프레임으로 변환
rslt_score = {var: r2 for var, r2 in sorted(r2_values.items(), key=lambda item: item[1], reverse=True)}
rslt_out = pd.DataFrame(list(rslt_score.items()), columns=['Variable', 'R²'])
# 결과를 Excel 파일로 저장
rslt_out.to_excel('C://2024_project//project//A10_광양시_주차장입지분석//P2_Module//1.분석모델//입지배분모델//입지_데이터//9.old//로지스틱예측결과//로지스틱예측결과(r2).xlsx', index=False)


#######################################################################################################################
'''