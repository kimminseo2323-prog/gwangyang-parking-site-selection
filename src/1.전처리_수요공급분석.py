###########################################################
# 과 제 명: A10_광양시_주차장입지분석
# 파 일 명: 1.데이터전처리
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
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler
###################################################################################################

##데이터 불러오기
# 주차장 현황
#parking = pd.read_excel(path_data_in + '//주차장최적지 분석(데이터센터)기초자료//★광양시 주차장 현황(외부용) (1).xlsx')
car = pd.read_excel(path_data_in + '//광양시 등록차량 정보.xlsx')
dup_car = car.drop_duplicates()
dup_car.columns
dup_car.groupby('용도구분')['차량등록번호'].size().reset_index()

########################################################################################################################
# 행정경계 추출
HJD = gpd.read_file(path_data_out + '//센서스행정경계_202402//BND_ADM_DONG_PG//BND_ADM_DONG_PG.shp',encoding='cp949')
HJD_gw = (HJD[HJD['ADM_CD'].str.startswith('3606')]).reset_index(drop=True)
HJD_gw = HJD_gw[[ 'ADM_NM', 'ADM_CD', 'geometry']]
HJD_gw.to_file(PATH_RESULT3 + '//행정경계//행정경계.gpkg', driver='GPKG')

#100m 격자 추출
grid_100 = gpd.read_file(path_data_out + '//100m격자//nlsp_021001001.shp',encoding='utf-8')
grid_100 = gpd.GeoDataFrame(grid_100, geometry='geometry', crs='EPSG:5186')
grid_100 = grid_100.to_crs(epsg=5186)
grid_100_hjd = gpd.read_file(PATH_RESULT3 + '//주차장수요//격자_행정동매핑.gpkg', driver='GPKG')
########################################################################################################################
#주차장 데이터
parking1 = pd.read_excel(path_data_in + '//주차장최적지 분석(데이터센터)기초자료//광양시_주차장현황_수정.xlsx',sheet_name='수작업정리')      #공영주차장
parking3_1 = pd.read_excel(path_data_in + '//주차장최적지 분석(데이터센터)기초자료//건축물대장 현황(3169)//일반건축물 전체.xlsx',skiprows=8) #일반표제부
parking3_2 = pd.read_excel(path_data_in + '//주차장최적지 분석(데이터센터)기초자료//건축물대장 현황(3169)//총괄표제부 전체.xlsx',skiprows=7) #총괄표제부

#공영주차장
parking1.rename(columns={'구   분':'ADM_NM'},inplace=True)
parking1_geo = pd.merge(parking1,HJD_gw,on='ADM_NM',how='left')

parking1_geo['공영주차장'] = parking1_geo[[ '노상', '노외']].sum(axis=1)
total_parking = parking1_geo['공영주차장'].sum()
parking1_geo['공영주차장 비율'] =(parking1_geo['공영주차장'] / total_parking * 100).round(2)
parking1_geo = parking1_geo[['ADM_NM','ADM_CD','공영주차장','공영주차장 비율', '노상', '노외', '노상_유료', '노상_무료', '노외_유료', '노외_무료', 'geometry']]
parking1_geo = gpd.GeoDataFrame(parking1_geo, geometry='geometry')


parking1_geo.to_file(PATH_RESULT3 + '//주차장공급//공영주차장.gpkg', driver='GPKG')
parking1_geo.to_excel(path_result + '//주차장공급//공영주차장.xlsx',index=False)
########################################################################################################################
#건축물대장 주차장
## 건축물 대장 표제부 검증
docu_head = parking3_2.copy() #2873 ,총괄표제부
docu_unit = parking3_1.copy() #30314 ,일반표제부
docu_head.rename(columns={'시군구코드':'시군구','대지위치주소':'대지위치'}, inplace=True)
docu_unit.rename(columns={'시군구코드':'시군구','대지위치주소':'대지위치'}, inplace=True)
#docu_unit['총주차수'] = docu_unit[['옥내기계식대수', '옥외기계식대수', '옥내자주식대수', '옥외자주식대수']].sum(axis=1)


# 1. 총괄표제부 중복제거
# 1.1 총괄표제부 중복현환
a10 = docu_head.groupby(['시군구', '대지위치']).size().reset_index()
a11 = docu_head.groupby(['시군구', '대지위치'])['총주차수'].sum().reset_index()
a13 = pd.merge(a10, a11, how='left', on=['시군구', '대지위치'])
a15 = docu_head.groupby(['시군구', '대지위치'])['총주차수'].max().reset_index()
a15.columns = ['시군구', '대지위치', '총괄주차장'] #2,868

docu_unit.info()
docu_unit['총 주차대수'].fillna(value=0.0, inplace=True)
# 2. 건축물대장 표제부 중복현황
docu_unit['주차수합산'] = docu_unit[['옥내기계식대수','옥외기계식대수','옥내자주식대수','옥외자주식대수']].sum(axis=1)
docu_unit['총 주차대수'].fillna(value=0.0, inplace=True)
docu_unit['총주차수'] = np.where( docu_unit['주차수합산']> docu_unit['총 주차대수'],docu_unit['주차수합산'],docu_unit['총 주차대수'])


a20 = docu_unit.groupby(['시군구', '대지위치']).size().reset_index()
a20.columns = ['시군구', '대지위치', '중복건수']
a21 = docu_unit.groupby(['시군구', '대지위치'])['총주차수'].sum().reset_index()
a211 = docu_unit.groupby(['시군구', '대지위치'])['총주차수'].max().reset_index()
a211.columns = ['시군구', '대지위치', '최대주차장']
a23 = pd.merge(a20, a21, how='left', on=['시군구', '대지위치'])
a23 = pd.merge(a23, a211, how='left', on=['시군구', '대지위치']) #23,216
a23.columns = ['시군구', '대지위치', '중복건수', '총주차수합계', '최대주차장']
a231 = a23[(a23['총주차수합계'] > 0) & (a23['중복건수'] > 1)] #657
a23
# 3. 건축물대장 표제부에 총괄표제부 정보 추가 : 23,216
a30 = pd.merge(a23, a15, how='left', on=['시군구', '대지위치'])
a30 = a30.fillna(0)

# 3.1==> 총괄표제부주차장 >0 인경우 최종주차장수산출
f01 = a30[a30['총괄주차장'] > 0]
f01['최종주차장수'] = f01['총괄주차장']  # 634건
f01 = f01[['시군구', '대지위치', '최종주차장수']]

# 3.2 총괄표제부주차장수 >= 0   # 22,582건
a31 = a30[a30['총괄주차장'] <= 0]
a31['rt'] = a31['총주차수합계'] / a31['최대주차장']
a31['dup_fix'] = np.where(a31['rt'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 1, 0)

# 3.3 건축물대장주차장수 합계가 최댓값과 같거나, 정율배
f02 = a31[a31['dup_fix'] == 1]
f02['최종주차장수'] = f02['최대주차장']  # 3,986건
f02 = f02[['시군구', '대지위치', '최종주차장수']]

# 3.3 건축물대장주차장수 합계가 정율이 아닌 경우 : 18,596
a32 = a31[a31['dup_fix'] == 0]

# 3.4 건축물대장주차장수 합계가 0인 경우  : 18,553
f03 = a32[a32['총주차수합계'] == 0]
f03['최종주차장수'] = f03['총주차수합계']  # 6148건
f03 = f03[['시군구', '대지위치', '최종주차장수']]

a33 = a32[a32['총주차수합계'] > 0]  # 43건

## 최빈값 산출하여 최빈값과 최댓값 비교 # 43건
a34 = pd.merge(a33, docu_unit, how='left', on=['시군구', '대지위치'])
a35 = a34.groupby(['시군구', '대지위치', '중복건수', '총주차수합계', '최대주차장', 'rt'])['총주차수'].agg(
    lambda x: x.mode().iloc[0]).reset_index()
a35.columns = ['시군구', '대지위치', '중복건수', '총주차수합계', '최대주차장', 'rt', '최빈주차장']
a35['rt_최대최빈'] = a35['최빈주차장'] / a35['최대주차장']

## 최빈값이 최댓값의 95% 이상인 경우 : 최빈값 적용
f04 = a35[a35['rt_최대최빈'] >= 0.95]
f04['최종주차장수'] = f04['최빈주차장']  # 3건
f04 = f04[['시군구', '대지위치', '최종주차장수']]

## 최빈값이 최댓값의 95% 미만인 경우 : 합계값 적용
f05 = a35[a35['rt_최대최빈'] < 0.95]
f05['최종주차장수'] = f05['총주차수합계']  # 40건
f05 = f05[['시군구', '대지위치', '최종주차장수']]

##############################################
##7. 총괄표제부에는 있는데, 표제부 데이터가 없는 경우
a40 = pd.merge(a15, a23, how='left', on=['시군구', '대지위치'])
a40['noinfo'] = np.where(a40['총주차수합계'].isnull(), 1, 0)
a40
# 룰생성 74건 : 총괄표제부에 있는데, 표제부 데이터가 없는 경우 총괄표제부 주차장수 추가
f06 = a40[(a40['noinfo'] == 1) & (a40['총괄주차장'] > 0)]
f06[f06['대지위치'].str.contains('금호동', na=False)]
f06['최종주차장수'] = f06['총괄주차장']
f06 = f06[['시군구', '대지위치', '최종주차장수']]
f06
## 최종 주차장수 통합
parking_cnt = pd.concat([f01, f02, f03, f04, f05, f06], axis=0)
parking_cnt['최종주차장수'].sum() # 414012.0, 23290 rows
# '읍', '면', '동'으로 끝나는 부분 추출하는 함수
def extract_adm_nm(address):
    # 공백을 기준으로 주소를 나눈 후, 순서대로 부분을 검사
    parts = address.split()
    for part in parts:
        if part.endswith(('읍', '면', '동')):
            return part
    return ''

# 'ADM_NM' 컬럼 생성
parking_cnt['ADM_NM'] = parking_cnt['대지위치'].apply(extract_adm_nm)


# 수작업 수정 (도이동,마동,성황동,중군동,중동,황금동,황길동)

parking_cnt.loc[parking_cnt['ADM_NM'] == '성황동']
parking_cnt.loc[parking_cnt['ADM_NM'] == '도이동', 'ADM_NM'] = '골약동'
parking_cnt.loc[parking_cnt['ADM_NM'] == '마동', 'ADM_NM'] = '중마동'
parking_cnt.loc[parking_cnt['ADM_NM'] == '성황동', 'ADM_NM'] = '골약동'
parking_cnt.loc[parking_cnt['ADM_NM'] == '중군동', 'ADM_NM'] = '골약동'
parking_cnt.loc[parking_cnt['ADM_NM'] == '중동', 'ADM_NM'] = '중마동'
parking_cnt.loc[parking_cnt['ADM_NM'] == '황금동', 'ADM_NM'] = '골약동'
parking_cnt.loc[parking_cnt['ADM_NM'] == '황길동', 'ADM_NM'] = '골약동'


parking_cnt.to_excel(path_middle + '//건축물대장대상목록.xlsx', index=False)

#특정지역 지오코딩1
parking_cnt = pd.read_excel(path_middle + '//건축물대장대상목록.xlsx')
parking_cnt_gold = parking_cnt[parking_cnt['ADM_NM'] == '금호동']
parking_cnt_gold.to_excel(path_middle + '//건축물대장대상목록_금호동.xlsx', index=False)
#금호동 지오코딩
parking_cnt_gold_geo = gpd.read_file(path_middle + '//특정지역지오코딩//지오코딩완료//건축물대장대상목록_금호동_WGS.shp',encoding='cp949')
parking_cnt_gold_geo2 = gpd.GeoDataFrame(parking_cnt_gold_geo, geometry='geometry',crs='epsg:4326')
parking_cnt_gold_geo2 = parking_cnt_gold_geo2.to_crs(epsg=5186)
parking_cnt_gold_geo2['최종주차장'] = parking_cnt_gold_geo2['최종주차장'].astype(int)
parking_cnt_gold_geo2 = parking_cnt_gold_geo2.sort_values(by='최종주차장', ascending=False)
parking_cnt_gold_geo2.to_file(PATH_RESULT3 + '//주차장공급//특정지역시각화(ppt자료)//건축물대장_금호동주차장.gpkg', driver='GPKG')

#pkl 변환
parking_cnt_1 = gpd.read_file(path_middle + '//특정지역지오코딩//지오코딩완료//건축물대장대상목록_광양읍_WGS.shp',encoding='cp949')
parking_cnt_2 = gpd.read_file(path_middle + '//특정지역지오코딩//지오코딩완료//건축물대장대상목록_중마동_WGS.shp',encoding='cp949')
parking_cnt_1.to_pickle(path_middle + '//특정지역지오코딩//건축물대장대상목록_광양읍.pkl')
parking_cnt_2.to_pickle(path_middle + '//특정지역지오코딩//건축물대장대상목록_중마동.pkl')

#행정동 매핑
parking_3_3 =  parking_cnt.groupby('ADM_NM')['최종주차장수'].sum().reset_index()
total_parking = parking_3_3['최종주차장수'].sum()
parking_3_3['부설주차장_비율'] =(parking_3_3['최종주차장수'] / total_parking * 100).round(2)
parking_3_3.rename(columns={'최종주차장수':'부설주차장'},inplace=True)
parking_3_hjd = pd.merge(parking_3_3,HJD_gw,on='ADM_NM',how='left')
parking_3_geo = gpd.GeoDataFrame(parking_3_hjd, geometry='geometry')
parking_3_geo.to_file(PATH_RESULT3 + '//주차장공급//건축물대장주차장.gpkg', driver='GPKG')
parking_3_3.to_excel(path_result + '//주차장공급//건축물대장주차장.xlsx',index=False)


#최종 주차장 데이터셋 생성
parking1_geo = pd.read_excel(path_result + '//주차장공급//공영주차장.xlsx')
parking_3_3 = pd.read_excel(path_result + '//주차장공급//건축물대장주차장.xlsx')
parking1_3 = parking1_geo[['ADM_NM', 'ADM_CD', '공영주차장']]
parking_result = pd.merge(parking1_3,parking_3_3,on='ADM_NM',how='left')
parking_result['총주차수'] = parking_result['공영주차장'] + parking_result['부설주차장']
total_parking = parking_result['총주차수'].sum()
parking_result['비율'] =(parking_result['총주차수'] / total_parking * 100).round(2)
parking_result2 = pd.merge(parking_result,HJD_gw,on='ADM_NM',how='left')
parking_result2 = parking_result2[['ADM_NM', '총주차수', '공영주차장', '부설주차장', '비율', 'geometry']]
parking_result = gpd.GeoDataFrame(parking_result2, geometry='geometry')
parking_result.to_file(PATH_RESULT3 + '//주차장공급//주차장공급현황.gpkg', driver='GPKG')
parking_result.to_excel(path_result + '//주차장공급//주차장공급현황.xlsx',index=False)



#차량등록대수
car = pd.read_excel(path_data_in + '//광양시 등록차량 정보.xlsx')
pre_car = car.copy()
pre_car = pre_car.drop_duplicates().reset_index(drop=True)

pre_car['행정동'].value_counts()
#수작업 전처리
pre_car.loc[pre_car['행정동'] == '전라남도 광양시 황금동', '행정동'] = '전라남도 광양시 골약동'
pre_car.loc[pre_car['행정동'] == '전라남도 광양시', '행정동'] = '전라남도 광양시 광영동'
pre_car['행정동'].value_counts() #확인
pre_car.to_excel(path_data_analyze+ '//등록차량정보//등록차량정보.xlsx',index=False)
#차량등록대수 지오코딩 파일생성
total_rows = len(pre_car) #지오코딩을 위하여 10000개씩 끊어서 저장
chunk_size = 9999 # 나누고자 하는 행의 크기
for i in range(0, total_rows, chunk_size):

    chunk_df = pre_car.iloc[i:i + chunk_size]
    path = path_data_analyze + '//등록차량정보//지오코딩전//'
    chunk_df.to_csv(path + f'pre_car_{i // chunk_size + 1}.csv',encoding='cp949', index=False)
print("데이터가 성공적으로 나누어졌습니다.")

#지오코딩 후 통합 및 좌표계 변환


#차량등록대수_행정동별 현황
pre_car['행정동'] = pre_car['행정동'].str.split().str[2]
pre_car_g = pre_car.groupby(['행정동','용도구분']).size().reset_index(name='등록차량대수')
pre_car_gp = pre_car_g.pivot(index='행정동', columns='용도구분', values='등록차량대수').fillna(0).reset_index()
pre_car_gp_geo = pd.merge(pre_car_gp,HJD_gw,left_on='행정동',right_on='ADM_NM',how='left')
pre_car_gp_geo['총차량수'] = pre_car_gp_geo[[ '관용', '영업용', '자가용']].sum(axis=1)
pre_car_gp_geo = pre_car_gp_geo[['행정동','ADM_CD', '총차량수','관용', '영업용', '자가용',  'geometry']]
pre_car_gp_geo = gpd.GeoDataFrame(pre_car_gp_geo, geometry='geometry')
pre_car_gp_geo['총차량수'].sum()
pre_car_gp_geo.to_file(PATH_RESULT3 + '//주차장수요//등록차량대수.gpkg', driver='GPKG')

pre_car_gp_2 = pre_car_gp_geo[['행정동','ADM_CD', '총차량수','관용', '영업용', '자가용']]
pre_car_gp_2['관용비율'] = ((pre_car_gp_2['관용']/pre_car_gp_2['총차량수'])*100).round(3)
pre_car_gp_2['영업용비율'] = ((pre_car_gp_2['영업용']/pre_car_gp_2['총차량수'])*100).round(2)
pre_car_gp_2['자가용비율'] = ((pre_car_gp_2['자가용']/pre_car_gp_2['총차량수'])*100).round(2)
total_car = pre_car_gp_2['총차량수'].sum()
pre_car_gp_2['차량비율'] = ((pre_car_gp_2['총차량수']/total_car)*100).round(1)
pre_car_gp_2.to_excel(path_result + '//주차장수요//등록차량대수.xlsx',index=False)

########################################################################################################################
#거주인구
h_pop = pd.read_excel(path_data_out + '//광양시_주민등록인구_202407.xlsx')
h_pop_kosis = pd.read_excel(path_data_out + '//행정구역_읍면동_별_5세별_주민등록인구_2011년__20240826095702.xlsx')
h_pop.rename(columns={'구분':'ADM_NM'},inplace=True)

h_pop_geo = pd.merge(h_pop,HJD_gw,on='ADM_NM',how='left')
h_pop_geo = h_pop_geo[['ADM_NM', 'ADM_CD', '세대수', '주민등록인구수', '주민등록인구_남', '주민등록인구_여','geometry']]
h_pop_geo['주민등록인구수'].sum()
h_pop_geo = gpd.GeoDataFrame(h_pop_geo, geometry='geometry')
h_pop_geo.to_file(PATH_RESULT3 + '//주차장수요//거주인구.gpkg', driver='GPKG')
h_pop_2 = h_pop_geo[['ADM_NM', 'ADM_CD', '세대수', '주민등록인구수', '주민등록인구_남', '주민등록인구_여']]
h_pop_2['남자비율'] = ((h_pop_2['주민등록인구_남']/h_pop_2['주민등록인구수']).round(1))*100
h_pop_2.to_excel(path_result + '//주차장수요//거주인구.xlsx',index=False)

#차량등록대수 대비 주차장 확보율
parking_result.rename(columns={'ADM_NM':'행정동'},inplace=True)
result_1 = pd.merge(pre_car_gp_geo,parking_result[['행정동','총주차수']],on='행정동',how='left')
result_1 = result_1[['행정동','총차량수','총주차수','geometry']]
result_1['주차장확보율'] = ((result_1['총주차수']/result_1['총차량수'])*100).round(1)
result_1 = result_1[['행정동','주차장확보율','총차량수','총주차수','geometry']]
result_1 = gpd.GeoDataFrame(result_1, geometry='geometry')
result_1.to_file(PATH_RESULT3 + '//주차장수요//주차장확보율.gpkg', driver='GPKG')
result_1.to_excel(path_result + '//주차장수요//주차장확보율.xlsx',index=False)

#거주인구 대비 차량등록대수 비율
h_pop_3 = h_pop_2[['ADM_NM','주민등록인구수','주민등록인구_남','주민등록인구_여']]
h_pop_3.rename(columns={'ADM_NM':'행정동'},inplace=True)
result_2 = pd.merge(h_pop_3,result_1,on='행정동',how='left')
result_2 = result_2[['행정동', '주민등록인구수', '주민등록인구_남', '주민등록인구_여', '총차량수', '총주차수','geometry']]
result_2['주차장공급비율'] = ((result_2['총주차수'] / result_2['주민등록인구수'])*100).round(1)
result_2['차량대수비율'] = ((result_2['총차량수'] / result_2['주민등록인구수'])*100).round(1)
#result_2 = result_2[['행정동', '주민등록인구수', '주민등록인구_남', '주민등록인구_여', '총차량수', '총주차수','주차장공급비율', '차량대수비율', 'geometry']]

# 평균 비율 계산
mean_주차장공급비율 = result_2['주차장공급비율'].mean()
mean_차량대수비율 = result_2['차량대수비율'].mean()
# 평균 이하/이상의 구분 컬럼 추가
result_2['주차장공급비율_평균정도'] = result_2['주차장공급비율'].apply(lambda x: '이상' if x >= mean_주차장공급비율 else '이하')
result_2['차량대수비율_평균정도'] = result_2['차량대수비율'].apply(lambda x: '이상' if x >= mean_차량대수비율 else '이하')

# 결과 컬럼 정리 및 GeoDataFrame 생성
result_2 = result_2[['행정동', '주민등록인구수', '주민등록인구_남', '주민등록인구_여', '총차량수', '총주차수',
                     '주차장공급비율', '차량대수비율', '주차장공급비율_평균정도', '차량대수비율_평균정도', 'geometry']]
result_2 = gpd.GeoDataFrame(result_2, geometry='geometry')
result_2.to_file(PATH_RESULT3 + '//주차장수요//거주인구대비수요.gpkg', driver='GPKG')
result_2.to_excel(path_result + '//주차장수요//거주인구대비수요.xlsx',index=False)


#유동인구 분석
# 유동인구 데이터 합치기
# 특정 폴더 내의 모든 파일 리스트 가져오기
folder_path = path_data_in+'//광양_유동인구_2021~2022//성연령//'
file_list = os.listdir(folder_path)
combined_data_sex = pd.DataFrame()

# 각 CSV 파일 읽어들여 combined_data에 추가하기
for file_name in file_list:
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        # CSV 파일 읽기 (sep='|'로 설정)
        df = pd.read_csv(file_path, sep='|')
        # 읽어들인 데이터를 combined_data에 추가
        combined_data_sex = pd.concat([combined_data_sex, df], ignore_index=True)
combined_data_sex.to_pickle(path_data_analyze + '//유동인구//성연령별_유동인구.pkl')

folder_path = path_data_in+'//광양_유동인구_2021~2022//시간대//'
file_list = os.listdir(folder_path)
combined_data_time = pd.DataFrame()
# 각 CSV 파일 읽어들여 combined_data에 추가하기
for file_name in file_list:
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        # CSV 파일 읽기 (sep='|'로 설정)
        df = pd.read_csv(file_path, sep='|')
        # 읽어들인 데이터를 combined_data에 추가
        combined_data_time = pd.concat([combined_data_time, df], ignore_index=True)
combined_data_time.to_pickle(path_data_analyze + '//유동인구//시간대별_유동인구.pkl')

folder_path = path_data_in+'//광양_유동인구_2021~2022//요일//'
file_list = os.listdir(folder_path)
combined_data_day = pd.DataFrame()
# 각 CSV 파일 읽어들여 combined_data에 추가하기
for file_name in file_list:
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        # CSV 파일 읽기 (sep='|'로 설정)
        df = pd.read_csv(file_path, sep='|')
        # 읽어들인 데이터를 combined_data에 추가
        combined_data_day = pd.concat([combined_data_day, df], ignore_index=True)
combined_data_day.to_pickle(path_data_analyze + '//유동인구//요일별_유동인구.pkl')


sex_upop = pd.read_pickle(path_data_analyze + '//유동인구//성연령별_유동인구.pkl')
time_upop = pd.read_pickle(path_data_analyze + '//유동인구//시간대별_유동인구.pkl')
day_upop = pd.read_pickle(path_data_analyze + '//유동인구//요일별_유동인구.pkl')


sex_upop.columns
time_upop.columns
day_upop.columns

sex_upop['geometry'] = sex_upop.apply(lambda row: Point(row['X_COORD'], row['Y_COORD']), axis=1)
sex_upop_geo = gpd.GeoDataFrame(sex_upop, geometry='geometry',crs='epsg:5179')
sex_upop_geo = sex_upop_geo.to_crs(epsg=5186)
sex_upop_geo.to_file(PATH_RESULT3 +'//주차장수요/sex유동인구.gpkg', driver='GPKG')

time_upop['geometry'] = time_upop.apply(lambda row: Point(row['X_COORD'], row['Y_COORD']), axis=1)
time_upop_geo = gpd.GeoDataFrame(time_upop, geometry='geometry',crs='epsg:5179')
time_upop_geo = time_upop_geo.to_crs(epsg=5186)
time_upop_geo.to_file(PATH_RESULT3 +'//주차장수요//time유동인구.gpkg', driver='GPKG')

day_upop['geometry'] = day_upop.apply(lambda row: Point(row['X_COORD'], row['Y_COORD']), axis=1)
day_upop_geo = gpd.GeoDataFrame(day_upop, geometry='geometry',crs='epsg:5179')
day_upop_geo = day_upop_geo.to_crs(epsg=5186)
day_upop_geo.to_file(PATH_RESULT3 +'//주차장수요//day유동인구.gpkg', driver='GPKG')

#행정경계 밖에 존재하는 유동인구 포인트 처리방안 (제거할지, 가까운 행정동으로 배정할지 )
sex_upop_geo.columns
sex_upop_geo['total'] = sex_upop_geo[[ 'MAN_FLOW_POP_CNT_10G',
       'MAN_FLOW_POP_CNT_20G', 'MAN_FLOW_POP_CNT_30G', 'MAN_FLOW_POP_CNT_40G',
       'MAN_FLOW_POP_CNT_50G', 'MAN_FLOW_POP_CNT_60GU',
       'WMAN_FLOW_POP_CNT_10G', 'WMAN_FLOW_POP_CNT_20G',
       'WMAN_FLOW_POP_CNT_30G', 'WMAN_FLOW_POP_CNT_40G',
       'WMAN_FLOW_POP_CNT_50G', 'WMAN_FLOW_POP_CNT_60GU']].sum(axis=1)
sex_upop_geo['total'].sum()
time_upop_geo.columns
time_upop_geo['total'] = time_upop_geo[[  'TMST_00', 'TMST_01', 'TMST_02',
       'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07', 'TMST_08',
       'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13', 'TMST_14',
       'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19', 'TMST_20',
       'TMST_21', 'TMST_22', 'TMST_23']].sum(axis=1)
time_upop_geo['total'].sum()
day_upop_geo.columns
day_upop_geo['total'] = day_upop_geo[[ 'FLOW_POP_CNT_MON', 'FLOW_POP_CNT_TUS',
       'FLOW_POP_CNT_WED', 'FLOW_POP_CNT_THU', 'FLOW_POP_CNT_FRI',
       'FLOW_POP_CNT_SAT', 'FLOW_POP_CNT_SUN']].sum(axis=1)
day_upop_geo['total'].sum()

sex_upop_geo #868,377
time_upop_geo #849,751
day_upop_geo #1,139,947
HJD_gw #행정경계

sex_upop_hjd = gpd.sjoin(sex_upop_geo, HJD_gw, how='left', op='within')
sex_upop_hjd.isnull().sum() #22298
time_upop_geo = gpd.sjoin(time_upop_geo, HJD_gw, how='left', op='within')
time_upop_geo.isnull().sum() #21347
day_upop_geo = gpd.sjoin(day_upop_geo, HJD_gw, how='left', op='within')
day_upop_geo.isnull().sum() #27055

sex_upop_hjd2 = sex_upop_hjd.dropna(subset=['ADM_NM'])
time_upop_geo2 = time_upop_geo.dropna(subset=['ADM_NM'])
day_upop_geo2 = day_upop_geo.dropna(subset=['ADM_NM'])

sex_upop_hjd2['STD_YM'] = sex_upop_hjd2['STD_YM'].astype(str)
sex_upop_hjd2['연도'] = sex_upop_hjd2['STD_YM'].str[:4]  # 연도는 처음 4자리
sex_upop_hjd2['월'] = sex_upop_hjd2['STD_YM'].str[4:]  # 월은 5번째 이후의 2자리

time_upop_geo2['STD_YM'] = time_upop_geo2['STD_YM'].astype(str)
time_upop_geo2['연도'] = time_upop_geo2['STD_YM'].str[:4]  # 연도는 처음 4자리
time_upop_geo2['월'] = time_upop_geo2['STD_YM'].str[4:]  # 월은 5번째 이후의 2자리

day_upop_geo2['STD_YM'] = day_upop_geo2['STD_YM'].astype(str)
day_upop_geo2['연도'] = day_upop_geo2['STD_YM'].str[:4]  # 연도는 처음 4자리
day_upop_geo2['월'] = day_upop_geo2['STD_YM'].str[4:]  # 월은 5번째 이후의 2자리


sex_upop_hjd2 = sex_upop_hjd2[['STD_YM', '연도', '월','ADM_NM', 'ADM_CD','total',
        'MAN_FLOW_POP_CNT_10G','MAN_FLOW_POP_CNT_20G', 'MAN_FLOW_POP_CNT_30G',
        'MAN_FLOW_POP_CNT_40G','MAN_FLOW_POP_CNT_50G', 'MAN_FLOW_POP_CNT_60GU',
        'WMAN_FLOW_POP_CNT_10G', 'WMAN_FLOW_POP_CNT_20G','WMAN_FLOW_POP_CNT_30G',
        'WMAN_FLOW_POP_CNT_40G','WMAN_FLOW_POP_CNT_50G', 'WMAN_FLOW_POP_CNT_60GU', 'geometry',]]

time_upop_geo2 = time_upop_geo2[['STD_YM','연도', '월', 'ADM_NM', 'ADM_CD','total',
       'TMST_00', 'TMST_01', 'TMST_02','TMST_03', 'TMST_04','TMST_05', 'TMST_06', 'TMST_07', 'TMST_08','TMST_09', 'TMST_10',
       'TMST_11', 'TMST_12', 'TMST_13', 'TMST_14','TMST_15', 'TMST_16','TMST_17', 'TMST_18', 'TMST_19', 'TMST_20','TMST_21',
       'TMST_22', 'TMST_23', 'geometry']]

day_upop_geo2 = day_upop_geo2[['STD_YM', '연도', '월', 'ADM_NM', 'ADM_CD','total',
                               'FLOW_POP_CNT_MON', 'FLOW_POP_CNT_TUS','FLOW_POP_CNT_WED', 'FLOW_POP_CNT_THU',
                               'FLOW_POP_CNT_FRI','FLOW_POP_CNT_SAT', 'FLOW_POP_CNT_SUN', 'geometry',]]

sex_upop_hjd2.to_csv(path_pretreat + '//유동인구//sex_upop_hjd2.csv',index=False)
time_upop_geo2.to_csv(path_pretreat + '//유동인구//time_upop_geo2.csv',index=False)
day_upop_geo2.to_csv(path_pretreat + '//유동인구//day_upop_geo2.csv',index=False)


#연월,행정동별 합산
sex_1 = sex_upop_hjd2.groupby(['연도','월','ADM_NM'])[['total',
       'MAN_FLOW_POP_CNT_10G', 'MAN_FLOW_POP_CNT_20G', 'MAN_FLOW_POP_CNT_30G',
       'MAN_FLOW_POP_CNT_40G', 'MAN_FLOW_POP_CNT_50G', 'MAN_FLOW_POP_CNT_60GU',
       'WMAN_FLOW_POP_CNT_10G', 'WMAN_FLOW_POP_CNT_20G','WMAN_FLOW_POP_CNT_30G',
       'WMAN_FLOW_POP_CNT_40G','WMAN_FLOW_POP_CNT_50G', 'WMAN_FLOW_POP_CNT_60GU']].sum().reset_index()

sex_1.to_excel(path_pretreat + '//유동인구//연월,행정동별_유동인구(합계)_성연령별기준.xlsx',index=False)

sex_2 = sex_1.copy()
# 남자와 여자 각각의 합계 컬럼 생성
sex_2['남자'] = sex_2.filter(like='MAN_FLOW_POP_CNT').sum(axis=1)
sex_2['여자'] = sex_2.filter(like='WMAN_FLOW_POP_CNT').sum(axis=1)

# 연령대별 합계 컬럼 생성 (각 연령대별 합계를 계산)
sex_2['10대'] = sex_2.filter(like='10G').sum(axis=1)
sex_2['20대'] = sex_2.filter(like='20G').sum(axis=1)
sex_2['30대'] = sex_2.filter(like='30G').sum(axis=1)
sex_2['40대'] = sex_2.filter(like='40G').sum(axis=1)
sex_2['50대'] = sex_2.filter(like='50G').sum(axis=1)
sex_2['60대'] = sex_2.filter(like='60GU').sum(axis=1)
sex_2.to_excel(path_pretreat + '//유동인구//연월,행정동,성,연령별_유동인구(합계)_성연령별기준.xlsx',index=False)

sex_3 = sex_1.copy()
# 남자와 여자 인구의 연령대별 합계 계산
man_age_groups = {
    '10대': sex_3.filter(like='MAN_FLOW_POP_CNT_10G').sum(axis=0),
    '20대': sex_3.filter(like='MAN_FLOW_POP_CNT_20G').sum(axis=0),
    '30대': sex_3.filter(like='MAN_FLOW_POP_CNT_30G').sum(axis=0),
    '40대': sex_3.filter(like='MAN_FLOW_POP_CNT_40G').sum(axis=0),
    '50대': sex_3.filter(like='MAN_FLOW_POP_CNT_50G').sum(axis=0),
    '60대': sex_3.filter(like='MAN_FLOW_POP_CNT_60GU').sum(axis=0),
}

woman_age_groups = {
    '10대': sex_3.filter(like='WMAN_FLOW_POP_CNT_10G').sum(axis=0),
    '20대': sex_3.filter(like='WMAN_FLOW_POP_CNT_20G').sum(axis=0),
    '30대': sex_3.filter(like='WMAN_FLOW_POP_CNT_30G').sum(axis=0),
    '40대': sex_3.filter(like='WMAN_FLOW_POP_CNT_40G').sum(axis=0),
    '50대': sex_3.filter(like='WMAN_FLOW_POP_CNT_50G').sum(axis=0),
    '60대': sex_3.filter(like='WMAN_FLOW_POP_CNT_60GU').sum(axis=0),
}

# 데이터 프레임으로 변환
result = pd.DataFrame({
    '남자': man_age_groups,
    '여자': woman_age_groups
}).reset_index()
# 컬럼 이름 변경
result.columns = ['연령대', '남자', '여자']

#연월,행정동별 합산
time_1 = time_upop_geo2.groupby(['연도','월','ADM_NM'])[['total','TMST_00', 'TMST_01',
       'TMST_02', 'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07',
       'TMST_08', 'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13',
       'TMST_14', 'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19',
       'TMST_20', 'TMST_21', 'TMST_22', 'TMST_23']].sum().reset_index()

time_1.to_excel(path_pretreat + '//유동인구//연월,행정동별_유동인구(합계)_시간대별기준.xlsx',index=False)
time_2 = time_1.groupby(['연도', '월', 'ADM_NM'])[['total', 'TMST_00', 'TMST_01', 'TMST_02',
       'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07', 'TMST_08',
       'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13', 'TMST_14',
       'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19', 'TMST_20',
       'TMST_21', 'TMST_22', 'TMST_23']].sum().reset_index()
time_2.to_excel(path_pretreat + '//유동인구//연월,행정동별_유동인구(합계)2_시간대별기준.xlsx',index=False)

#연월,행정동별 합산
day_1 = day_upop_geo2.groupby(['연도','월','ADM_NM'])[['total', 'FLOW_POP_CNT_MON',
       'FLOW_POP_CNT_TUS', 'FLOW_POP_CNT_WED', 'FLOW_POP_CNT_THU',
       'FLOW_POP_CNT_FRI', 'FLOW_POP_CNT_SAT', 'FLOW_POP_CNT_SUN']].sum().reset_index()

day_1.to_excel(path_pretreat + '//유동인구//연월,행정동별_유동인구(합계)_요일별기준.xlsx',index=False)

#최근기준 성연령별유동인구별 주차장 공급비율 및 차량대수비율 계산
result_ratio = gpd.read_file(PATH_RESULT3 + '//주차장수요//주차장확보율.gpkg',driver='GPKG')
sex_ratio = pd.read_excel(path_pretreat + '//유동인구//연월,행정동,성,연령별_유동인구(합계)_성연령별기준.xlsx')
sex_ratio_f = sex_ratio[(sex_ratio['연도'] == 2022) & (sex_ratio['월'] == 12)]
sex_ratio_f.columns
result_ratio.columns
#유동인구 대비 차량등록대수 비율
f_pop_1 = sex_ratio_f[['ADM_NM','total' ]]
f_pop_1.rename(columns={'ADM_NM':'행정동','total':'유동인구'},inplace=True)
result_2f = pd.merge(f_pop_1,result_ratio,on='행정동',how='left')
result_2f = result_2f[['행정동','유동인구', '총차량수', '총주차수','geometry']]
result_2f['주차장공급비율'] = ((result_2f['총주차수'] / result_2f['유동인구'])*100).round(1)
result_2f['차량대수비율'] = ((result_2f['총차량수'] / result_2f['유동인구'])*100).round(1)

# 평균 비율 계산
mean_주차장공급비율 = result_2f['주차장공급비율'].mean()
mean_차량대수비율 = result_2f['차량대수비율'].mean()
# 평균 이하/이상의 구분 컬럼 추가
result_2f['주차장공급비율_평균정도'] = result_2f['주차장공급비율'].apply(lambda x: '이상' if x >= mean_주차장공급비율 else '이하')
result_2f['차량대수비율_평균정도'] = result_2f['차량대수비율'].apply(lambda x: '이상' if x >= mean_차량대수비율 else '이하')

# 결과 컬럼 정리 및 GeoDataFrame 생성
result_2f = result_2f[['행정동', '유동인구', '총차량수', '총주차수','주차장공급비율', '차량대수비율', '주차장공급비율_평균정도', '차량대수비율_평균정도', 'geometry']]
result_2f = gpd.GeoDataFrame(result_2f, geometry='geometry')
result_2f.to_file(PATH_RESULT3 + '//주차장수요//유동인구대비수요.gpkg', driver='GPKG')
result_2f.to_excel(path_result + '//주차장수요//유동인구대비수요.xlsx',index=False)



# 공영주차장 주소 데이터
parking_1_point = pd.read_excel(path_data_in2 + '//주차장 정리본2.xlsx')
# 오류좌표 수작업 수정
# 매화아파트, 베스파 앞(전라남도 광양시 구산리 738-4) : 34.983584, 127.578891로 수정 (매화아파트)
parking_1_point.loc[parking_1_point['주차장 주소'] == '전라남도 광양시 구산리 738-4', '위도'] = '34.98358'
parking_1_point.loc[parking_1_point['주차장 주소'] == '전라남도 광양시 구산리 738-4', '경도'] = '127.57889'
# (전라남도 광양시 인동리 66-1) :34.967754, 127.588899 (전남도립미술관)
parking_1_point.loc[parking_1_point['주차장 주소'] == '전라남도 광양시 인동리 66-1', '위도'] = '34.96775'
parking_1_point.loc[parking_1_point['주차장 주소'] == '전라남도 광양시 인동리 66-1', '경도'] = '127.58889'

parking_1_point['geometry'] = parking_1_point.apply(lambda row: Point(row['경도'], row['위도']), axis=1)
parking_1_point_geo = gpd.GeoDataFrame(parking_1_point, geometry='geometry',crs='epsg:4326')
parking_1_point_geo = parking_1_point_geo.to_crs(epsg=5186)
parking_1_point_geo.to_file(PATH_RESULT3 + '//주차장공급//공영주차장위치좌표.gpkg', driver='GPKG')

parking_1_point_geo2 = parking_1_point_geo[['읍면동','노상_노외 구분','유료_무료 구분','주차면수']]
#parking_1_point_geo2 = parking_1_point_geo2.groupby('읍면동').agg({'노상_노외 구분': 'count', '유료_무료 구분': 'count', '주차면수': 'sum'}).reset_index()
pivot_table_sum = pd.pivot_table(parking_1_point_geo2, index='읍면동', values='주차면수', aggfunc='sum', fill_value=0)
pivot_table_count = pd.pivot_table(parking_1_point_geo2, index='읍면동', columns='노상_노외 구분', aggfunc='size', fill_value=0)
pivot_table_combined = pd.concat([pivot_table_count, pivot_table_sum], axis=1).reset_index()

pivot_table_combined.rename(columns = {'읍면동' : 'ADM_NM'}, inplace = True)

parking_1_point_HJD = pd.merge(HJD_gw,pivot_table_combined,on='ADM_NM',how='left')
parking_1_point_HJD.fillna(0,inplace = True)

parking_1_point_HJD = parking_1_point_HJD[['ADM_NM','노상', '노외', '주차면수']]
parking_1_point_HJD['공영주차장 개수'] = parking_1_point_HJD[['노상','노외']].sum(axis=1)
parking_1_point_HJD.to_excel(path_result + '//주차장공급//공영주차장_좌표기준.xlsx',index=False)







#거주인구 기준 공영주차장 결핍도 분석 - 공영주차장 정확한 주소 정보 필요
home_pop = gpd.read_file(PATH_RESULT3 + "//주차장수요//거주인구.gpkg", driver='GPKG') #2024.07월 말
grid_100 = gpd.read_file(PATH_RESULT3 + "//주차장수요//100_격자거주인구.shp", encoding='utf-8') #2024.04
parking_1_point_geo = gpd.read_file(PATH_RESULT3 + '//주차장공급//공영주차장위치좌표.gpkg')


home_pop['주민등록인구수'].sum()
grid_100['val'].sum()


# 공영주차장 공급량 (금호동 유무에 따라)
'''
parking_1_point_geo.rename(columns = {'읍면동':'행정동'},inplace=True)
parking_fac = parking_1_point_geo.groupby('행정동')['주차면수'].sum().reset_index()
parking_fac.rename(columns={'주차면수': '공영주차면수'}, inplace=True)

home_pop.rename(columns = {'ADM_NM':'행정동'},inplace=True)
parking_fac1 = pd.merge(parking_fac,home_pop, on='행정동',how='outer')
parking_fac1 = parking_fac1.fillna(0)
parking_fac1['공급량(F)'] = (parking_fac1['공영주차면수'] / parking_fac1['주민등록인구수']) * 1000
# '공급량' 컬럼을 표준화하여 '표준화지수'라는 새로운 컬럼으로 추가
scaler = MinMaxScaler()
parking_fac1['공급량_표준화지수'] = scaler.fit_transform(parking_fac1[['공급량(F)']])
parking_fac1 = parking_fac1[['행정동','공영주차면수','주민등록인구수','공급량(F)','공급량_표준화지수','geometry']]
parking_fac11 = parking_fac1[['행정동','공영주차면수','주민등록인구수','공급량(F)','공급량_표준화지수']]
parking_fac11.to_excel(path_result + '//결핍도분석//공영주차장_공급량(F).xlsx',index=False)
parking_fac1 = gpd.GeoDataFrame(parking_fac1, geometry='geometry', crs='EPSG:5186')
parking_fac1.to_file(PATH_RESULT3 + '//결핍도분석//공영주차장_공급량(F).gpkg', driver='GPKG')
'''
parking1_geo.rename(columns = {'ADM_NM':'행정동'},inplace=True)
parking1_geo.columns
parking_fac = parking1_geo.groupby('행정동')['공영주차장'].sum().reset_index()
parking_fac.rename(columns={'공영주차장': '공영주차면수'}, inplace=True)

home_pop.rename(columns = {'ADM_NM':'행정동'},inplace=True)
parking_fac1 = pd.merge(parking_fac,home_pop, on='행정동',how='outer')
parking_fac1 = parking_fac1.fillna(0)
parking_fac1['공급량(F)'] = (parking_fac1['공영주차면수'] / parking_fac1['주민등록인구수']) * 1000
# '공급량' 컬럼을 표준화하여 '표준화지수'라는 새로운 컬럼으로 추가
scaler = MinMaxScaler()
parking_fac1['공급량_표준화지수'] = scaler.fit_transform(parking_fac1[['공급량(F)']])
parking_fac1 = parking_fac1[['행정동','공영주차면수','주민등록인구수','공급량(F)','공급량_표준화지수','geometry']]
parking_fac11 = parking_fac1[['행정동','공영주차면수','주민등록인구수','공급량(F)','공급량_표준화지수']]
parking_fac11.to_excel(path_result + '//결핍도분석//공영주차장_공급량(F).xlsx',index=False)
parking_fac1 = gpd.GeoDataFrame(parking_fac1, geometry='geometry', crs='EPSG:5186')
parking_fac1.to_file(PATH_RESULT3 + '//결핍도분석//공영주차장_공급량(F).gpkg', driver='GPKG')



# 격자 행정동 매핑
grid_100 = grid_100.to_crs(epsg=5186)
mapping = gpd.sjoin(grid_100, home_pop,how='inner')

# 특정 컬럼 값이 중복되는 행 추출
duplicate_rows = mapping[mapping.duplicated(subset='gid', keep=False)]
#mapping한 기존격자에서 중복 되는 행 제거
mapping = mapping.drop_duplicates(subset='gid', keep=False)

##중복되는 행의 처리를 위한 작업
#중복되는 행들과 경계면 오버레이하여 걸쳐진 부분 면적 구함
res_union = gpd.overlay(duplicate_rows, home_pop, how='intersection')

# 두 컬럼의 값이 동일한 행 추출
same_res_union = res_union[res_union['행정동_1'] == res_union['행정동_2']]
#면적계산
same_res_union['area'] = same_res_union['geometry'].area
# 같은 GID 값끼리 그룹화하여 면적컬럼에서 가장 큰 값을 가진 읍면동컬럼으로 업데이트
result_union = same_res_union.groupby('gid').apply(lambda x: x.loc[x['area'].idxmax()]).reset_index(drop=True)
# 행정동 매핑한 격자와 여러개의 행정동이 걸처진 격자처리 후 두개의 데이터프레임을 다시 합치는 과정
mapping = mapping[['gid','ADM_CD','행정동','val','geometry']]

result_union = result_union[['gid', 'ADM_CD_1', '행정동_1', 'val', 'geometry']]
# 'A' 컬럼의 이름을 'X'로 변경
result_union.rename(columns={'ADM_CD_1': 'ADM_CD'}, inplace=True)
result_union.rename(columns={'행정동_1': '행정동'}, inplace=True)

#결과
Gwangyang_grid = pd.concat([mapping, result_union], axis=0)
# 인덱스를 리셋합니다.
Gwangyang_grid = Gwangyang_grid.reset_index(drop=True)
Gwangyang_grid.to_file(PATH_RESULT3 + '//주차장수요//격자_행정동매핑.gpkg', driver='GPKG')

grid_100_val = Gwangyang_grid.groupby('행정동')['val'].sum().reset_index()
grid_100_val.to_excel(path_result + '//주차장수요//격자거주인구_2024.04기준.xlsx', index=False)


###############################################
# 이용접근성 지수
def assessibility(facil, size , layer):
    # 시설 버퍼 생성
    buffer = gpd.GeoSeries(facil.buffer(size))
    facil_buffer = pd.concat([facil, buffer], axis=1)
    facil_buffer.rename(columns={0: 'buffer_geometry'}, inplace=True)
    facil_buffer_geo = gpd.GeoDataFrame(facil_buffer, geometry='buffer_geometry')
    facil_buffer_geo_2 = facil_buffer_geo.drop('geometry',axis=1)
    facil_intersection = gpd.overlay(layer, facil_buffer_geo_2, how='intersection')

    # 격자 교차면적 계산
    a = layer.loc[0, 'geometry'].area / 2  # 격자 한개 면적 계산
    facil_intersection['area'] = facil_intersection['geometry'].area
    dupli = facil_intersection[facil_intersection['gid'].duplicated(keep=False)]

    # 중복 gid sum 계산 후 전체 면적에서 빼기
    nan = dupli.groupby('gid')['area'].sum().reset_index() #수정
    #nan.reset_index(inplace=True)
    nan = nan[['gid', 'area']]
    nan['ex_area'] = a - nan['area']
    del nan['area']
    merge = pd.merge(nan, dupli, on='gid')
    merge = merge[merge['area'] > merge['ex_area']]
    merge.sort_values('area', ascending=False, inplace=True)
    merge.drop_duplicates(subset='gid', keep='first', inplace=True)
    del merge['ex_area']
    facil_intersection.drop_duplicates(subset='gid', keep='first', inplace=True)
    facil_intersection = facil_intersection[facil_intersection['area'] > a]
    facil_intersection = pd.concat([facil_intersection, merge], axis=0)

    # 격자 인구 분리
    list = facil_intersection['gid'].unique()
    buffer_nan = layer[~layer['gid'].isin(list)]
    buffer_ok = layer[layer['gid'].isin(list)]

    ok = buffer_ok.groupby('hjd_nm')['val'].sum().reset_index() #수정
    #ok.reset_index(inplace=True)
    ok.columns = ['hjd_nm', 'ok_val']

    nan = buffer_nan.groupby('hjd_nm')['val'].sum().reset_index() #수정
    #nan.reset_index(inplace=True)
    nan.columns = ['hjd_nm', 'nan_val']

    result_a = pd.merge(ok, nan, on='hjd_nm', how='outer')
    result_a['nan_val'] = result_a['nan_val'].fillna(0).astype(float)
    result_a['이용접근성(A)'] = ((result_a['ok_val']/(result_a['ok_val']+result_a['nan_val']))).round(2)

    return result_a, facil_intersection

# 행정동별 함수호출
Gwangyang_grid.columns = ['gid','hjd_cd','hjd_nm','val','geometry']

parking_accessibility = assessibility(parking_1_point_geo,500,Gwangyang_grid)
parking_acc = parking_accessibility[0] #이용접근
parking_buffer = parking_accessibility[1] #버퍼


parking_acc = pd.merge(parking_acc,home_pop, left_on='hjd_nm',right_on='행정동',how='outer')
parking_acc = parking_acc.fillna(0)
scaler = MinMaxScaler()
parking_acc['이용접근성_표준화'] = scaler.fit_transform(parking_acc[['이용접근성(A)']])

parking_acc_geo = parking_acc[['행정동','ok_val','nan_val','이용접근성(A)','이용접근성_표준화','geometry']]
parking_acc_geo = gpd.GeoDataFrame(parking_acc_geo, geometry='geometry', crs='EPSG:5186')
parking_acc = parking_acc[['행정동','ok_val','nan_val','이용접근성(A)','이용접근성_표준화']]

parking_buffer.to_file(PATH_RESULT3 + '//결핍도분석//공영주차장_버퍼.gpkg', driver='GPKG')
parking_acc_geo.to_file(PATH_RESULT3 + '//결핍도분석//공영주차장_이용접근성(A).gpkg', driver='GPKG')
parking_acc.to_excel(path_result + '//결핍도분석//공영주차장_이용접근성(A).xlsx',index=False)


#결핍지수 산출

def_parking = pd.merge(parking_fac1,parking_acc,on='행정동')
def_parking.columns
def_parking = def_parking[['행정동','공급량(F)','공급량_표준화지수','이용접근성(A)','이용접근성_표준화','geometry']]
def_parking['결핍지수'] = def_parking[['공급량_표준화지수','이용접근성_표준화']].mean(axis=1)
def_parking['결핍지수'] = 1 - def_parking['결핍지수']
def_parking = def_parking[['행정동','공급량(F)','공급량_표준화지수','이용접근성(A)','이용접근성_표준화','결핍지수','geometry']]
def_parking = gpd.GeoDataFrame(def_parking, geometry='geometry', crs='EPSG:5186')
def_parking.to_file(PATH_RESULT3 + '//결핍도분석//공영주차장_결핍지수.gpkg', driver='GPKG')
def_parking_1 = def_parking[['행정동','공급량(F)','공급량_표준화지수','이용접근성(A)','이용접근성_표준화','결핍지수']]
def_parking_1.to_excel(path_result + '//결핍도분석//공영주차장_결핍지수.xlsx',index=False)


#PPT 격자 생성
grid_100_hjd = gpd.read_file(PATH_RESULT3 + '//주차장수요//격자_행정동매핑.gpkg', driver='GPKG')
parking_1_point_geo = gpd.read_file(PATH_RESULT3 + '//주차장공급//공영주차장위치좌표.gpkg') #공영주차장 위치좌표
parking_cnt = pd.read_excel(path_middle + '//건축물대장대상목록.xlsx') # 건축물대장 주차장 위치좌표

#ppt1 -금호동,중마동 전체주차장 격자
ppt1_1 = parking_1_point_geo[(parking_1_point_geo['읍면동'] == '금호동') | (parking_1_point_geo['읍면동'] == '중마동')]
#건축물대장 지오코딩
parking_cnt_gold = parking_cnt[parking_cnt['ADM_NM'] == '금호동'] # 금호동
parking_cnt_gold.to_excel(path_middle + '//특정지역지오코딩//건축물대장대상목록_금호동.xlsx', index=False)
#금호동 지오코딩
parking_cnt_gold_geo = gpd.read_file(path_middle + '//특정지역지오코딩//지오코딩완료//건축물대장대상목록_금호동_WGS.shp',encoding='cp949')
parking_cnt_gold_geo2 = gpd.GeoDataFrame(parking_cnt_gold_geo, geometry='geometry',crs='epsg:4326')
parking_cnt_gold_geo2 = parking_cnt_gold_geo2.to_crs(epsg=5186)
parking_cnt_gold_geo2['최종주차장'] = parking_cnt_gold_geo2['최종주차장'].astype(int)
parking_cnt_gold_geo2 = parking_cnt_gold_geo2.sort_values(by='최종주차장', ascending=False)
parking_cnt_gold_geo2.to_file(PATH_RESULT3 + '//주차장공급//특정지역시각화(ppt자료)//좌표//건축물대장_금호동주차장.gpkg', driver='GPKG')

parking_cnt_jung = parking_cnt[parking_cnt['ADM_NM'] == '중마동'] # 중마동
parking_cnt_jung.to_excel(path_middle + '//특정지역지오코딩//건축물대장대상목록_중마동.xlsx', index=False)
#중마동 지오코딩
parking_cnt_jung_geo = gpd.read_file(path_middle + '//특정지역지오코딩//지오코딩완료//건축물대장대상목록_중마동_WGS.shp',encoding='cp949')
parking_cnt_jung_geo2 = gpd.GeoDataFrame(parking_cnt_jung_geo, geometry='geometry',crs='epsg:4326')
parking_cnt_jung_geo2 = parking_cnt_jung_geo2.to_crs(epsg=5186)
parking_cnt_jung_geo2['최종주차장'] = parking_cnt_jung_geo2['최종주차장'].astype(int)

parking_cnt_jung_geo2 = parking_cnt_jung_geo2.sort_values(by='최종주차장', ascending=False)
parking_cnt_jung_geo2.to_file(PATH_RESULT3 + '//주차장공급//특정지역시각화(ppt자료)//좌표//건축물대장_중마동주차장.gpkg', driver='GPKG')

#ppt1 -금호동,중마동 전체주차장 격자
parking_cnt_gold_geo2.columns
ppt1_1 = ppt1_1[['읍면동','주차장 주소','주차면수','geometry']]
ppt1_1.columns = ['행정동명','주소','주차면수','geometry']
parking_cnt_jung_geo2.columns
ppt1_2 = pd.concat([parking_cnt_gold_geo2, parking_cnt_jung_geo2], axis=0)
ppt1_2 = ppt1_2[['행정동명','대지위치','최종주차장','geometry']]
ppt1_2.columns = ['행정동명','주소','주차면수','geometry']
#금호동 중마동 전체주차장 좌표
ppt1_xy = pd.concat([ppt1_1, ppt1_2], axis=0)

#격자에 매핑
# 포인트와 격자 간의 공간적 조인 (Spatial Join)
joined_gdf = gpd.sjoin(ppt1_xy, grid_100_hjd, how='inner', predicate='within')
# 각 격자별 주차면수 집계
grid_summary = joined_gdf.groupby('index_right')['주차면수'].sum().reset_index()
# 격자 데이터프레임에 주차면수 컬럼 추가
ppt1 = grid_100_hjd.merge(grid_summary, left_index=True, right_on='index_right', how='left')
# '주차면수' 컬럼이 없는 격자는 0으로 설정
ppt1['주차면수'] = ppt1['주차면수'].fillna(0)
ppt1 = ppt1[['index_right','gid', 'ADM_CD', '행정동', 'val', '주차면수', 'geometry']]
ppt1.to_file(PATH_RESULT3 + '//주차장공급//특정지역시각화(ppt자료)//격자//전체주차장_금호동중마동(ppt).gpkg', driver='GPKG')
ppt1.columns
ppt1.loc[ppt1['행정동'] == '금호동', '주차면수'].max()
ppt1.loc[ppt1['행정동'] == '중마동', '주차면수'].max()


#ppt2 - 공영주차장 격자
# 포인트와 격자 간의 공간적 조인 (Spatial Join)
joined_gdf2 = gpd.sjoin(parking_1_point_geo, grid_100_hjd, how='inner', predicate='within')
# 각 격자별 주차면수 집계
grid_summary = joined_gdf2.groupby('index_right')['주차면수'].sum().reset_index()
# 격자 데이터프레임에 주차면수 컬럼 추가
ppt2 = grid_100_hjd.merge(grid_summary, left_index=True, right_on='index_right', how='left')
# '주차면수' 컬럼이 없는 격자는 0으로 설정
ppt2['주차면수'] = ppt2['주차면수'].fillna(0)
ppt2 = ppt2[['index_right','gid', 'ADM_CD', '행정동', 'val', '주차면수', 'geometry']]
ppt2.to_file(PATH_RESULT3 + '//주차장공급//특정지역시각화(ppt자료)//격자//공영주차장(ppt).gpkg', driver='GPKG')


ppt2.loc[ppt2['주차면수'] == ppt2.loc[ppt2['행정동'] == '중마동', '주차면수'].max(), 'gid'].iloc[0]
ppt2.loc[ppt2['행정동'] == '광영동', '주차면수'].max()
ppt2.loc[ppt2['주차면수'] == ppt2.loc[ppt2['행정동'] == '광영동', '주차면수'].max(), 'gid'].iloc[0]
ppt2.loc[ppt2['행정동'] == '광양읍', '주차면수'].max()
ppt2.loc[ppt2['주차면수'] == ppt2.loc[ppt2['행정동'] == '광양읍', '주차면수'].max(), 'gid'].iloc[0]


#ppt3 - 중마동,광양읍 차량등록대수 격자
pre_car = pd.read_excel(path_data_analyze+ '//등록차량정보//등록차량정보.xlsx')
pre_car.columns
ppt3_1 = pre_car[(pre_car['행정동'] == '전라남도 광양시 중마동') | (pre_car['행정동'] == '전라남도 광양시 광양읍')]
ppt3_1 = ppt3_1.reset_index()
# 청크 크기 설정
chunk_size = 9999
# 데이터프레임의 총 행 수
total_rows = len(ppt3_1)
# 데이터프레임을 청크로 나누어 저장
for i in range(0, total_rows, chunk_size):
    # 청크 추출
    chunk = ppt3_1.iloc[i:i + chunk_size]
    # 청크를 파일로 저장
    path = path_data_analyze + '//등록차량정보//특정지역//'
    chunk.to_csv(path +f'chunk_{i // chunk_size + 1}.csv', index=False, encoding='cp949')

dc1 = gpd.read_file(path_data_analyze + '//등록차량정보//특정지역//지오코딩후//chunk_1_WGS.shp',encoding='cp949')
dc2 = gpd.read_file(path_data_analyze + '//등록차량정보//특정지역//지오코딩후//chunk_2_WGS.shp',encoding='cp949')
dc3 = gpd.read_file(path_data_analyze + '//등록차량정보//특정지역//지오코딩후//chunk_3_WGS.shp',encoding='cp949')
dc4 = gpd.read_file(path_data_analyze + '//등록차량정보//특정지역//지오코딩후//chunk_4_WGS.shp',encoding='cp949')
dc5 = gpd.read_file(path_data_analyze + '//등록차량정보//특정지역//지오코딩후//chunk_5_WGS.shp',encoding='cp949')
dc6 = pd.read_csv(path_data_analyze + '//등록차량정보//특정지역//지오코딩후//chunk_6_WGS.csv',encoding='cp949')
dc7 = gpd.read_file(path_data_analyze + '//등록차량정보//특정지역//지오코딩후//chunk_7_WGS.shp',encoding='cp949')

dc6['geometry'] = dc6.apply(lambda row: Point(row['X'], row['Y']), axis=1)
dc6 = gpd.GeoDataFrame(dc6, geometry='geometry',crs='epsg:4326')

dc1 = dc1[['차량등록번', '용도구분', '행정동', 'geometry']]
dc2 = dc2[['차량등록번', '용도구분', '행정동', 'geometry']]
dc3 = dc3[['차량등록번', '용도구분', '행정동', 'geometry']]
dc4 = dc4[['차량등록번', '용도구분', '행정동', 'geometry']]
dc5 = dc5[['차량등록번', '용도구분', '행정동', 'geometry']]
dc6 = dc6[['차량등록번호', '용도구분', '행정동', 'geometry']]
dc7 = dc7[['차량등록번', '용도구분', '행정동', 'geometry']]

dc6.rename(columns = {'차량등록번호':'차량등록번'},inplace=True)
dataframes = [dc1, dc2, dc3, dc4, dc5, dc6, dc7]
ppt3_2 = pd.concat(dataframes, ignore_index=True)
ppt3_2.rename(columns = {'차량등록번':'차량등록번호'},inplace=True)
ppt3_2 = gpd.GeoDataFrame(ppt3_2, geometry='geometry',crs='epsg:4326')
ppt3_2 = ppt3_2.to_crs(epsg=5186)
ppt3_2.to_file(PATH_RESULT3 + '//주차장수요//특정지역시각화(ppt자료)//등록차량정보(중마동,광양읍).gpkg', driver='GPKG')
ppt3_2.reset_index(drop=False, inplace=True)
#격자에 매핑
# 포인트와 격자 간의 공간적 조인 (Spatial Join)
ppt3_2['gid'] = ppt3_2.sjoin(grid_100_hjd, how='left', op='within')['gid']
count_per_grid = ppt3_2.groupby('gid').size().reset_index(name='등록차량수')
count_per_grid['등록차량수']= count_per_grid['등록차량수'].fillna(0)
count_per_grid2 = count_per_grid.groupby('gid')['등록차량수'].sum().reset_index()
ppt3 = pd.merge(grid_100_hjd,count_per_grid2,on='gid',how='left')
ppt3=ppt3.fillna(0)
ppt3.to_file(PATH_RESULT3 + '//주차장수요//특정지역시각화(ppt자료)//등록차량정보격자(중마동,광양읍).gpkg', driver='GPKG')


#ppt4 - 격자 거주인구
grid_100['val'] = grid_100['val'].fillna(0)
grid_100 = grid_100[['gid','val','geometry']]
grid_100 = grid_100.to_crs(epsg=5186)
grid_100.to_file(PATH_RESULT3 + '//주차장수요//격자거주인구.gpkg', driver='GPKG')

#ppt5 - 태인동 등록차량대수
pre_car = pd.read_excel(path_data_analyze+ '//등록차량정보//등록차량정보.xlsx')
ppt5_1 = pre_car[(pre_car['행정동'] == '전라남도 광양시 태인동')].reset_index(drop= True)
ppt5_1.to_csv(path_data_analyze + '//등록차량정보//특정지역//태인동.csv',encoding='cp949',index=False)

ppt5_2= gpd.read_file(path_data_analyze + '//등록차량정보//특정지역//지오코딩후//태인동_WGS.shp',encoding='cp949')
ppt5_2 = gpd.GeoDataFrame(ppt5_2, geometry='geometry',crs='epsg:4326')
ppt5_2 = ppt5_2.to_crs(epsg=5186)
# 포인트와 격자 간의 공간적 조인 (Spatial Join)
ppt5_2['gid'] = ppt5_2.sjoin(grid_100_hjd, how='left', op='within')['gid']
count_per_grid3 = ppt5_2.groupby('gid').size().reset_index(name='등록차량수')
count_per_grid3['등록차량수']= count_per_grid3['등록차량수'].fillna(0)
count_per_grid4 = count_per_grid3.groupby('gid')['등록차량수'].sum().reset_index()
ppt5 = pd.merge(grid_100_hjd,count_per_grid4,on='gid',how='left')
ppt5=ppt5.fillna(0)
ppt5.to_file(PATH_RESULT3 + '//주차장수요//특정지역시각화(ppt자료)//등록차량정보격자(태인동).gpkg', driver='GPKG')


#유동인구 격자 시각화 최근기준
flowpop_2122 = gpd.read_file(PATH_RESULT3 +'//주차장수요//sex유동인구.gpkg',driver='GPKG')
flowpop_2212 = flowpop_2122[flowpop_2122['STD_YM'] == 202212]

flowpop_2212['gid'] = flowpop_2212.sjoin(grid_100_hjd, how='left', op='within')['gid']
columns = ['MAN_FLOW_POP_CNT_10G',
       'MAN_FLOW_POP_CNT_20G', 'MAN_FLOW_POP_CNT_30G', 'MAN_FLOW_POP_CNT_40G',
       'MAN_FLOW_POP_CNT_50G', 'MAN_FLOW_POP_CNT_60GU',
       'WMAN_FLOW_POP_CNT_10G', 'WMAN_FLOW_POP_CNT_20G',
       'WMAN_FLOW_POP_CNT_30G', 'WMAN_FLOW_POP_CNT_40G',
       'WMAN_FLOW_POP_CNT_50G', 'WMAN_FLOW_POP_CNT_60GU']
flowpop_2212['flow_pop'] = flowpop_2212[columns].sum(axis=1)

count_per_grid4 = flowpop_2212.groupby('gid')['flow_pop'].sum().reset_index(name='유동인구')
count_per_grid4['유동인구']= count_per_grid4['유동인구'].fillna(0)
count_per_grid5 = count_per_grid4.groupby('gid')['유동인구'].sum().reset_index()
ppt6 = pd.merge(grid_100_hjd,count_per_grid5,on='gid',how='left')
ppt6=ppt6.fillna(0)
ppt6.to_file(PATH_RESULT3 + '//주차장수요//특정지역시각화(ppt자료)//202212유동인구.gpkg', driver='GPKG')

########################################################################################################################
# 총괄표제부 쏠림현상

aa = a30[a30['최대주차장'] != a30['총괄주차장']]
aa.columns
#bb = docu_unit.copy()
#aa['총주차수'] = aa[['옥내기계식대수', '옥외기계식대수', '옥내자주식대수', '옥외자주식대수']].sum(axis=1)
#bb['총주차수'] = bb[['옥내기계식대수', '옥외기계식대수', '옥내자주식대수', '옥외자주식대수']].sum(axis=1)

def extract_adm_nm(address):
    # 공백을 기준으로 주소를 나눈 후, 순서대로 부분을 검사
    parts = address.split()
    for part in parts:
        if part.endswith(('읍', '면', '동')):
            return part
    return ''

# 'ADM_NM' 컬럼 생성
aa['ADM_NM'] = aa['대지위치'].apply(extract_adm_nm)


# 수작업 수정 (도이동,마동,성황동,중군동,중동,황금동,황길동)

aa.loc[aa['ADM_NM'] == '성황동']
aa.loc[aa['ADM_NM'] == '도이동', 'ADM_NM'] = '골약동'
aa.loc[aa['ADM_NM'] == '마동', 'ADM_NM'] = '중마동'
aa.loc[aa['ADM_NM'] == '성황동', 'ADM_NM'] = '골약동'
aa.loc[aa['ADM_NM'] == '중군동', 'ADM_NM'] = '골약동'
aa.loc[aa['ADM_NM'] == '중동', 'ADM_NM'] = '중마동'
aa.loc[aa['ADM_NM'] == '황금동', 'ADM_NM'] = '골약동'
aa.loc[aa['ADM_NM'] == '황길동', 'ADM_NM'] = '골약동'


aa.to_csv(path_data_analyze + '//건축물대장//합계다른경우.csv',encoding='cp949',index=False)

#지오코딩
aa_point = gpd.read_file(path_data_analyze + '//건축물대장//포인트생성//합계다른경우_WGS.shp',encoding='cp949')
aa_point[['총주차수합', '최대주차장','총괄주차장']] = aa_point[['총주차수합', '최대주차장','총괄주차장']].astype(float)
aa_point['차이값'] = aa_point['총괄주차장'] - aa_point['총주차수합']
aa_sorted = aa_point.sort_values(by='차이값', ascending=False)
aa_sorted

#aa_point[['옥내기계식', '옥외기계식', '옥내자주식', '옥외자주식']] = aa_point[['옥내기계식', '옥외기계식', '옥내자주식', '옥외자주식']].replace('NaN', np.nan).fillna(0.0).astype(float)
#bb_point[['옥내기계식', '옥외기계식', '옥내자주식', '옥외자주식']] = bb_point[['옥내기계식', '옥외기계식', '옥내자주식', '옥외자주식']].replace('NaN', np.nan).fillna(0.0).astype(float)

#aa_point['총주차수'] = aa_point[['옥내기계식', '옥외기계식', '옥내자주식', '옥외자주식']].sum(axis=1)
#bb_point['총주차수'] = bb_point[['옥내기계식', '옥외기계식', '옥내자주식', '옥외자주식']].sum(axis=1)

aa_point2 = gpd.GeoDataFrame(aa_sorted, geometry='geometry',crs='epsg:4326')
aa_point2 = aa_point2.to_crs(epsg=5186)
aa_point3 = aa_point2.fillna(0.0)

# 포인트와 격자 간의 공간적 조인 (Spatial Join)
aa_point3['gid'] = aa_point3.sjoin(grid_100_hjd, how='left', op='within')['gid']
aa_point3_grid = aa_point3.groupby('gid')[['총주차수합', '최대주차장','총괄주차장']].sum().reset_index()
aa_point3_grid = aa_point3_grid.merge(
    aa_point3[['gid', '대지위치']].drop_duplicates(subset='gid'),
    on='gid',
    how='left'
)

ppt7_1 = pd.merge(grid_100_hjd,aa_point3_grid,on='gid',how='left')
ppt7_1 = ppt7_1.fillna(0.0)
ppt7_1 = ppt7_1[['gid', 'ADM_CD', '행정동','대지위치','총주차수합', '최대주차장','총괄주차장','geometry']]
ppt7_2 = ppt7_1.merge(docu_unit[['대지위치', '주용도코드명']], on='대지위치', how='left')
ppt7_2 = ppt7_2.merge(ppt7_1[['대지위치']], on='대지위치')
ppt7_2.drop_duplicates(inplace=True)
ppt7_3 = ppt7_2.drop_duplicates(subset='gid')

ppt7_3.to_file(PATH_RESULT3 + '//주차장공급//특정지역시각화(ppt자료)//격자//합계다른경우.gpkg',driver='GPKG')

########################################################################################################################
