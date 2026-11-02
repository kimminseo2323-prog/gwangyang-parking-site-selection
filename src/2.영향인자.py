###########################################################
# 과 제 명: A10_광양시_주차장입지분석
# 파 일 명: 2.영향인자 생성
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
from shapely.geometry import Point
from shapely.ops import nearest_points


import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler
###################################################################################################
grid_100 = gpd.read_file(PATH_RESULT3 + '//주차장수요//격자거주인구.gpkg', driver='GPKG')
HJD_B = gpd.read_file(PATH_RESULT3 + '//행정경계//행정경계.gpkg', driver='GPKG')


# 입지분석 해당 행정동 격자만 추출
HJD_B = HJD_B[(HJD_B['ADM_NM'] == '광양읍') | (HJD_B['ADM_NM'] == '중마동')]
# 교차된 격자 저장을 위한 빈 DataFrame
result_grid_gdf = gpd.GeoDataFrame(columns=grid_100.columns)
for _, region in HJD_B.iterrows():
    # 각 경계면의 geometry와 격자 데이터의 교차
    region_geom = region['geometry']
    intersected = grid_100[grid_100.geometry.intersects(region_geom)]
    result_grid_gdf = pd.concat([result_grid_gdf, intersected])

result_grid_gdf = gpd.GeoDataFrame(result_grid_gdf, geometry='geometry')
result_grid_gdf.to_file(PATH_RESULT3 + '//영향인자//격자_중마동,광양읍.gpkg', driver='GPKG')

# 300M 버퍼 생성 및 버퍼 내 특정컬럼값 합산 함수
def aggregate_column_excluding_center(grid_gdf, column_name, buffer_distance=300):
    """
    중심점 좌표를 기준으로 버퍼를 생성하여, 특정 컬럼에 대한 300m 버퍼 내의 격자 값을 합산합니다.
    중심점을 포함한 격자는 제외합니다.

    Parameters:
    - grid_gdf: GeoDataFrame, 격자 데이터
    - column_name: str, 합산할 컬럼명
    - buffer_distance: int, 버퍼의 거리 (미터 단위), 기본값 300

    Returns:
    - buffer_gdf: GeoDataFrame, 합산된 결과를 포함한 GeoDataFrame
    """

    # 중심점 좌표 계산
    grid_gdf['centroid'] = grid_gdf.geometry.centroid

    # 각 중심점을 기준으로 300m 버퍼 생성
    buffer_gdf = grid_gdf.copy()
    buffer_gdf['buffer'] = buffer_gdf.geometry.buffer(buffer_distance)

    def find_and_aggregate(column_name, buffer_gdf, grid_gdf):
        results = []
        for index, row in buffer_gdf.iterrows():
            buffer = row['buffer']
            # 버퍼 내 포함된 격자 찾기
            covered_grids = grid_gdf[grid_gdf.geometry.intersects(buffer)]
            # 중심점을 포함한 격자 제외
            covered_grids = covered_grids[covered_grids.index != index]
            # 해당 컬럼의 값 합산
            if not covered_grids.empty:
                summed_value = covered_grids[column_name].sum()
            else:
                summed_value = 0
            results.append(summed_value)
        return results

    # 지정된 컬럼에 대한 합산 결과
    summed_values = find_and_aggregate(column_name, buffer_gdf, grid_gdf)

    # 결과를 원본 데이터와 결합
    buffer_gdf[column_name + '_300M'] = summed_values

    # 결과 GeoDataFrame 반환
    return buffer_gdf


## 거주인구
# 격자내 거주인구수
# 300m 이내 거주인구수
# 격자내 70대이상 거주인구수 ÷ 거주인구수
# 300m이내 70대이상 거주인구수 ÷ 거주인구수
# 격자내 거주인구 ÷ 300m이내 거주인구수


result_grid_gdf.rename(columns={'val':'거주인구'}, inplace=True) #격자내 거주인구수
result_grid_gdf = aggregate_column_excluding_center(result_grid_gdf, '거주인구')  #300m 이내 거주인구수
livepop_70 = gpd.read_file(path_data_out + '//100m격자//연령대별거주인구//(B100)국토통계_인구정보-70대 인구 수(전체)-(격자) 100M_전라남도 광양시_202404//nlsp_021001040.shp',encoding='utf-8')
livepop_80 = gpd.read_file(path_data_out + '//100m격자//연령대별거주인구//(B100)국토통계_인구정보-80대 인구 수(전체)-(격자) 100M_전라남도 광양시_202404//nlsp_021001043.shp',encoding='utf-8')
livepop_90 = gpd.read_file(path_data_out + '//100m격자//연령대별거주인구//(B100)국토통계_인구정보-90대 인구 수(전체)-(격자) 100M_전라남도 광양시_202404//nlsp_021001046.shp',encoding='utf-8')
livepop_100up = gpd.read_file(path_data_out + '//100m격자//연령대별거주인구//(B100)국토통계_인구정보-100세이상 인구 수(전체)-(격자) 100M_전라남도 광양시_202404//nlsp_021001049.shp',encoding='utf-8')
livepop_100up['val'].sum()
dfs = [livepop_70, livepop_80, livepop_90, livepop_100up] # 데이터프레임 병합
merged_df = pd.concat(dfs, ignore_index=True)
merged_df.fillna(0.0, inplace=True)
result = merged_df.groupby('gid')['val'].sum().reset_index()
result['val'].sum()
result_grid_gdf = pd.merge(result_grid_gdf,result,on='gid',how='left').reset_index(drop=True)
result_grid_gdf.rename(columns={'val':'거주인구_70대이상'}, inplace=True)
result_grid_gdf['거주인구_70대이상/거주인구'] = (result_grid_gdf['거주인구_70대이상']/result_grid_gdf['거주인구']).round(2).reset_index(drop=True) #격자내 70대이상 거주인구수 ÷ 거주인구수
result_grid_gdf = aggregate_column_excluding_center(result_grid_gdf, '거주인구_70대이상')  #300m 이내 70대 이상 거주인구수
result_grid_gdf['거주인구_70대이상_300M/거주인구'] = (result_grid_gdf['거주인구_70대이상_300M']/result_grid_gdf['거주인구']).round(2).reset_index(drop=True) #300m이내 70대이상 거주인구수 ÷ 거주인구수
result_grid_gdf['거주인구_거주인구/거주인구_300M'] =  (result_grid_gdf['거주인구']/result_grid_gdf['거주인구_300M']).round(2).reset_index(drop=True) #격자내 거주인구 ÷ 300m이내 거주인구수

#inf or nan 값 0 대체
result_grid_gdf.replace([np.inf, -np.inf], np.nan, inplace=True)
result_grid_gdf.fillna(0.0, inplace=True)
#result_grid_gdf.columns
result_grid_gdf = result_grid_gdf[['gid', '거주인구', '거주인구_300M','거주인구_70대이상', '거주인구_70대이상_300M',
                                   '거주인구_70대이상/거주인구','거주인구_70대이상_300M/거주인구', '거주인구_거주인구/거주인구_300M',
                                   'centroid', 'buffer','geometry']]
# 입지배분 모델 해당 격자 Centroid 데이터 저장
gdf_centroid = gpd.GeoDataFrame(result_grid_gdf[['centroid']], geometry='centroid')
gdf_centroid.to_file(PATH_RESULT3 + '//영향인자//격자_중심점.gpkg', driver='GPKG')

# 입지배분 모델 해당 격자 Buffer 데이터 저장
gdf_buffer = gpd.GeoDataFrame(result_grid_gdf[['buffer']], geometry='buffer')
gdf_buffer.to_file(PATH_RESULT3 + '//영향인자//격자_300M버퍼.gpkg', driver='GPKG')

result_grid_gdf = result_grid_gdf.drop(columns=['centroid', 'buffer'])
# 입지배분 모델 해당 격자 Geometry 데이터 저장
gdf_geometry = gpd.GeoDataFrame(result_grid_gdf, geometry='geometry')
gdf_geometry.to_file(PATH_RESULT3 + '//영향인자//거주인구//거주인구_격자.gpkg', driver='GPKG')

#복사
result1 = gdf_geometry.copy()
result1
## 유동인구
'''
flowpop_2122 = gpd.read_file(PATH_RESULT3 +'//주차장수요//sex유동인구.gpkg',driver='GPKG')
flowpop_2212 = flowpop_2122[flowpop_2122['STD_YM'] == 202212]

flowpop_2212['gid'] = flowpop_2212.sjoin(grid_homepop, how='left', op='within')['gid']
columns = ['MAN_FLOW_POP_CNT_10G',
       'MAN_FLOW_POP_CNT_20G', 'MAN_FLOW_POP_CNT_30G', 'MAN_FLOW_POP_CNT_40G',
       'MAN_FLOW_POP_CNT_50G', 'MAN_FLOW_POP_CNT_60GU',
       'WMAN_FLOW_POP_CNT_10G', 'WMAN_FLOW_POP_CNT_20G',
       'WMAN_FLOW_POP_CNT_30G', 'WMAN_FLOW_POP_CNT_40G',
       'WMAN_FLOW_POP_CNT_50G', 'WMAN_FLOW_POP_CNT_60GU']
flowpop_2212['flow_pop'] = flowpop_2212[columns].sum(axis=1)
flowpop_2212['20_flow'] = flowpop_2212[['MAN_FLOW_POP_CNT_20G','WMAN_FLOW_POP_CNT_20G']].sum(axis=1)
flowpop_2212['30_flow'] = flowpop_2212[['MAN_FLOW_POP_CNT_30G','WMAN_FLOW_POP_CNT_30G']].sum(axis=1)
flowpop_2212['40_flow'] = flowpop_2212[['MAN_FLOW_POP_CNT_40G','WMAN_FLOW_POP_CNT_40G']].sum(axis=1)
flowpop_2212['50_flow'] = flowpop_2212[['MAN_FLOW_POP_CNT_50G','WMAN_FLOW_POP_CNT_50G']].sum(axis=1)
flowpop_2212['60_flow'] = flowpop_2212[['MAN_FLOW_POP_CNT_60GU','WMAN_FLOW_POP_CNT_60GU']].sum(axis=1)

count_per_grid4 = flowpop_2212.groupby('gid')[['flow_pop','20_flow','30_flow','40_flow','50_flow','60_flow']].sum().reset_index()
count_per_grid4 =  count_per_grid4.fillna(0)
count_per_grid5 = count_per_grid4.groupby('gid')[['flow_pop','20_flow','30_flow','40_flow','50_flow','60_flow']].sum().reset_index()
ppt6 = pd.merge(grid_homepop,count_per_grid5,on='gid',how='left')
grid_flowpop=ppt6.fillna(0)
grid_flowpop.to_file(PATH_RESULT3 + '//영향인자//격자유동인구(연령).gpkg', driver='GPKG')
#격자 시간별유동인구(202212기준) 생성
flowpop_time_2122 = gpd.read_file(PATH_RESULT3 +'//주차장수요//time유동인구.gpkg',driver='GPKG')
flowpop_time_2122 = flowpop_time_2122[flowpop_time_2122['STD_YM'] == 202212]
flowpop_time_2122['gid'] = flowpop_time_2122.sjoin(grid_homepop, how='left', op='within')['gid']
flowpop_time_2122.columns
columns = [ 'TMST_00', 'TMST_01', 'TMST_02',
       'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07', 'TMST_08',
       'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13', 'TMST_14',
       'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19', 'TMST_20',
       'TMST_21', 'TMST_22', 'TMST_23']
flowpop_time_2122['flow_pop'] = flowpop_time_2122[columns].sum(axis=1)
count_per_grid4 = flowpop_time_2122.groupby('gid')[['flow_pop','TMST_00', 'TMST_01', 'TMST_02',
       'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07', 'TMST_08',
       'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13', 'TMST_14',
       'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19', 'TMST_20',
       'TMST_21', 'TMST_22', 'TMST_23']].sum().reset_index()
count_per_grid4 =  count_per_grid4.fillna(0)
count_per_grid5 = count_per_grid4.groupby('gid')[['flow_pop','TMST_00', 'TMST_01', 'TMST_02',
       'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07', 'TMST_08',
       'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13', 'TMST_14',
       'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19', 'TMST_20',
       'TMST_21', 'TMST_22', 'TMST_23']].sum().reset_index()
ppt6 = pd.merge(grid_homepop,count_per_grid5,on='gid',how='left')
grid_flowpop_time=ppt6.fillna(0)
grid_flowpop_time.to_file(PATH_RESULT3 + '//영향인자//격자유동인구(시간).gpkg', driver='GPKG')
'''
# 격자내 유동인구수
# 300m 이내 유동인구수
# 격자내 유동인구 ÷ 300m이내 유동인구수
# 2시 유동인구수 ÷ 시간평균유동인구수
# 18시 유동인구수 ÷ 시간평균유동인구수
grid_flowpop_age = gpd.read_file(PATH_RESULT3 + '//영향인자//유동인구//격자유동인구(연령).gpkg', driver='GPKG')
result1 = pd.merge(result1,grid_flowpop_age[['gid','flow_pop']],on='gid',how='left')
result1.rename(columns={'flow_pop':'유동인구'}, inplace=True) # 격자내 유동인구수
result1 = aggregate_column_excluding_center(result1, '유동인구')  # 300m 이내 유동인구수
result1['유동인구_유동인구/유동인구_300M'] = (result1['유동인구']/result1['유동인구_300M']).round(2).reset_index(drop=True) # 격자내 유동인구 ÷ 300m이내 유동인구수

grid_flowpop_time = gpd.read_file(PATH_RESULT3 + '//영향인자//유동인구//격자유동인구(시간).gpkg', driver='GPKG')
grid_flowpop_time = grid_flowpop_time[grid_flowpop_time['gid'].isin(result1['gid'])]

flowpop_time_2122 = gpd.read_file(PATH_RESULT3 +'//주차장수요//time유동인구.gpkg',driver='GPKG')
flowpop_time_2212 = flowpop_time_2122[flowpop_time_2122['STD_YM'] == 202212]
flowpop_time_2212 = flowpop_time_2212.fillna(0.0)
flowpop_time_2212['시간합'] = flowpop_time_2212[[ 'TMST_00', 'TMST_01', 'TMST_02',
       'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07', 'TMST_08',
       'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13', 'TMST_14',
       'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19', 'TMST_20',
       'TMST_21', 'TMST_22', 'TMST_23']].sum(axis=1)


flowpop_time_2212['gid'] = flowpop_time_2212.sjoin(grid_flowpop_time, how='left', op='within')['gid']
flowpop_time_2212_1 = flowpop_time_2212[flowpop_time_2212['gid'].notna()]
flowpop_time_2212_r = flowpop_time_2212_1.groupby('gid')[[ 'TMST_00', 'TMST_01', 'TMST_02',
       'TMST_03', 'TMST_04', 'TMST_05', 'TMST_06', 'TMST_07', 'TMST_08',
       'TMST_09', 'TMST_10', 'TMST_11', 'TMST_12', 'TMST_13', 'TMST_14',
       'TMST_15', 'TMST_16', 'TMST_17', 'TMST_18', 'TMST_19', 'TMST_20',
       'TMST_21', 'TMST_22', 'TMST_23','시간합']].sum().reset_index()

flowpop_time_2212_r.to_pickle(path_result + '//입지분석//사전작업//유동인구//flowpop_time.pkl')


flowpop_time_2212_2 = flowpop_time_2212_1.groupby('gid')[['TMST_02','TMST_18','시간합']].sum().reset_index()
flowpop_time_2212_2['시간평균'] = flowpop_time_2212_2['시간합']/24

flowpop_time_2212_2['유동인구_2시/시간평균'] = (flowpop_time_2212_2['TMST_02']/flowpop_time_2212_2['시간평균']).round(2).reset_index(drop=True) # 2시 유동인구수 ÷ 시간평균유동인구수
flowpop_time_2212_2['유동인구_18시/시간평균'] = (flowpop_time_2212_2['TMST_18']/flowpop_time_2212_2['시간평균']).round(2).reset_index(drop=True) # 2시 유동인구수 ÷ 시간평균유동인구수
flowpop_time_2212_3 = flowpop_time_2212_2[[ 'gid','유동인구_2시/시간평균','유동인구_18시/시간평균']]

result2 = pd.merge(result1,flowpop_time_2212_3,on='gid',how='left')
result2.fillna(0.0, inplace=True)
r_flowpop = result2[['gid', '유동인구', '유동인구_300M','유동인구_유동인구/유동인구_300M', '유동인구_2시/시간평균', '유동인구_18시/시간평균','geometry']]
r_flowpop = gpd.GeoDataFrame(r_flowpop, geometry='geometry')
r_flowpop.to_file(PATH_RESULT3 + '//영향인자//유동인구//유동인구_격자.gpkg', driver='GPKG')

#복사
result3 = result2.copy()
result3 = result3[['gid', '거주인구', '거주인구_300M', '거주인구_70대이상', '거주인구_70대이상_300M',
       '거주인구_70대이상/거주인구', '거주인구_70대이상_300M/거주인구', '거주인구_거주인구/거주인구_300M',
       '유동인구', '유동인구_300M','유동인구_유동인구/유동인구_300M', '유동인구_2시/시간평균', '유동인구_18시/시간평균','geometry']]

## 등록차량대수
'''
pre_car = pd.read_excel(path_data_analyze+ '//등록차량정보//등록차량정보.xlsx')
pre_car_d = pre_car[~pre_car['행정동'].isin(['전라남도 광양시 중마동', '전라남도 광양시 광양읍'])]
# 청크 크기 설정
chunk_size = 9999
# 데이터프레임의 총 행 수
total_rows = len(pre_car_d)
# 데이터프레임을 청크로 나누어 저장
for i in range(0, total_rows, chunk_size):
    # 청크 추출
    chunk = pre_car_d.iloc[i:i + chunk_size]
    # 청크를 파일로 저장
    path = path_data_analyze + '//영향인자//등록차량정보(중마동,광양읍제외)//'
    chunk.to_csv(path +f'chunk_{i // chunk_size + 1}.csv', index=False, encoding='cp949')

dc1 = gpd.read_file(path_data_analyze + '//등록차량정보//등록차량정보(중마동,광양읍제외)//지오코딩후//chunk_1_WGS.shp',encoding='cp949')
dc2 = gpd.read_file(path_data_analyze + '//등록차량정보//등록차량정보(중마동,광양읍제외)//지오코딩후//chunk_2_WGS.shp',encoding='cp949')
dc3 = gpd.read_file(path_data_analyze + '//등록차량정보//등록차량정보(중마동,광양읍제외)//지오코딩후//chunk_3_WGS.shp',encoding='cp949')
dc4 = gpd.read_file(path_data_analyze + '//등록차량정보//등록차량정보(중마동,광양읍제외)//지오코딩후//chunk_4_WGS.shp',encoding='cp949')
'''
# 격자내 등록차량대수
# 300m 이내 등록차량대수
# 격자내 등록차량대수 ÷ 300m이내 등록차량대수
grid_car_1 = gpd.read_file(PATH_RESULT3 + '//주차장수요//특정지역시각화(ppt자료)//등록차량정보격자(중마동,광양읍).gpkg', driver='GPKG')
grid_car_2 =  grid_car_1[(grid_car_1['행정동'] == '광양읍') | (grid_car_1['행정동'] == '중마동')]
result3= pd.merge(result3,grid_car_2[['gid','등록차량수']],on='gid',how='left') # 격자내 등록차량대수
result3 = aggregate_column_excluding_center(result3, '등록차량수')
result3['등록차량수_등록차량/등록차량_300M'] = (result3['등록차량수']/result3['등록차량수_300M']).round(2).reset_index(drop=True) # 격자내 등록차량대수 ÷ 300m이내 등록차량대수

result3.fillna(0.0, inplace=True)
r_car = result3[['gid', '등록차량수', '등록차량수_300M','등록차량수_등록차량/등록차량_300M','geometry']]
r_car = gpd.GeoDataFrame(r_car, geometry='geometry')
r_car.to_file(PATH_RESULT3 + '//영향인자//등록차량대수//등록차량대수_격자.gpkg', driver='GPKG')

#복사
result4 = result3.copy()
result5 = result4[['gid', '거주인구', '거주인구_300M', '거주인구_70대이상', '거주인구_70대이상_300M',
       '거주인구_70대이상/거주인구', '거주인구_70대이상_300M/거주인구', '거주인구_거주인구/거주인구_300M',
       '유동인구', '유동인구_300M','유동인구_유동인구/유동인구_300M', '유동인구_2시/시간평균', '유동인구_18시/시간평균',
       '등록차량수', '등록차량수_300M','등록차량수_등록차량/등록차량_300M','geometry']]


## 주차장
'''
parking_1_point_geo = gpd.read_file(PATH_RESULT3 + '//주차장공급//공영주차장위치좌표.gpkg') #공영주차장 위치좌표
parking_cnt = pd.read_excel(path_middle + '//건축물대장대상목록.xlsx') # 건축물대장 주차장 위치좌표
ppt1_1 = parking_1_point_geo[(parking_1_point_geo['읍면동'] == '광양읍') | (parking_1_point_geo['읍면동'] == '중마동')]
## 중마동생성
parking_cnt_jung = parking_cnt[parking_cnt['ADM_NM'] == '중마동'] # 중마동
parking_cnt_jung.to_excel(path_middle + '//특정지역지오코딩//건축물대장대상목록_중마동.xlsx', index=False)
##중마동 지오코딩
parking_cnt_jung_geo = gpd.read_file(path_middle + '//특정지역지오코딩//지오코딩완료//건축물대장대상목록_중마동_WGS.shp',encoding='cp949')
parking_cnt_jung_geo2 = gpd.GeoDataFrame(parking_cnt_jung_geo, geometry='geometry',crs='epsg:4326')
parking_cnt_jung_geo2 = parking_cnt_jung_geo2.to_crs(epsg=5186)
parking_cnt_jung_geo2['최종주차장'] = parking_cnt_jung_geo2['최종주차장'].astype(int)

parking_cnt_jung_geo2 = parking_cnt_jung_geo2.sort_values(by='최종주차장', ascending=False)
parking_cnt_jung_geo2.to_file(PATH_RESULT3 + '//주차장공급//특정지역시각화(ppt자료)//좌표//건축물대장_중마동주차장.gpkg', driver='GPKG')

## 광양읍생성
parking_cnt_gwang = parking_cnt[parking_cnt['ADM_NM'] == '광양읍'] # 중마동
parking_cnt_gwang.to_excel(path_middle + '//특정지역지오코딩//건축물대장대상목록_광양읍.xlsx', index=False)

## 광양읍 지오코딩
parking_cnt_gwang_geo = gpd.read_file(path_middle + '//특정지역지오코딩//지오코딩완료//건축물대장대상목록_광양읍_WGS.shp',encoding='cp949')
parking_cnt_gwang_geo2 = gpd.GeoDataFrame(parking_cnt_gwang_geo, geometry='geometry',crs='epsg:4326')
parking_cnt_gwang_geo2 = parking_cnt_gwang_geo2.to_crs(epsg=5186)
parking_cnt_gwang_geo2['최종주차장'] = parking_cnt_gwang_geo2['최종주차장'].astype(int)

parking_cnt_gwang_geo2 = parking_cnt_gwang_geo2.sort_values(by='최종주차장', ascending=False)
parking_cnt_gwang_geo2.to_file(PATH_RESULT3 + '//주차장공급//특정지역시각화(ppt자료)//좌표//건축물대장_광양읍주차장.gpkg', driver='GPKG')
'''
# 격자내 공영주차장 여부 O
# 300m내 다른 공영주차장 여부 O
# 300m내 다른 공영주차장 총면수 O
# 가장 가까운 다른 공영주차장까지의 직선거리
# 격자내 부설주차장수 O
# 300m 이내 부설주차장수 O
# 격자내 부설주차장수 ÷ 300m이내 부설주차장수
# 격자내 주택주차장수
# 300m 이내 주택주차장수
# 격자내 주택주차장수 ÷ 300m이내 주택주차장수
# 격자내 주택주차장수 ÷ 부설주차장수
# 300m내 주택주차장수 ÷ 300m내 부설주차장수
# 가장 가까운 다른 공영주차장까지의 직선거리


# 공영주차장
parking_1_point_geo = gpd.read_file(PATH_RESULT3 + '//주차장공급//공영주차장위치좌표.gpkg') #공영주차장 위치좌표
ppt1_1 = parking_1_point_geo[(parking_1_point_geo['읍면동'] == '광양읍') | (parking_1_point_geo['읍면동'] == '중마동')]
parking_public = ppt1_1[['읍면동','주차장 주소','주차면수','geometry']]
parking_public.columns = ['행정동명','주소','공영주차장_주차면수','geometry']
result5['grid_centroid'] = result5.geometry.centroid
parking_public['gid'] = parking_public.sjoin(result5,how='left', op='within')['gid']
#parking_public.rename(columns={'geometry':'public_points'}, inplace=True)


# 부설주차장
org_1 = pd.read_excel(path_data_in + '//주차장최적지 분석(데이터센터)기초자료//건축물대장 현황(3169)//일반건축물 전체.xlsx',skiprows=8)
org_2 = pd.read_excel(path_data_in + '//주차장최적지 분석(데이터센터)기초자료//건축물대장 현황(3169)//총괄표제부 전체.xlsx',skiprows=7)
org_1 = org_1[['대지위치주소','주용도코드명']]
org_2 = org_2[['대지위치주소','주용도코드명']]
org_juso = pd.concat([org_1,org_2],axis=0)
org_juso.rename(columns={'대지위치주소':'대지위치'},inplace=True)
org_juso.drop_duplicates(inplace=True)

parking_cnt_jung_geo2 = gpd.read_file(PATH_RESULT3 + '//주차장공급//특정지역시각화(ppt자료)//좌표//건축물대장_중마동주차장.gpkg', driver='GPKG')
parking_cnt_gwang_geo2 = gpd.read_file(PATH_RESULT3 + '//주차장공급//특정지역시각화(ppt자료)//좌표//건축물대장_광양읍주차장.gpkg', driver='GPKG')

p_jung_1 = pd.merge(parking_cnt_jung_geo2,org_juso,on='대지위치',how='left').reset_index(drop=True)
p_gwang_1 = pd.merge(parking_cnt_gwang_geo2,org_juso,on='대지위치',how='left').reset_index(drop=True)
parking1 = pd.concat([p_jung_1, p_gwang_1], axis=0)
parking_busul = parking1[['행정동명','대지위치','최종주차장','주용도코드명','geometry']]
parking_busul.columns = ['행정동명','주소','부설주차장_주차면수','주용도코드명','geometry']

ppt1_xy = pd.concat([parking_public, parking_busul], axis=0)
ppt1_xy = ppt1_xy[['행정동명', '주소', '공영주차장_주차면수', '부설주차장_주차면수','주용도코드명', 'geometry']]
ppt1_xy['총주차면수'] = ppt1_xy[[ '공영주차장_주차면수', '부설주차장_주차면수']].sum(axis=1)

ppt1_xy = ppt1_xy[ppt1_xy['총주차면수'] != 0.0]
ppt1_xy.to_file(PATH_RESULT3 + '//영향인자//주차장//영향인자_총주차장.gpkg', driver='GPKG') #주차면수 1이상 포인트좌표

# 최종 주차장 격자 생성
parking_r = ppt1_xy.copy()
parking_r['gid'] = parking_r.sjoin(result5, how='left', op='within')['gid']
test = parking_r[parking_r['행정동명'] == 0]
parking_r.loc[parking_r['행정동명'] == 0, '행정동명'] = '중마동'


parking_r['주용도코드명'].unique()
filtered_df = parking_r[parking_r['주용도코드명'].isin(['단독주택', '공동주택'])]
result = filtered_df.groupby('gid')['총주차면수'].sum().reset_index()
result.rename(columns={'총주차면수': '주택주차장_주차면수'}, inplace=True)
parking_r = pd.merge(parking_r,result,on='gid',how='left')

parking_r2 = parking_r.groupby('gid').agg({
    '행정동명': 'first',                   # '행정동명'의 첫 번째 값
    '총주차면수': 'sum',
    '공영주차장_주차면수': 'sum',          # '공영주차장_주차면수'의 합계
    '부설주차장_주차면수': 'sum',         # '부설주차장_주차면수'의 합계
    '주택주차장_주차면수': 'sum'          # '주택주차장_주차면수'의 합계
}).reset_index()


result5 = pd.merge(result5,parking_r2,on='gid',how='left')
result5.fillna(0.0,inplace=True)
result5['공영주차장_여부'] = result5['공영주차장_주차면수'].apply(lambda x: 'O' if x > 0.0 else 'X') # 격자내 공영주차장 여부
result5 = aggregate_column_excluding_center(result5, '공영주차장_주차면수') # 300m내 다른 공영주차장 총면수
result5['공영주차장_주차면수_300M_여부'] = result5['공영주차장_주차면수_300M'].apply(lambda x: 'O' if x > 0.0 else 'X') # 300m내 다른 공영주차장 총면수


result5 = aggregate_column_excluding_center(result5, '부설주차장_주차면수') # 300m 이내 부설주차장수

result5['주택주차장_여부'] = result5['주택주차장_주차면수'].apply(lambda x: 'O' if x > 0.0 else 'X') # 격자내 주택주차장 여부
result5 = aggregate_column_excluding_center(result5, '주택주차장_주차면수') # 300m내 다른 주택주차장 총면수
result5['주택주차장_주차면수_300M_여부'] = result5['주택주차장_주차면수_300M'].apply(lambda x: 'O' if x > 0.0 else 'X') # 300m내 다른 주택차장 총면수

result5['부설주차장/부설주차장_300M'] = (result5['부설주차장_주차면수']/result5['부설주차장_주차면수_300M']).round(2).reset_index(drop=True) #격자내 부설주차장수 ÷ 300m이내 부설주차장수
result5['주택주차장/주택주차장_300M'] = (result5['주택주차장_주차면수']/result5['주택주차장_주차면수_300M']).round(2).reset_index(drop=True) #격자내 주택주차장수 ÷ 300m이내 주택주차장수
result5['주택주차장/부설주차장'] = (result5['주택주차장_주차면수']/result5['부설주차장_주차면수']).round(2).reset_index(drop=True) #격자내 주택주차장수 ÷ 부설주차장수
result5['주택주차장_300M/부설주차장_300M'] = (result5['주택주차장_주차면수_300M']/result5['부설주차장_주차면수_300M']).round(2).reset_index(drop=True) #300m내 주택주차장수 ÷ 300m내 부설주차장수
result5.fillna(0.0, inplace=True)

# 가장 가까운 다른 공영주차장까지의 직선거리
result5_cent = result5.copy()
result5_cent.rename(columns = {'geometry':'org_geometry','grid_centroid':'geometry'},inplace=True)
result5_cent = gpd.GeoDataFrame(result5_cent, geometry='geometry') #격자 중심점
parking_public = gpd.GeoDataFrame(parking_public, geometry='geometry')

distances = []
for idx, grid_row in result5_cent.iterrows():
    grid_gid = grid_row['gid']
    grid_point = grid_row.geometry  # 격자의 중심점 좌표
    parking_outside_grid = parking_public[parking_public['gid'] != grid_gid]
    nearest_parking = parking_outside_grid.geometry.apply(lambda x: grid_point.distance(x)).idxmin()
    nearest_parking_geom = parking_outside_grid.loc[nearest_parking].geometry

    min_distance = grid_point.distance(nearest_parking_geom)
    distances.append(min_distance)
result5_cent['최근접_공영주차장직선거리'] = distances # 4. 계산된 거리를 새로운 컬럼으로 추가

r_parking = result5_cent[['gid','행정동명', '총주차면수', '공영주차장_주차면수', '부설주차장_주차면수', '주택주차장_주차면수',
       '공영주차장_여부', '공영주차장_주차면수_300M','공영주차장_주차면수_300M_여부', '주택주차장_여부', '주택주차장_주차면수_300M',
       '주택주차장_주차면수_300M_여부', '부설주차장_주차면수_300M', '부설주차장/부설주차장_300M',
       '주택주차장/주택주차장_300M', '주택주차장/부설주차장', '주택주차장_300M/부설주차장_300M','최근접_공영주차장직선거리','org_geometry']]
r_parking.rename(columns = {'org_geometry':'geometry'},inplace=True)
r_parking = gpd.GeoDataFrame(r_parking, geometry='geometry')
r_parking.to_file(PATH_RESULT3 + '//영향인자//주차장//주차장_격자.gpkg', driver='GPKG')

#복사
result6 = result5_cent[['gid', '거주인구', '거주인구_300M', '거주인구_70대이상', '거주인구_70대이상_300M','거주인구_70대이상/거주인구', '거주인구_70대이상_300M/거주인구', '거주인구_거주인구/거주인구_300M',
       '유동인구', '유동인구_300M', '유동인구_유동인구/유동인구_300M', '유동인구_2시/시간평균','유동인구_18시/시간평균', '등록차량수', '등록차량수_300M', '등록차량수_등록차량/등록차량_300M',
       '행정동명', '총주차면수', '공영주차장_주차면수', '부설주차장_주차면수','주택주차장_주차면수', '공영주차장_여부', '공영주차장_주차면수_300M','공영주차장_주차면수_300M_여부', '부설주차장_주차면수_300M', '주택주차장_여부', '주택주차장_주차면수_300M',
       '주택주차장_주차면수_300M_여부', '부설주차장/부설주차장_300M', '주택주차장/주택주차장_300M','주택주차장/부설주차장', '주택주차장_300M/부설주차장_300M', '최근접_공영주차장직선거리','org_geometry']]
result6.rename(columns = {'org_geometry':'geometry'},inplace=True)
result6 = gpd.GeoDataFrame(result6, geometry='geometry')

result6.to_file(path_middle + '//영향인자//result_6.gpkg', driver='GPKG')
result6.to_pickle(path_middle + '//영향인자//result_6.pkl')
result6.to_excel(path_middle + '//영향인자//result_6.xlsx')
result6
# 국가관심지점포인트 좌표
'''
버스터미널/정류장(세분류), 의원/병원(소분류), 체인화편의점(세분류), 은행/저축기관(소분류), 공원(소분류), 주민센터
격자내 주요시설여부
300m 이내 유형별 주요시설여부
300m 이내 유형별 주요시설갯수
300m 이내 전체 주요시설갯수
(주민센터만) 주민센터까지 직선거리
'''
result6 = gpd.read_file(path_middle + '//영향인자//result_6.gpkg', driver='GPKG')
poi_point = gpd.read_file(path_data_out + '//국가관심지점//JEOLLANAM_POI//TN_POI//TN_POI_전라남도.shp',encodings='utf-8')
code = pd.read_excel(path_data_out + "//국가관심지점//JEOLLANAM_POI//TN_ASORTCL//TN_ASORTCL.xlsx",sheet_name='Sheet2')

# Index(['NF_ID', 'POI_NM', 'POI_CL_DC', 'ORIGIN_SE', 'REFRN_ID', 'POI_GEOSD','DACOL_DT', 'OBCHG_DT', 'MNENT_NM', 'geometry'],
poi_point_5179 = gpd.GeoDataFrame(poi_point, geometry='geometry',crs='epsg:5179')
poi_point_5186 = poi_point_5179.to_crs(epsg=5186)
# CRS 확인
print(poi_point_5186.crs)
print(grid_100.crs)

# 포인터와 폴리곤의 공간적 조인 (Spatial Join)
points_with_gids = gpd.sjoin(poi_point_5186, grid_100, how='inner', predicate='within')
points_with_gids= points_with_gids[['gid','NF_ID', 'POI_NM', 'POI_CL_DC', 'ORIGIN_SE', 'REFRN_ID', 'POI_GEOSD','DACOL_DT', 'OBCHG_DT', 'MNENT_NM', 'geometry' ]]

code = code.rename(columns = {'실제입력코드' : 'POI_CL_DC'})
code = code.reset_index()
code = code[['대분류', '중분류', '소분류', '세분류', '세세분류','POI_CL_DC']]
code['POI_CL_DC'] = code['POI_CL_DC'].astype('int64')
points_with_gids['POI_CL_DC'] = points_with_gids['POI_CL_DC'].astype('int64')


poi2 = pd.merge(points_with_gids,code, on='POI_CL_DC')
poi2 = poi2[['gid', 'NF_ID', 'POI_NM', 'POI_CL_DC', '대분류','중분류', '소분류', '세분류', '세세분류', 'geometry']]

# 버스터미널/정류장(세분류)
poi_trans = poi2[poi2['세분류'].isin(['버스터미널/정류장'])]
poi_trans['중분류'] = '버스정류장'
poi_trans = poi_trans[['gid','중분류','세분류','POI_NM','geometry']]
poi_trans.columns = ['gid','중분류','소분류','POI_NM','geometry']
poi_trans = poi_trans.drop_duplicates()
poi_trans.head()

# 의원/병원(소분류)
poi_medical = poi2[poi2['소분류'].isin(['병원','의원','종합병원'])]
poi_medical['중분류'] = '의원/병원'
poi_medical = poi_medical[['중분류','소분류','POI_NM','geometry']]
poi_medical = poi_medical.drop_duplicates()
poi_medical.head()

#공원
poi_park = poi2[poi2['세분류'].isin(['근린공원','어린이공원','소공원'])]
poi_park['중분류'] = '공원'
poi_park = poi_park[['중분류','세분류','POI_NM','geometry']]
poi_park.columns = ['중분류','소분류','POI_NM','geometry']
poi_park = poi_park.drop_duplicates()
poi_park.head()

#편의점
poi_com = poi2[(poi2['세분류'].isin(['체인화편의점']))]
poi_com['중분류'] = '편의점'
poi_com = poi_com[['중분류','세분류','POI_NM','geometry']]
poi_com.columns = ['중분류','소분류','POI_NM','geometry']
poi_com = poi_com.drop_duplicates()
poi_com.tail()

#금융
poi_com = poi2[(poi2['소분류'].isin(['은행및저축기관']))]
poi_com['중분류'] = '은행/저축기관'
poi_com = poi_com[['중분류','세분류','POI_NM','geometry']]
poi_com.columns = ['중분류','소분류','POI_NM','geometry']
poi_com = poi_com.drop_duplicates()
poi_com.tail()

poi_gw = pd.concat([poi_trans,poi_medical,poi_park,poi_com],axis=0)
points_with_gid2 = gpd.sjoin(poi_gw, grid_100, how='inner', predicate='within')

points_with_gid2= points_with_gid2[['gid_right','중분류', '소분류', 'POI_NM', 'geometry' ]]
points_with_gid2.rename(columns={'gid_right':'gid'},inplace=True)
points_with_gid2.to_file(PATH_RESULT3 + "//영향인자//주요시설//국가관심시설위치.gpkg", driver='GPKG')

#주민센터
center = gpd.read_file(path_data_out + '//국가관심지점//주민센터//주민센터위치_WGS.shp',encoding='cp949')
center_4326 = gpd.GeoDataFrame(center, geometry='geometry',crs='epsg:4326')
center_5186 = center_4326.to_crs(epsg=5186)
center_5186 = center_5186[['주민센터명', '주소','geometry']]
center_5186_gid = gpd.sjoin(center_5186, grid_100, how='inner', predicate='within')
center_5186_gid['중분류'] = '주민센터'
center_5186_gid = center_5186_gid[['gid','중분류','주민센터명', '주소','geometry']]
center_5186_gid.to_file(PATH_RESULT3 + "//영향인자//주요시설//주민센터위치.gpkg", driver='GPKG')

#최종 광양시 주요시설 위치 데이터 완성
poi_gid = points_with_gid2[['gid','중분류','geometry']]
center_gid = center_5186_gid[['gid','중분류','geometry']]
poi_gw2 = pd.concat([poi_gid,center_gid],axis=0)
poi_gw2.to_file(PATH_RESULT3 + "//영향인자//주요시설//최종주요시설위치.gpkg", driver='GPKG') #2279

poi_gw3= poi_gw2.groupby(['gid', '중분류']).size().reset_index(name='시설개수')
poi_gw3_pivot = poi_gw3.pivot_table(
    index='gid',              # 'gid'를 인덱스로 사용
    columns='중분류',          # '중분류' 값들을 컬럼으로 변환
    values='시설개수',         # '시설개수'를 값으로 사용
    aggfunc='sum',            # 중복된 값이 있을 경우 합계로 처리
    fill_value=0              # NaN 값을 0으로 채우기
).reset_index()
poi_gw3_pivot.columns = ['gid','공원', '버스정류장', '은행/저축기관', '의원/병원', '주민센터']

result7 = result6.copy()
result7 = pd.merge(result7,poi_gw3_pivot,on='gid',how='left').reset_index(drop=True)
result7.fillna(0,inplace=True)
result7['전체주요시설'] = result7[['공원', '버스정류장', '은행/저축기관', '의원/병원', '주민센터']].sum(axis=1)


result7['버스정류장_여부'] = result7['버스정류장'].apply(lambda x: 'O' if x > 0.0 else 'X') # 격자내 공영주차장 여부
result7 = aggregate_column_excluding_center(result7, '버스정류장') # 300m내 다른 공영주차장 총면수
result7['버스정류장_300M_여부'] = result7['버스정류장_300M'].apply(lambda x: 'O' if x > 0.0 else 'X') # 300m내 다른 공영주차장 총면수


result7['공원_여부'] = result7['공원'].apply(lambda x: 'O' if x > 0.0 else 'X') # 격자내 공영주차장 여부
result7 = aggregate_column_excluding_center(result7, '공원') # 300m내 다른 공영주차장 총면수
result7['공원_300M_여부'] = result7['공원_300M'].apply(lambda x: 'O' if x > 0.0 else 'X') # 300m내 다른 공영주차장 총면수


result7['은행/저축기관_여부'] = result7['은행/저축기관'].apply(lambda x: 'O' if x > 0.0 else 'X') # 격자내 공영주차장 여부
result7 = aggregate_column_excluding_center(result7, '은행/저축기관') # 300m내 다른 공영주차장 총면수
result7['은행/저축기관_300M_여부'] = result7['은행/저축기관_300M'].apply(lambda x: 'O' if x > 0.0 else 'X') # 300m내 다른 공영주차장 총면수


result7['의원/병원_여부'] = result7['의원/병원'].apply(lambda x: 'O' if x > 0.0 else 'X') # 격자내 공영주차장 여부
result7 = aggregate_column_excluding_center(result7, '의원/병원') # 300m내 다른 공영주차장 총면수
result7['의원/병원_300M_여부'] = result7['의원/병원_300M'].apply(lambda x: 'O' if x > 0.0 else 'X') # 300m내 다른 공영주차장 총면수


result7['주민센터_여부'] = result7['주민센터'].apply(lambda x: 'O' if x > 0.0 else 'X') # 격자내 공영주차장 여부
result7 = aggregate_column_excluding_center(result7, '주민센터') # 300m내 다른 공영주차장 총면수
result7['주민센터_300M_여부'] = result7['주민센터_300M'].apply(lambda x: 'O' if x > 0.0 else 'X') # 300m내 다른 공영주차장 총면수


result7['전체주요시설_여부'] = result7['전체주요시설'].apply(lambda x: 'O' if x > 0.0 else 'X') # 격자내 공영주차장 여부
result7 = aggregate_column_excluding_center(result7, '전체주요시설') # 300m내 다른 공영주차장 총면수
result7['전체주요시설_300M_여부'] = result7['전체주요시설_300M'].apply(lambda x: 'O' if x > 0.0 else 'X') # 300m내 다른 공영주차장 총면수


# 가장 가까운 주민센터까지의 거리
result8 = result7.copy()
result8_cent = result8.copy()
result8_cent['grid_centroid'] = result8_cent.geometry.centroid
result8_cent.rename(columns = {'geometry':'org_geometry','grid_centroid':'geometry'},inplace=True)
result8_cent = gpd.GeoDataFrame(result5_cent, geometry='geometry') #격자 중심점
# center_5186_gid 주민센터 위치

distances = []
for idx, grid_row in result8_cent.iterrows():
    grid_gid = grid_row['gid']
    grid_point = grid_row.geometry  # 격자의 중심점 좌표
    parking_outside_grid = center_5186_gid[center_5186_gid['gid'] != grid_gid]
    nearest_parking = parking_outside_grid.geometry.apply(lambda x: grid_point.distance(x)).idxmin()
    nearest_parking_geom = parking_outside_grid.loc[nearest_parking].geometry

    min_distance = grid_point.distance(nearest_parking_geom)
    distances.append(min_distance)
result8_cent['최근접_주민센터직선거리'] = distances # 4. 계산된 거리를 새로운 컬럼으로 추가


result8_fac = result8_cent[['gid', '공원', '버스정류장', '은행/저축기관', '의원/병원',
       '주민센터', '전체주요시설', '버스정류장_여부', '버스정류장_300M','버스정류장_300M_여부',
       '공원_여부', '공원_300M', '공원_300M_여부', '은행/저축기관_여부',
       '은행/저축기관_300M', '은행/저축기관_300M_여부', '의원/병원_여부', '의원/병원_300M',
       '의원/병원_300M_여부', '주민센터_여부', '주민센터_300M', '주민센터_300M_여부', '전체주요시설_여부',
       '전체주요시설_300M', '전체주요시설_300M_여부', '최근접_주민센터직선거리','org_geometry']]
result8_fac.rename(columns = {'org_geometry':'geometry'},inplace=True)
result8_fac = gpd.GeoDataFrame(result8_fac, geometry='geometry')
result8_fac.to_file(PATH_RESULT3 + '//영향인자//주요시설//주요시설_격자.gpkg', driver='GPKG')

#복사
result9 = result8_cent[['gid', '거주인구', '거주인구_300M', '거주인구_70대이상', '거주인구_70대이상_300M',
       '거주인구_70대이상/거주인구', '거주인구_70대이상_300M/거주인구', '거주인구_거주인구/거주인구_300M',
       '유동인구', '유동인구_300M', '유동인구_유동인구/유동인구_300M', '유동인구_2시/시간평균',
       '유동인구_18시/시간평균', '등록차량수', '등록차량수_300M', '등록차량수_등록차량/등록차량_300M', '행정동명',
       '총주차면수', '공영주차장_주차면수', '부설주차장_주차면수', '주택주차장_주차면수', '공영주차장_여부',
       '공영주차장_주차면수_300M', '공영주차장_주차면수_300M_여부', '부설주차장_주차면수_300M', '주택주차장_여부',
       '주택주차장_주차면수_300M', '주택주차장_주차면수_300M_여부', '부설주차장/부설주차장_300M',
       '주택주차장/주택주차장_300M', '주택주차장/부설주차장', '주택주차장_300M/부설주차장_300M',
       '최근접_공영주차장직선거리',  '공원', '버스정류장', '은행/저축기관', '의원/병원',
       '주민센터', '전체주요시설', '버스정류장_여부', '버스정류장_300M','버스정류장_300M_여부',
       '공원_여부', '공원_300M', '공원_300M_여부', '은행/저축기관_여부','은행/저축기관_300M',
       '은행/저축기관_300M_여부', '의원/병원_여부', '의원/병원_300M','의원/병원_300M_여부',
       '주민센터_여부', '주민센터_300M', '주민센터_300M_여부', '전체주요시설_여부',
       '전체주요시설_300M', '전체주요시설_300M_여부', '최근접_주민센터직선거리','org_geometry']]
result9.rename(columns = {'org_geometry':'geometry'},inplace=True)
result9 = gpd.GeoDataFrame(result9, geometry='geometry')

result9.to_file(path_middle + '//영향인자//result9.gpkg', driver='GPKG')
result9.to_pickle(path_middle + '//영향인자//result9.pkl')
result9.to_excel(path_middle + '//영향인자//result9.xlsx')

#복합
#(격자,300m) 유동/거주, 차량/거주, 차량/유동, 주차장/거주, 주차장/유동, 주차장/차량

result10 = result9.copy()
result10['복합_유동인구/거주인구'] = (result10['유동인구']/result10['거주인구']).round(2).reset_index(drop=True)
result10['복합_등록차량수/거주인구'] = (result10['등록차량수']/result10['거주인구']).round(2).reset_index(drop=True)
result10['복합_등록차량수/유동인구'] = (result10['등록차량수']/result10['유동인구']).round(2).reset_index(drop=True)
result10['복합_총주차면수/거주인구'] = (result10['총주차면수']/result10['거주인구']).round(2).reset_index(drop=True)
result10['복합_총주차면수/유동인구'] = (result10['총주차면수']/result10['유동인구']).round(2).reset_index(drop=True)
result10['복합_총주차면수/등록차량수'] = (result10['유동인구']/result10['등록차량수']).round(2).reset_index(drop=True)

#300M
result10 = aggregate_column_excluding_center(result10, '총주차면수') # 300m내 다른 공영주차장 총면수
result10['복합_유동인구/거주인구_300M'] = (result10['유동인구_300M']/result10['거주인구_300M']).round(2).reset_index(drop=True)
result10['복합_등록차량수/거주인구_300M'] = (result10['등록차량수_300M']/result10['거주인구_300M']).round(2).reset_index(drop=True)
result10['복합_등록차량수/유동인구_300M'] = (result10['등록차량수_300M']/result10['유동인구_300M']).round(2).reset_index(drop=True)
result10['복합_총주차면수/거주인구_300M'] = (result10['총주차면수_300M']/result10['거주인구_300M']).round(2).reset_index(drop=True)
result10['복합_총주차면수/유동인구_300M'] = (result10['총주차면수_300M']/result10['유동인구_300M']).round(2).reset_index(drop=True)
result10['복합_총주차면수/등록차량수_300M'] = (result10['유동인구_300M']/result10['등록차량수_300M']).round(2).reset_index(drop=True)
result10 = result10[['gid','거주인구', '거주인구_300M', '거주인구_70대이상', '거주인구_70대이상_300M',
       '거주인구_70대이상/거주인구', '거주인구_70대이상_300M/거주인구', '거주인구_거주인구/거주인구_300M',
       '유동인구', '유동인구_300M', '유동인구_유동인구/유동인구_300M', '유동인구_2시/시간평균',
       '유동인구_18시/시간평균', '등록차량수', '등록차량수_300M', '등록차량수_등록차량/등록차량_300M',
       '총주차면수','총주차면수_300M','공영주차장_주차면수', '부설주차장_주차면수', '주택주차장_주차면수', '공영주차장_여부',
       '공영주차장_주차면수_300M', '공영주차장_주차면수_300M_여부', '부설주차장_주차면수_300M', '주택주차장_여부',
       '주택주차장_주차면수_300M', '주택주차장_주차면수_300M_여부', '부설주차장/부설주차장_300M',
       '주택주차장/주택주차장_300M', '주택주차장/부설주차장', '주택주차장_300M/부설주차장_300M',
       '최근접_공영주차장직선거리', '공원', '버스정류장', '은행/저축기관', '의원/병원', '주민센터', '전체주요시설',
       '버스정류장_여부', '버스정류장_300M', '버스정류장_300M_여부', '공원_여부', '공원_300M',
       '공원_300M_여부', '은행/저축기관_여부', '은행/저축기관_300M', '은행/저축기관_300M_여부',
       '의원/병원_여부', '의원/병원_300M', '의원/병원_300M_여부', '주민센터_여부', '주민센터_300M',
       '주민센터_300M_여부', '전체주요시설_여부', '전체주요시설_300M', '전체주요시설_300M_여부',
       '최근접_주민센터직선거리', '복합_유동인구/거주인구', '복합_등록차량수/거주인구','복합_등록차량수/유동인구',
       '복합_총주차면수/거주인구', '복합_총주차면수/유동인구', '복합_총주차면수/등록차량수',
       '복합_유동인구/거주인구_300M', '복합_등록차량수/거주인구_300M', '복합_등록차량수/유동인구_300M',
       '복합_총주차면수/거주인구_300M','복합_총주차면수/유동인구_300M', '복합_총주차면수/등록차량수_300M', 'geometry']]

result10.fillna(0.0, inplace=True)
result11 = result10[['gid','복합_유동인구/거주인구', '복합_등록차량수/거주인구','복합_등록차량수/유동인구',
       '복합_총주차면수/거주인구', '복합_총주차면수/유동인구', '복합_총주차면수/등록차량수',
       '복합_유동인구/거주인구_300M', '복합_등록차량수/거주인구_300M', '복합_등록차량수/유동인구_300M',
       '복합_총주차면수/거주인구_300M','복합_총주차면수/유동인구_300M', '복합_총주차면수/등록차량수_300M', 'geometry']]
result11.to_file(PATH_RESULT3 + '//영향인자//복합//주요시설_복합.gpkg', driver='GPKG')

result10.to_file(path_middle + '//영향인자//result10_전체최종.gpkg', driver='GPKG')
result10.to_pickle(path_middle + '//영향인자//result10_전체최종.pkl')
result10.to_excel(path_middle + '//영향인자//result10_전체최종.xlsx')

factor_result = result10.copy() #최종
factor_result = factor_result[['gid','공영주차장_여부','거주인구','거주인구_300M','거주인구_70대이상/거주인구', '거주인구_70대이상_300M/거주인구', '거주인구_거주인구/거주인구_300M',
                               '유동인구', '유동인구_300M', '유동인구_유동인구/유동인구_300M', '유동인구_2시/시간평균','유동인구_18시/시간평균',
                               '등록차량수', '등록차량수_300M', '등록차량수_등록차량/등록차량_300M',
                               '부설주차장_주차면수','부설주차장_주차면수_300M','부설주차장/부설주차장_300M','주택주차장_주차면수','주택주차장_주차면수_300M',
                               '주택주차장/주택주차장_300M','주택주차장/부설주차장','주택주차장_300M/부설주차장_300M',
                               '공영주차장_주차면수_300M_여부','공영주차장_주차면수_300M','최근접_공영주차장직선거리',
                               '공원', '버스정류장', '은행/저축기관', '의원/병원', '주민센터',
                               '버스정류장_여부','공원_여부','의원/병원_여부','은행/저축기관_여부','주민센터_여부',
                               '버스정류장_300M_여부','공원_300M_여부','은행/저축기관_300M_여부','의원/병원_300M_여부','주민센터_300M_여부',
                               '버스정류장_300M','공원_300M','은행/저축기관_300M','의원/병원_300M','주민센터_300M', '전체주요시설_300M','최근접_주민센터직선거리',
                               '복합_유동인구/거주인구', '복합_등록차량수/거주인구', '복합_등록차량수/유동인구',
                               '복합_총주차면수/거주인구', '복합_총주차면수/유동인구', '복합_총주차면수/등록차량수',
                               '복합_유동인구/거주인구_300M', '복합_등록차량수/거주인구_300M', '복합_등록차량수/유동인구_300M',
                               '복합_총주차면수/거주인구_300M', '복합_총주차면수/유동인구_300M', '복합_총주차면수/등록차량수_300M','geometry']]

factor_result.to_file(PATH_RESULT3 + '//영향인자//격자_최종영향인자SET.gpkg', driver='GPKG')
factor_result.to_pickle(path_result + '//영향인자//factor_result.pkl')
factor_result.to_excel(path_result + '//영향인자//factor_result.xlsx')

#########################################################################################################################

# pkl 변환
factor_parking = gpd.read_file(PATH_RESULT3 + '//영향인자//주차장//주차장_격자.gpkg', driver='GPKG')
factor_car = gpd.read_file(PATH_RESULT3 + '//영향인자//등록차량대수//등록차량대수_격자.gpkg', driver='GPKG')
factor_parking_patial = factor_parking[['gid', '총주차면수','공영주차장_주차면수', '부설주차장_주차면수', '주택주차장_주차면수']]
factor_car_patial = factor_car[['gid', '등록차량수']]
factor_patial = pd.merge(factor_parking_patial, factor_car_patial, on='gid')
#factor_patial.to_pickle(path_middle + '//영향인자//factor_part.pkl')

#광양읍 ,중마동 분리
grid_gw = gpd.read_file(path_middle + '//영향인자//광양읍격자.gpkg', driver='GPKG')
grid_jm = gpd.read_file(path_middle + '//영향인자//중마동격자.gpkg', driver='GPKG')
grid_gw['HJD_NM'] = '광양읍'
grid_jm['HJD_NM'] = '중마동'
grid_gw = grid_gw[['gid','HJD_NM']]
grid_jm = grid_jm[['gid','HJD_NM']]
grid_hjd = pd.concat([grid_gw, grid_jm], axis=0)

factor_patial_hjd = pd.merge(factor_patial, grid_hjd, on='gid')
factor_patial_gy = factor_patial_hjd[factor_patial_hjd['HJD_NM'] == '광양읍']
factor_patial_jm = factor_patial_hjd[factor_patial_hjd['HJD_NM'] == '중마동']

factor_patial_gy.to_pickle(path_middle + '//영향인자//factor_patial_gy.pkl')
factor_patial_jm.to_pickle(path_middle + '//영향인자//factor_patial_jm.pkl')








