#%%
###########################################################
# 과 제 명: A10_광양시_주차장입지분석
# 파 일 명: 입지배분모델
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

path_result = os.path.join(config_info["PATH_DATA_OUTPUT_RESULT"])
PATH_RESULT3 = os.path.join(config_info["PATH_RESULT"],'QGIS_DATA')
###################################################################################################


import numpy as np
import geopandas as gpd
import pandas as pd
import warnings
from pulp import *
import time
import datetime
warnings.filterwarnings(action='ignore')

# 시간 출력
def get_time():
    return datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')

# ---------------------------------------------------------------------------------------------
# 모델 p-median 모델
# INPUT    DataFrame      factor_df     격자별 영향인자 데이터프레임
#          GeoDataFrame   geo_df        분석 대상 격자 지오데이터프레임
#          str            factor_nm     영향인자 변수 이름
#          int            p             선정할 입지 갯수
#          str            out_file_nm   결과 파일 이름
# OUTPUT   DataFrame      final_result  선택된 입지 정보 데이터프레임
# ---------------------------------------------------------------------------------------------

def p_median (factor_df,geo_df,factor_nm,p,out_file_nm) :

    total_start_time = time.time()

    ###1-2-1 영향인자 지오데이터프레임 생성
    factor = factor_df.copy()
    factor = factor[['gid',factor_nm]]
    factor[factor_nm] = factor[factor_nm].astype('float')
    geo_df = geo_df[['gid','geometry']]
    factor_geo = pd.merge(factor,geo_df,on='gid')
    factor_geo = gpd.GeoDataFrame(factor_geo,geometry= 'geometry')

    #영향인자, 격자ID
    factor_final = factor_geo[factor_nm].to_list()
    gid_final = factor_geo['gid'].to_list()
    location = gid_final
    a = dict(zip(location,factor_final))

    ###1-2-2 거리 매트릭스 생성
    #거리매트릭스 구하기
    result = pd.DataFrame()
    for j in range(0,len(factor_geo)):
        gid_1 = factor_geo.loc[j,'gid']
        geometry_1 = factor_geo.loc[j,'geometry']
        temp = pd.DataFrame()
        for i in range(0,len(factor_geo)):
            if i == len(factor_geo)-1:
                gid_2 = factor_geo.loc[i,'gid']
                geometry_2 = factor_geo.loc[i,'geometry']
                distance = geometry_1.distance(geometry_2)
                cal = pd.DataFrame({'gid':[gid_2],'distance': [distance]})
                temp = pd.concat([temp,cal],axis=0)
                df = temp.copy()
                df = pd.DataFrame(df['distance'])
                df.columns = [gid_1]
                result = pd.concat([result,df],axis=1)
            else:
                gid_2 = factor_geo.loc[i,'gid']
                geometry_2 = factor_geo.loc[i,'geometry']
                distance = geometry_1.distance(geometry_2)
                cal = pd.DataFrame({'gid':[gid_2],'distance':[distance]})
                temp = pd.concat([temp,cal],axis=0)
    print('[' + get_time() + ']' + "1. 거리 매트릭스 생성 완료")

    ###1-2-3 선형계획법 해 구하기
    D={}

    for i in range(0,len(location)):
        D[i] = dict(zip(location[i:],[dict(zip(location, result.iloc[i,0:].to_list()))]))

    merged = {}
    for i in range(0,len(location)):
        merged = {**merged, **D[i]}


    X = LpVariable.dicts('X_%s_%s',(location,location),
        cat= 'Binary',
        lowBound=0,
        upBound=1)

    prob = LpProblem('P Median', LpMinimize)
    prob += sum(sum(a[i]*merged[i][j]*X[i][j] for j in location)for i in location)
    prob += sum(X[i][i] for i in location) == p

    for i in location : prob += sum(X[i][j] for j in location) ==1
    for i in location:
        for j in location : prob+= X[i][j] <= X[j][j]

    prob.writeLP("p-median_common.lp")
    prob.solve()

    print('[' + get_time() + ']' + "2. 최적 입지 선정 완료")

    ###1-2-4 결과 데이터프레임 생성
    lp_result = pd.DataFrame()
    for v in prob.variables():
          if v.varValue == 1 :
              temp = pd.DataFrame({'result':[v.name],'distance':[v.varValue]})
              lp_result = pd.concat([lp_result,temp])
    lp_result['gid_1'] = lp_result['result'].apply(lambda x : x.split('_')[1])
    lp_result['gid_2'] = lp_result['result'].apply(lambda x : x.split('_')[2])

    result = pd.DataFrame({'final_gid': lp_result['gid_2'].unique()})
    final_result = pd.merge(result,factor_geo,left_on='final_gid',right_on='gid',how='inner')
    final_result = final_result[['gid',factor_nm,'geometry']]
    final_result.to_csv(path_result + '//입지분석결과//'+out_file_nm+'.csv',encoding= 'cp949',index=False)
    print('[' + get_time() + ']'+"3. 결과 저장 완료")

    print('소요시간 : ', str((time.time() - total_start_time) / 60)[:7] + ' 분 소요')

    return final_result