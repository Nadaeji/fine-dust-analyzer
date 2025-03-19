# train.py
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

# 디렉토리 생성
if not os.path.exists('models'):
    os.makedirs('models')

# 1. 데이터 로드
data = pd.read_csv("../data/pm25_pm10_merged_wind.csv")

# 2. 데이터 전처리
def get_season(month):
    if month in [3, 4, 5]:
        return "봄"
    elif month in [6, 7, 8]:
        return "여름"
    elif month in [9, 10, 11]:
        return "가을"
    else:
        return "겨울"

data['Date'] = pd.to_datetime(data['Date'])
data['Month'] = data['Date'].dt.month
data['Season'] = data['Month'].apply(get_season)

# 2018년 이전 데이터 제거
data = data[data['Date'].dt.year >= 2018]

# PM2.5 및 PM10 값이 0보다 큰 데이터만 사용
data = data[(data['PM2.5 (µg/m³)'] > 0) & (data['PM10 (µg/m³)'] > 0)]
data['Wind_X'] = data['Wind Speed (m/s)'] * np.cos(np.radians(data['Wind Direction (degrees)']))
data['Wind_Y'] = data['Wind Speed (m/s)'] * np.sin(np.radians(data['Wind Direction (degrees)']))

# 중국 도시와 주변국 도시 정의 (영어)
china_cities = ['Beijing', 'Shanghai', 'Guangzhou', 'Chongqing', 'Wuhan', 
                'Nanjing', 'Hangzhou', 'Chengdu']
nearby_cities = ['Seoul', 'Tokyo', 'Delhi', 'Bangkok', 'Busan', 'Daegu', 'Osaka', 
                 'Sapporo', 'Fukuoka', 'Kyoto', 'Almaty', 'Bishkek', 'Dushanbe', 
                 'Kathmandu', 'Yangon', 'Guwahati', 'Ulaanbaatar', 'Irkutsk']

# 중국 데이터 집계
china_data = data[data['City'].isin(china_cities)].groupby(['Date', 'Season']).agg({
    'PM2.5 (µg/m³)': 'mean', 'PM10 (µg/m³)': 'mean', 'Wind_X': 'mean', 'Wind_Y': 'mean'
}).reset_index()

# 계절별 평균 풍속과 바람 방향 계산 (중국 데이터 기준)
season_wind = data[data['City'].isin(china_cities)].groupby('Season').agg({
    'Wind_X': 'mean',
    'Wind_Y': 'mean'
}).to_dict()

# 주변국 데이터 피벗 (PM2.5와 PM10 모두 준비)
nearby_data_pm25 = data[data['City'].isin(nearby_cities)].pivot(
    index=['Date', 'Season'], columns='City', values='PM2.5 (µg/m³)'
).reset_index()
nearby_data_pm10 = data[data['City'].isin(nearby_cities)].pivot(
    index=['Date', 'Season'], columns='City', values='PM10 (µg/m³)'
).reset_index()

# 데이터 병합 (PM2.5와 PM10 각각)
merged_data_pm25 = pd.merge(china_data, nearby_data_pm25, on=['Date', 'Season'], how='inner')
merged_data_pm10 = pd.merge(china_data, nearby_data_pm10, on=['Date', 'Season'], how='inner')

# 3. SVR 모델 학습 및 하이퍼파라미터 튜닝 (GridSearchCV 사용, CPU 기반)
seasons = ['봄', '여름', '가을', '겨울']
scaler_pm25 = StandardScaler()  # PM2.5용 StandardScaler
scaler_pm10 = StandardScaler()  # PM10용 StandardScaler
svr_models_pm25 = {}
svr_models_pm10 = {}
evaluation_scores_pm25 = {}
evaluation_scores_pm10 = {}

# 하이퍼파라미터 그리드 정의
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 0.5, 1]
}

# PM2.5 모델 학습
for season in seasons:
    season_data = merged_data_pm25[merged_data_pm25['Season'] == season].dropna()
    if len(season_data) < 10:
        continue
    
    X = season_data[['PM2.5 (µg/m³)', 'Wind_X', 'Wind_Y']]
    X_scaled = scaler_pm25.fit_transform(X)
    
    svr_models_pm25[season] = {}
    evaluation_scores_pm25[season] = {}
    for city in nearby_cities:
        if city not in season_data.columns:
            continue
        
        y = season_data[city].dropna()
        if len(y) < 10:
            continue
        
        X_city = X_scaled[:len(y)]
        X_train, X_test, y_train, y_test = train_test_split(X_city, y, test_size=0.2, random_state=42)
        
        # GridSearchCV로 하이퍼파라미터 튜닝
        svr = SVR(kernel='rbf')
        grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # 최적 모델로 예측
        best_svr = grid_search.best_estimator_
        y_pred = best_svr.predict(X_test)
        
        # 평가 점수 계산
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evaluation_scores_pm25[season][city] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}
        
        # 최적 모델 저장
        svr_models_pm25[season][city] = best_svr

# PM10 모델 학습
for season in seasons:
    season_data = merged_data_pm10[merged_data_pm10['Season'] == season].dropna()
    if len(season_data) < 10:
        continue
    
    X = season_data[['PM10 (µg/m³)', 'Wind_X', 'Wind_Y']]
    X_scaled = scaler_pm10.fit_transform(X)
    
    svr_models_pm10[season] = {}
    evaluation_scores_pm10[season] = {}
    for city in nearby_cities:
        if city not in season_data.columns:
            continue
        
        y = season_data[city].dropna()
        if len(y) < 10:
            continue
        
        X_city = X_scaled[:len(y)]
        X_train, X_test, y_train, y_test = train_test_split(X_city, y, test_size=0.2, random_state=42)
        
        # GridSearchCV로 하이퍼파라미터 튜닝
        svr = SVR(kernel='rbf')
        grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # 최적 모델로 예측
        best_svr = grid_search.best_estimator_
        y_pred = best_svr.predict(X_test)
        
        # 평가 점수 계산
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evaluation_scores_pm10[season][city] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}
        
        # 최적 모델 저장
        svr_models_pm10[season][city] = best_svr

# 4. 모델 및 데이터 저장
# StandardScaler 저장 (PM2.5와 PM10 각각)
joblib.dump(scaler_pm25, '../models/svr/scaler_pm25.pkl')
joblib.dump(scaler_pm10, '../models/svr/scaler_pm10.pkl')

# season_wind 저장
joblib.dump(season_wind, '../models/svr/season_wind.pkl')

# SVR 모델 저장 (PM2.5)
for season in svr_models_pm25:
    for city in svr_models_pm25[season]:
        joblib.dump(svr_models_pm25[season][city], f'../models/svr/svr_pm25_{season}_{city}.pkl')

# SVR 모델 저장 (PM10)
for season in svr_models_pm10:
    for city in svr_models_pm10[season]:
        joblib.dump(svr_models_pm10[season][city], f'../models/svr/svr_pm10_{season}_{city}.pkl')

# 평가 점수 저장 (PM2.5)
joblib.dump(evaluation_scores_pm25, '../models/svr/evaluation_scores_pm25.pkl')

# 평가 점수 저장 (PM10)
joblib.dump(evaluation_scores_pm10, '../models/svr/evaluation_scores_pm10.pkl')

print("모델 학습 및 저장 완료!")