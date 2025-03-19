import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

# 디렉토리 생성
if not os.path.exists('../models/rf'):
    os.makedirs('../models/rf')

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

# 군집 등급 매핑 딕셔너리
cluster_labels_pm25 = {
    0: "좋음 (PM2.5 ≤ 10 µg/m³)",
    1: "보통 (10 < PM2.5 ≤ 25 µg/m³)",
    2: "나쁨 (25 < PM2.5 ≤ 50 µg/m³)",
    3: "매우 나쁨 (PM2.5 > 50 µg/m³)"
}

cluster_labels_pm10 = {
    0: "좋음 (PM10 ≤ 30 µg/m³)",
    1: "보통 (30 < PM10 ≤ 80 µg/m³)",
    2: "나쁨 (80 < PM10 ≤ 150 µg/m³)",
    3: "매우 나쁨 (PM10 > 150 µg/m³)"
}

# 3. KMeans 군집화 및 RandomForestRegressor 모델 학습
seasons = ['봄', '여름', '가을', '겨울']
scaler_pm25 = StandardScaler()  # PM2.5용 StandardScaler
scaler_pm10 = StandardScaler()  # PM10용 StandardScaler
kmeans_pm25 = KMeans(n_clusters=4, random_state=42)
kmeans_pm10 = KMeans(n_clusters=4, random_state=42)
rf_models_pm25 = {}
rf_models_pm10 = {}
evaluation_scores_pm25 = {}
evaluation_scores_pm10 = {}

# 하이퍼파라미터 탐색 범위 정의
param_dist = {
    'n_estimators': [50, 100, 200, 300],          # 트리 개수
    'max_depth': [5, 10, 20, 30, None],           # 최대 깊이
    'min_samples_split': [2, 5, 10, 20],          # 분할 최소 샘플 수
    'min_samples_leaf': [1, 2, 5, 10],            # 리프 최소 샘플 수
    'max_features': ['auto', 'sqrt', 'log2']      # 피처 고려 비율
}

# PM2.5 모델 학습
for season in seasons:
    season_data = merged_data_pm25[merged_data_pm25['Season'] == season].dropna()
    if len(season_data) < 10:
        continue
    
    X = season_data[['PM2.5 (µg/m³)', 'Wind_X', 'Wind_Y']]
    X_scaled = scaler_pm25.fit_transform(X)
    
    # KMeans로 PM2.5 군집화
    pm25_values = season_data[['PM2.5 (µg/m³)']]
    kmeans_pm25.fit(pm25_values)
    season_data['PM2.5_Cluster'] = kmeans_pm25.labels_
    
    rf_models_pm25[season] = {}
    evaluation_scores_pm25[season] = {}
    for city in nearby_cities:
        if city not in season_data.columns:
            continue
        
        y = season_data[city].dropna()
        if len(y) < 10:
            continue
        
        X_city = X_scaled[:len(y)]
        X_train, X_test, y_train, y_test = train_test_split(X_city, y, test_size=0.2, random_state=42)
        
        # RandomizedSearchCV로 하이퍼파라미터 튜닝
        rf = RandomForestRegressor(random_state=42)
        rf_search = RandomizedSearchCV(
            rf, 
            param_distributions=param_dist, 
            n_iter=20,  # 20개의 조합 시도
            cv=5,       # 5-fold 교차 검증
            scoring='neg_mean_squared_error', 
            random_state=42, 
            n_jobs=-1   # 모든 CPU 코어 사용
        )
        rf_search.fit(X_train, y_train)
        
        # 최적 모델로 예측
        best_rf = rf_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        
        # 평가 점수 계산
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evaluation_scores_pm25[season][city] = {
            'MSE': mse, 
            'RMSE': rmse, 
            'MAE': mae, 
            'R²': r2,
            'Best_Params': rf_search.best_params_  # 최적 파라미터 저장
        }
        
        # 모델 저장
        rf_models_pm25[season][city] = best_rf

# PM10 모델 학습
for season in seasons:
    season_data = merged_data_pm10[merged_data_pm10['Season'] == season].dropna()
    if len(season_data) < 10:
        continue
    
    X = season_data[['PM10 (µg/m³)', 'Wind_X', 'Wind_Y']]
    X_scaled = scaler_pm10.fit_transform(X)
    
    # KMeans로 PM10 군집화
    pm10_values = season_data[['PM10 (µg/m³)']]
    kmeans_pm10.fit(pm10_values)
    season_data['PM10_Cluster'] = kmeans_pm10.labels_
    
    rf_models_pm10[season] = {}
    evaluation_scores_pm10[season] = {}
    for city in nearby_cities:
        if city not in season_data.columns:
            continue
        
        y = season_data[city].dropna()
        if len(y) < 10:
            continue
        
        X_city = X_scaled[:len(y)]
        X_train, X_test, y_train, y_test = train_test_split(X_city, y, test_size=0.2, random_state=42)
        
        # RandomizedSearchCV로 하이퍼파라미터 튜닝
        rf = RandomForestRegressor(random_state=42)
        rf_search = RandomizedSearchCV(
            rf, 
            param_distributions=param_dist, 
            n_iter=20,  # 20개의 조합 시도
            cv=5,       # 5-fold 교차 검증
            scoring='neg_mean_squared_error', 
            random_state=42, 
            n_jobs=-1   # 모든 CPU 코어 사용
        )
        rf_search.fit(X_train, y_train)
        
        # 최적 모델로 예측
        best_rf = rf_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        
        # 평가 점수 계산
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evaluation_scores_pm10[season][city] = {
            'MSE': mse, 
            'RMSE': rmse, 
            'MAE': mae, 
            'R²': r2,
            'Best_Params': rf_search.best_params_  # 최적 파라미터 저장
        }
        
        # 모델 저장
        rf_models_pm10[season][city] = best_rf

# 4. 모델 및 데이터 저장
# StandardScaler 저장 (PM2.5와 PM10 각각)
joblib.dump(scaler_pm25, '../models/rf/scaler_pm25.pkl')
joblib.dump(scaler_pm10, '../models/rf/scaler_pm10.pkl')

# KMeans 모델 저장
joblib.dump(kmeans_pm25, '../models/rf/kmeans_pm25.pkl')
joblib.dump(kmeans_pm10, '../models/rf/kmeans_pm10.pkl')

# season_wind 저장
joblib.dump(season_wind, '../models/rf/season_wind.pkl')

# RandomForest 모델 저장 (PM2.5)
for season in rf_models_pm25:
    for city in rf_models_pm25[season]:
        joblib.dump(rf_models_pm25[season][city], f'../models/rf/rf_pm25_{season}_{city}.pkl')

# RandomForest 모델 저장 (PM10)
for season in rf_models_pm10:
    for city in rf_models_pm10[season]:
        joblib.dump(rf_models_pm10[season][city], f'../models/rf/rf_pm10_{season}_{city}.pkl')

# 평가 점수 저장 (PM2.5)
joblib.dump(evaluation_scores_pm25, '../models/rf/evaluation_scores_pm25.pkl')

# 평가 점수 저장 (PM10)
joblib.dump(evaluation_scores_pm10, '../models/rf/evaluation_scores_pm10.pkl')

# 군집 라벨 저장
joblib.dump(cluster_labels_pm25, '../models/rf/cluster_labels_pm25.pkl')
joblib.dump(cluster_labels_pm10, '../models/rf/cluster_labels_pm10.pkl')

print("모델 학습 및 저장 완료!")