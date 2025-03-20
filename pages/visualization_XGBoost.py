import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    data = pd.read_csv("./data/pm25_pm10_merged.csv")  # 파일 경로 수정 필요
    data['Date'] = pd.to_datetime(data['Date'])
    data['Month'] = data['Date'].dt.month
    return data

# XGBoost + DBSCAN 모델 학습 (캐싱 적용)
@st.cache_resource
def train_model(data, pollutant='PM2.5'):
    pivot_data = data.pivot(index='Date', columns='City', values=f'{pollutant} (µg/m³)').reset_index().fillna(0)
    pivot_data['Month'] = data.groupby('Date')['Month'].first().values
    
    X = pivot_data[['Beijing', 'Month']]  # 입력 변수
    target_cities = ['Seoul', 'Tokyo', 'Delhi', 'Bangkok', 'Busan', 'Daegu', 'Osaka', 
                     'Sapporo', 'Fukuoka', 'Kyoto', 'Almaty', 'Bishkek', 'Dushanbe', 
                     'Kathmandu', 'Yangon', 'Guwahati', 'Ulaanbaatar', 'Irkutsk']

    # DBSCAN으로 데이터 군집화
    dbscan = DBSCAN(eps=30, min_samples=3)
    clusters = dbscan.fit_predict(X)
    pivot_data['Cluster'] = clusters

    # 각 클러스터와 도시별로 모델 학습
    models = {}
    train_scores = {}
    test_scores = {}
    cv_scores = {}
    rmse_scores = {}
    mse_scores = {}
    mae_scores = {}
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        if cluster != -1:  # 노이즈 제외
            cluster_data = pivot_data[pivot_data['Cluster'] == cluster]
            X_cluster = cluster_data[['Beijing', 'Month']]
            models[cluster] = {}
            train_scores[cluster] = {}
            test_scores[cluster] = {}
            cv_scores[cluster] = {}
            rmse_scores[cluster] = {}
            mse_scores[cluster] = {}
            mae_scores[cluster] = {}
            for city in target_cities:
                if city in cluster_data.columns:
                    y_cluster = cluster_data[city]
                    if len(X_cluster) > 5:
                        X_train, X_test, y_train, y_test = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42)
                        model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42, objective='reg:squarederror')
                        model.fit(X_train, y_train)
                        models[cluster][city] = model
                        train_scores[cluster][city] = model.score(X_train, y_train)
                        test_scores[cluster][city] = model.score(X_test, y_test)
                        if len(X_cluster) > 10:
                            cv_scores[cluster][city] = cross_val_score(model, X_cluster, y_cluster, cv=5, scoring='r2').mean()
                        else:
                            cv_scores[cluster][city] = "N/A"
                        # 테스트 데이터로 RMSE, MSE, MAE 계산
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse_scores[cluster][city] = np.sqrt(mse)
                        mse_scores[cluster][city] = mse
                        mae_scores[cluster][city] = mean_absolute_error(y_test, y_pred)

    return models, dbscan, pivot_data, train_scores, test_scores, cv_scores, rmse_scores, mse_scores, mae_scores

# 예측 함수 (2월로 고정)
def predict_all_cities(models, dbscan, beijing_value, pollutant='PM2.5'):
    input_value = [[beijing_value, 2]]  # 월을 2로 고정
    cluster = dbscan.fit_predict(np.vstack([input_value, dbscan.components_]))[0]
    predictions = {}
    target_cities = ['Seoul', 'Tokyo', 'Delhi', 'Bangkok', 'Busan', 'Daegu', 'Osaka', 
                     'Sapporo', 'Fukuoka', 'Kyoto', 'Almaty', 'Bishkek', 'Dushanbe', 
                     'Kathmandu', 'Yangon', 'Guwahati', 'Ulaanbaatar', 'Irkutsk']
    
    if cluster in models:
        for city in target_cities:
            if city in models[cluster]:
                pred_value = models[cluster][city].predict(input_value)[0]
                predictions[city] = float(pred_value)  # float32 -> float 변환
    else:
        predictions = {city: 0.0 for city in target_cities}  # 노이즈일 경우 0
    predictions['Beijing'] = float(beijing_value)  # 입력값도 float로 변환
    
    return predictions, cluster

# 등급 및 색상 계산 함수
def get_grade(value, pollutant='PM2.5'):
    if pollutant == 'PM2.5':
        if value <= 15:
            return "좋음", "green"
        elif value <= 50:
            return "보통", "blue"
        elif value <= 100:
            return "나쁨", "orange"
        else:
            return "매우 나쁨", "red"
    elif pollutant == 'PM10':
        if value <= 30:
            return "좋음", "green"
        elif value <= 80:
            return "보통", "blue"
        elif value <= 150:
            return "나쁨", "orange"
        else:
            return "매우 나쁨", "red"

# 도시 이름 매핑 (영어 → 한국어)
city_names_kr = {
    "Seoul": "서울", "Tokyo": "도쿄", "Beijing": "베이징", "Delhi": "델리", 
    "Bangkok": "방콕", "Busan": "부산", "Daegu": "대구", "Osaka": "오사카", 
    "Sapporo": "삿포로", "Fukuoka": "후쿠오카", "Kyoto": "교토", 
    "Almaty": "알마티", "Bishkek": "비슈케크", "Dushanbe": "두샨베", 
    "Kathmandu": "카트만두", "Yangon": "양곤", "Guwahati": "구와하티", 
    "Ulaanbaatar": "울란바토르", "Irkutsk": "이르쿠츠크"
}

# 도시 좌표
city_coords = {
    "Seoul": [37.5665, 126.9780], "Tokyo": [35.6895, 139.6917], "Beijing": [39.9042, 116.4074],
    "Delhi": [28.7041, 77.1025], "Bangkok": [13.7563, 100.5018], "Busan": [35.1796, 129.0756],
    "Daegu": [35.8704, 128.5911], "Osaka": [34.6937, 135.5023], "Sapporo": [43.0618, 141.3545],
    "Fukuoka": [33.5904, 130.4017], "Kyoto": [35.0116, 135.7681],
    "Almaty": [43.2220, 76.8512], "Bishkek": [42.8746, 74.5698], "Dushanbe": [38.5481, 68.7864],
    "Kathmandu": [27.7172, 85.3240], "Yangon": [16.8409, 96.1951], "Guwahati": [26.1445, 91.7362],
    "Ulaanbaatar": [47.8864, 106.9057], "Irkutsk": [52.2869, 104.2964],
}

# Streamlit 인터페이스
st.title("베이징 미세먼지가 주변국에 미치는 영향 (XGBoost + DBSCAN, 2월 예측)")

# 데이터 로드
data = load_data()
models_pm25, dbscan_pm25, pivot_data_pm25, train_scores_pm25, test_scores_pm25, cv_scores_pm25, rmse_scores_pm25, mse_scores_pm25, mae_scores_pm25 = train_model(data, 'PM2.5')
models_pm10, dbscan_pm10, pivot_data_pm10, train_scores_pm10, test_scores_pm10, cv_scores_pm10, rmse_scores_pm10, mse_scores_pm10, mae_scores_pm10 = train_model(data, 'PM10')

# 세션 상태 초기화
if 'predictions_pm25' not in st.session_state:
    st.session_state['predictions_pm25'] = None
    st.session_state['cluster_pm25'] = None
if 'predictions_pm10' not in st.session_state:
    st.session_state['predictions_pm10'] = None
    st.session_state['cluster_pm10'] = None

# 탭 생성
tab1, tab2 = st.tabs(["PM2.5 예측", "PM10 예측"])

# PM2.5 탭
with tab1:
    st.header("PM2.5 예측 (2월)")
    beijing_pm25 = st.slider("베이징 PM2.5 (µg/m³)", 0.0, 200.0, 50.0, key="pm25_input")

    if st.button("예측하기 (PM2.5)"):
        predictions, cluster = predict_all_cities(models_pm25, dbscan_pm25, beijing_pm25, 'PM2.5')
        st.session_state['predictions_pm25'] = predictions
        st.session_state['cluster_pm25'] = cluster

    # 예측 결과가 세션에 저장되어 있으면 표시
    if st.session_state['predictions_pm25'] is not None:
        predictions = st.session_state['predictions_pm25']
        cluster = st.session_state['cluster_pm25']

        with st.expander("주변국 PM2.5 예측"):
            pred_table_data = {
                "도시": [city_names_kr[city] for city in predictions.keys()],
                "PM2.5 (µg/m³)": [f"{pm25:.2f}" for pm25 in predictions.values()],
                "상태": [get_grade(pm25, 'PM2.5')[0] for pm25 in predictions.values()]
            }
            pred_df = pd.DataFrame(pred_table_data)
            st.dataframe(pred_df, use_container_width=True)

        with st.expander("모델 평가 점수 (PM2.5)"):
            eval_table_data = {
                "도시": [city_names_kr[city] for city in predictions.keys() if city != 'Beijing'],
                "훈련 R²": [train_scores_pm25[cluster][city] if cluster in train_scores_pm25 and city in train_scores_pm25[cluster] else "N/A" for city in predictions.keys() if city != 'Beijing'],
                "테스트 R²": [test_scores_pm25[cluster][city] if cluster in test_scores_pm25 and city in test_scores_pm25[cluster] else "N/A" for city in predictions.keys() if city != 'Beijing'],
                "교차 검증 R²": [cv_scores_pm25[cluster][city] if cluster in cv_scores_pm25 and city in cv_scores_pm25[cluster] else "N/A" for city in predictions.keys() if city != 'Beijing'],
                "RMSE": [rmse_scores_pm25[cluster][city] if cluster in rmse_scores_pm25 and city in rmse_scores_pm25[cluster] else "N/A" for city in predictions.keys() if city != 'Beijing'],
                "MSE": [mse_scores_pm25[cluster][city] if cluster in mse_scores_pm25 and city in mse_scores_pm25[cluster] else "N/A" for city in predictions.keys() if city != 'Beijing'],
                "MAE": [mae_scores_pm25[cluster][city] if cluster in mae_scores_pm25 and city in mae_scores_pm25[cluster] else "N/A" for city in predictions.keys() if city != 'Beijing']
            }
            eval_df = pd.DataFrame(eval_table_data)
            st.dataframe(eval_df, use_container_width=True)

        st.subheader("지도 (PM2.5)")
        m = folium.Map(location=[35, 120], zoom_start=4)
        for city, pm25 in predictions.items():
            lat, lon = city_coords[city]
            grade, color = get_grade(pm25, 'PM2.5')
            folium.CircleMarker(
                location=[lat, lon],
                radius=pm25 / 10,
                popup=f"{city_names_kr[city]}: {pm25:.2f} µg/m³ ({grade})",
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
        st_folium(m, width=700, height=500, key="map_pm25")

# PM10 탭
with tab2:
    st.header("PM10 예측 (2월)")
    beijing_pm10 = st.slider("베이징 PM10 (µg/m³)", 0.0, 300.0, 75.0, key="pm10_input")

    if st.button("예측하기 (PM10)"):
        predictions, cluster = predict_all_cities(models_pm10, dbscan_pm10, beijing_pm10, 'PM10')
        st.session_state['predictions_pm10'] = predictions
        st.session_state['cluster_pm10'] = cluster

    # 예측 결과가 세션에 저장되어 있으면 표시
    if st.session_state['predictions_pm10'] is not None:
        predictions = st.session_state['predictions_pm10']
        cluster = st.session_state['cluster_pm10']

        with st.expander("주변국 PM10 예측"):
            pred_table_data = {
                "도시": [city_names_kr[city] for city in predictions.keys()],
                "PM10 (µg/m³)": [f"{pm10:.2f}" for pm10 in predictions.values()],
                "상태": [get_grade(pm10, 'PM10')[0] for pm10 in predictions.values()]
            }
            pred_df = pd.DataFrame(pred_table_data)
            st.dataframe(pred_df, use_container_width=True)

        with st.expander("모델 평가 점수 (PM10)"):
            eval_table_data = {
                "도시": [city_names_kr[city] for city in predictions.keys() if city != 'Beijing'],
                "훈련 R²": [train_scores_pm10[cluster][city] if cluster in train_scores_pm10 and city in train_scores_pm10[cluster] else "N/A" for city in predictions.keys() if city != 'Beijing'],
                "테스트 R²": [test_scores_pm10[cluster][city] if cluster in test_scores_pm10 and city in test_scores_pm10[cluster] else "N/A" for city in predictions.keys() if city != 'Beijing'],
                "교차 검증 R²": [cv_scores_pm10[cluster][city] if cluster in cv_scores_pm10 and city in cv_scores_pm10[cluster] else "N/A" for city in predictions.keys() if city != 'Beijing'],
                "RMSE": [rmse_scores_pm10[cluster][city] if cluster in rmse_scores_pm10 and city in rmse_scores_pm10[cluster] else "N/A" for city in predictions.keys() if city != 'Beijing'],
                "MSE": [mse_scores_pm10[cluster][city] if cluster in mse_scores_pm10 and city in mse_scores_pm10[cluster] else "N/A" for city in predictions.keys() if city != 'Beijing'],
                "MAE": [mae_scores_pm10[cluster][city] if cluster in mae_scores_pm10 and city in mae_scores_pm10[cluster] else "N/A" for city in predictions.keys() if city != 'Beijing']
            }
            eval_df = pd.DataFrame(eval_table_data)
            st.dataframe(eval_df, use_container_width=True)

        st.subheader("지도 (PM10)")
        m = folium.Map(location=[35, 120], zoom_start=4)
        for city, pm10 in predictions.items():
            lat, lon = city_coords[city]
            grade, color = get_grade(pm10, 'PM10')
            folium.CircleMarker(
                location=[lat, lon],
                radius=pm10 / 10,
                popup=f"{city_names_kr[city]}: {pm10:.2f} µg/m³ ({grade})",
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
        st_folium(m, width=700, height=500, key="map_pm10")