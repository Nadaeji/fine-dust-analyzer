import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import numpy as np
import streamlit as st
import math

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("./data/pm25_pm10_merged_wind.csv")
        data['Date'] = pd.to_datetime(data['Date'])
        # 2015년 ~ 2018년 데이터 삭제
        data = data[data['Date'].dt.year >= 2019]
        # PM2.5와 PM10 값이 0인 행 제거
        data = data[(data['PM2.5 (µg/m³)'] > 0) & (data['PM10 (µg/m³)'] > 0)]
        return data
    except FileNotFoundError:
        st.error("데이터 파일을 찾을 수 없습니다. 올바른 경로를 확인하세요.")
        return None

# 초기 중심값 설정
initial_centroids = np.array([[5], [17.5], [37.5], [60]])

# PM2.5 및 PM10 등급 매핑 함수
def assign_cluster(value, pollutant='PM2.5'):
    if pollutant == 'PM2.5':
        if value <= 10:
            return 0  # "좋음"
        elif value <= 25:
            return 1  # "보통"
        elif value <= 50:
            return 2  # "나쁨"
        else:
            return 3  # "매우 나쁨"
    elif pollutant == 'PM10':
        if value <= 30:
            return 0  # "좋음"
        elif value <= 80:
            return 1  # "보통"
        elif value <= 150:
            return 2  # "나쁨"
        else:
            return 3  # "매우 나쁨"

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

# 시간 지연 데이터 생성 및 모델 학습 함수
def train_model(data, pollutant='PM2.5', lag_days=1):
    # 피벗 테이블 생성 후 결측치 제거
    pivot_data = data.pivot(index='Date', columns='City', values=f'{pollutant} (µg/m³)').dropna()
    
    # 입력: Beijing 현재 농도
    X = pivot_data[['Beijing']]
    # 출력: n일 뒤 다른 도시 농도 (shift 적용 후 결측치 제거)
    y = pivot_data[['Seoul', 'Tokyo', 'Delhi', 'Bangkok']].shift(-lag_days).dropna()
    X = X.iloc[:-lag_days]  # 길이 맞춤

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, init=initial_centroids, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    pivot_data = pivot_data.iloc[:-lag_days].copy()
    pivot_data['Cluster'] = clusters

    models = {}
    X_tests = {}
    y_tests = {}

    for cluster in range(kmeans.n_clusters):
        cluster_data = pivot_data[pivot_data['Cluster'] == cluster].dropna()  # 군집 내 결측치 제거
        X_cluster = cluster_data[['Beijing']]
        y_cluster = y.loc[cluster_data.index]

        if len(X_cluster) > 10:  # 최소 데이터 크기 확인
            X_train, X_test, y_train, y_test = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=1000, random_state=42)
            model.fit(X_train, y_train)
            models[cluster] = model
            X_tests[cluster] = X_test
            y_tests[cluster] = y_test

    return models, kmeans, X_tests, y_tests, pivot_data, scaler

# 예측 함수
def predict_pollutant(models, kmeans, scaler, beijing_value):
    beijing_scaled = scaler.transform(np.array([[beijing_value]]))
    cluster = kmeans.predict(beijing_scaled)[0]
    model = models.get(cluster, None)

    if model:
        prediction = model.predict(np.array([[beijing_value]]))[0]
        city_names = ["Seoul", "Tokyo", "Delhi", "Bangkok"]
        return dict(zip(city_names, prediction))
    else:
        return None

# 회귀 평가 함수: MSE, RMSE, MAE, R² 계산
def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

# 모델 스코어 및 회귀 평가 지표 계산
def score_model(models, X_tests, y_tests):
    scores = {}
    regression_metrics = {}

    for cluster, model in models.items():
        X_test = X_tests[cluster]
        y_test = y_tests[cluster]
        y_pred = model.predict(X_test)
        mse, rmse, mae, r2 = evaluate_regression(y_test, y_pred)
        scores[cluster] = r2
        regression_metrics[cluster] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

    return scores, regression_metrics

# 등급 및 색상 계산 함수
def get_grade(value, pollutant='PM2.5'):
    if pollutant == 'PM2.5':
        if value <= 10:
            return "좋음", "green"
        elif value <= 25:
            return "보통", "blue"
        elif value <= 50:
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

# 도시 좌표 딕셔너리
city_coords = {
    'Seoul': (37.5665, 126.978),
    'Tokyo': (35.6895, 139.6917),
    'Beijing': (39.9042, 116.4074),
    'Delhi': (28.7041, 77.1025),
    'Bangkok': (13.7563, 100.5018)
}

# 한국 계절 정의
seasons = {
    "봄 (3~5월)": [3, 4, 5],
    "여름 (6~8월)": [6, 7, 8],
    "가을 (9~11월)": [9, 10, 11],
    "겨울 (12~2월)": [12, 1, 2]
}

# Streamlit 앱
st.title("Beijing 기반 n일 뒤 도시별 미세먼지(PM2.5/PM10) 예측 및 지도 (KMeans + RandomForest)")

# 데이터 로드
data = load_data()

if data is not None:
    data['Month'] = data['Date'].dt.month
    season_options = list(seasons.keys())
    selected_season = st.selectbox("학습할 계절을 선택하세요", season_options, index=0)
    selected_months = seasons[selected_season]
    lag_days = st.slider("예측 지연 일수 (n일 뒤)", min_value=1, max_value=7, value=2)

    # 선택된 계절의 월로 데이터 필터링
    filtered_data = data[data['Month'].isin(selected_months)]

    # PM2.5와 PM10 모델 학습
    models_pm25, kmeans_pm25, X_tests_pm25, y_tests_pm25, pivot_data_pm25, scaler_pm25 = train_model(filtered_data, 'PM2.5', lag_days)
    models_pm10, kmeans_pm10, X_tests_pm10, y_tests_pm10, pivot_data_pm10, scaler_pm10 = train_model(filtered_data, 'PM10', lag_days)

    # 모델 성능 평가 (회귀 평가 지표 적용)
    scores_pm25, regression_metrics_pm25 = score_model(models_pm25, X_tests_pm25, y_tests_pm25)
    scores_pm10, regression_metrics_pm10 = score_model(models_pm10, X_tests_pm10, y_tests_pm10)

    # 탭 생성
    tab1, tab2 = st.tabs(["PM2.5 예측", "PM10 예측"])

    # PM2.5 탭
    with tab1:
        st.subheader(f"Beijing PM2.5 값을 입력해 {lag_days}일 뒤 예측")
        st.write(f"군집별 회귀 모델 성능 (회귀 평가 지표) - {selected_season}:")
        cols_pm25 = st.columns(len(scores_pm25))
        for idx, cluster in enumerate(scores_pm25.keys()):
            with cols_pm25[idx]:
                with st.expander(f"{cluster_labels_pm25[cluster]}"):
                    st.write(f"R² 스코어: {regression_metrics_pm25[cluster]['R2']:.4f}")
                    st.write(f"MSE: {regression_metrics_pm25[cluster]['MSE']:.4f}")
                    st.write(f"RMSE: {regression_metrics_pm25[cluster]['RMSE']:.4f}")
                    st.write(f"MAE: {regression_metrics_pm25[cluster]['MAE']:.4f}")

        beijing_pm25 = float(st.number_input("Beijing PM2.5 (µg/m³)", min_value=0.0, max_value=300.0, value=50.0, step=1.0, key="pm25"))
        if st.button("PM2.5 예측하기"):
            predictions = predict_pollutant(models_pm25, kmeans_pm25, scaler_pm25, beijing_pm25)
            predictions['Beijing'] = beijing_pm25

            pred_df = pd.DataFrame({
                'City': list(predictions.keys()),
                'PM2.5 (µg/m³)': list(predictions.values()),
                'Latitude': [city_coords[city][0] for city in predictions.keys()],
                'Longitude': [city_coords[city][1] for city in predictions.keys()]
            })

            pred_df[['Grade', 'Color']] = pred_df['PM2.5 (µg/m³)'].apply(lambda x: pd.Series(get_grade(x, 'PM2.5')))
            fig = px.scatter_mapbox(
                pred_df,
                lat="Latitude",
                lon="Longitude",
                size="PM2.5 (µg/m³)",
                color="Grade",
                color_discrete_map={"좋음": "green", "보통": "blue", "나쁨": "orange", "매우 나쁨": "red"},
                hover_name="City",
                hover_data={"PM2.5 (µg/m³)": True, "Grade": True, "Latitude": False, "Longitude": False},
                size_max=30,
                zoom=2,
                mapbox_style="open-street-map",
                title=f"Beijing PM2.5 = {beijing_pm25} µg/m³일 때 {lag_days}일 뒤 예측 ({selected_season})"
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig)

    # PM10 탭
    with tab2:
        st.subheader(f"Beijing PM10 값을 입력해 {lag_days}일 뒤 예측")
        st.write(f"군집별 회귀 모델 성능 (회귀 평가 지표) - {selected_season}:")
        cols_pm10 = st.columns(len(scores_pm10))
        for idx, cluster in enumerate(scores_pm10.keys()):
            with cols_pm10[idx]:
                with st.expander(f"{cluster_labels_pm10[cluster]}"):
                    st.write(f"R² 스코어: {regression_metrics_pm10[cluster]['R2']:.4f}")
                    st.write(f"MSE: {regression_metrics_pm10[cluster]['MSE']:.4f}")
                    st.write(f"RMSE: {regression_metrics_pm10[cluster]['RMSE']:.4f}")
                    st.write(f"MAE: {regression_metrics_pm10[cluster]['MAE']:.4f}")

        beijing_pm10 = float(st.number_input("Beijing PM10 (µg/m³)", min_value=0.0, max_value=500.0, value=50.0, step=1.0, key="pm10"))
        if st.button("PM10 예측하기"):
            predictions = predict_pollutant(models_pm10, kmeans_pm10, scaler_pm10, beijing_pm10)
            predictions['Beijing'] = beijing_pm10

            pred_df = pd.DataFrame({
                'City': list(predictions.keys()),
                'PM10 (µg/m³)': list(predictions.values()),
                'Latitude': [city_coords[city][0] for city in predictions.keys()],
                'Longitude': [city_coords[city][1] for city in predictions.keys()]
            })

            pred_df[['Grade', 'Color']] = pred_df['PM10 (µg/m³)'].apply(lambda x: pd.Series(get_grade(x, 'PM10')))
            fig = px.scatter_mapbox(
                pred_df,
                lat="Latitude",
                lon="Longitude",
                size="PM10 (µg/m³)",
                color="Grade",
                color_discrete_map={"좋음": "green", "보통": "blue", "나쁨": "orange", "매우 나쁨": "red"},
                hover_name="City",
                hover_data={"PM10 (µg/m³)": True, "Grade": True, "Latitude": False, "Longitude": False},
                size_max=30,
                zoom=2,
                mapbox_style="open-street-map",
                title=f"Beijing PM10 = {beijing_pm10} µg/m³일 때 {lag_days}일 뒤 예측 ({selected_season})"
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig)