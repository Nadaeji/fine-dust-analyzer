import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import streamlit as st

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("./data/pm25_pm10_merged.csv")  # 파일 경로 확인 필요
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except FileNotFoundError:
        st.error("데이터 파일을 찾을 수 없습니다. 올바른 경로를 확인하세요.")
        return None


initial_centroids = np.array([[5], [17.5], [37.5], [60]])
def assign_pm25_cluster(pm25):
    if pm25 <= 10:
        return 0  # "좋음"
    elif pm25 <= 25:
        return 1  # "보통"
    elif pm25 <= 50:
        return 2  # "나쁨"
    else:
        return 3  # "매우 나쁨"
# KMeans + RandomForest 모델 학습
def train_model(data):
    # Pivot 데이터 만들기
    pivot_data = data.pivot(index='Date', columns='City', values='PM2.5 (µg/m³)').reset_index().fillna(0)
    X = pivot_data[['Beijing']]  # 입력 변수: Beijing PM2.5만 사용
    y = pivot_data[['Seoul', 'Tokyo', 'Delhi', 'Bangkok']]  # 출력 변수

    # WHO 기준에 맞춰 KMeans 초기 중심 설정
    initial_centroids = np.array([[5], [17.5], [37.5], [60]])  # WHO 기준 PM2.5 값 기반

    # 데이터 정규화 (KMeans가 거리 기반이므로 스케일링 필요)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans로 데이터 군집화
    kmeans = KMeans(n_clusters=4, init=initial_centroids, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    pivot_data['Cluster'] = clusters

    # 각 군집별로 모델 학습
    models = {}
    X_tests = {}
    y_tests = {}

    for cluster in range(kmeans.n_clusters):
        cluster_data = pivot_data[pivot_data['Cluster'] == cluster]
        X_cluster = cluster_data[['Beijing']]
        y_cluster = cluster_data[['Seoul', 'Tokyo', 'Delhi', 'Bangkok']]

        if len(X_cluster) > 10:  # 데이터가 너무 적으면 학습하지 않음
            X_train, X_test, y_train, y_test = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=1000, random_state=42)
            model.fit(X_train, y_train)
            models[cluster] = model
            X_tests[cluster] = X_test
            y_tests[cluster] = y_test

    return models, kmeans, X_tests, y_tests, pivot_data, scaler 

# 예측 함수
def predict_pm25(models, kmeans, scaler, beijing_pm25):
    beijing_scaled = scaler.transform(np.array([[beijing_pm25]]))  # reshape 적용
    cluster = kmeans.predict(beijing_scaled)[0]  # 군집 예측
    model = models.get(cluster, None)  # 해당 군집 모델 가져오기

    if model:
        prediction = model.predict(np.array([[beijing_pm25]]))[0]  # 2D 입력 필요
        city_names = ["Seoul", "Tokyo", "Delhi", "Bangkok"]
        return dict(zip(city_names, prediction))
    else:
        return None



# 모델 스코어 계산
def score_model(models, X_tests, y_tests):
    scores = {}

    for cluster, model in models.items():
        X_test = X_tests[cluster]
        y_test = y_tests[cluster]

        # 모델의 R² 스코어 계산
        r2_score = model.score(X_test, y_test)

        # 결과 저장 (군집별 R² 스코어)
        scores[cluster] = r2_score

    return scores


# 등급 계산 함수
def get_grade(pm25):
    if pm25 <= 15:
        return "좋음", "green"
    elif pm25 <= 35:
        return "보통", "blue"
    elif pm25 <= 75:
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

# Streamlit 앱
st.title("Beijing PM2.5 기반 도시별 미세먼지 예측 및 시간별 지도 (KMeans + RandomForest)")

# 데이터 로드
data = load_data()
models, kmeans, X_tests, y_tests, pivot_data, scaler = train_model(data)

predicted_pm25 = predict_pm25(models, kmeans, scaler, 30)

# 모델 스코어 계산
scores = score_model(models, X_tests, y_tests)

# 탭 1: 예측 지도

st.subheader("베이징 PM2.5 값을 입력해 예측")
st.write("군집별 모델 성능 (R² 스코어):")
for cluster, score in scores.items():
    st.write(f"군집 {cluster}: {score:.4f}")
st.write("※ R² 스코어는 모델이 데이터 변동성을 얼마나 설명하는지를 나타냅니다. 1에 가까울수록 예측력이 높습니다.")

beijing_pm25 = float(st.number_input("Beijing PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=100.0, step=1.0))


if st.button("예측하기"):
    predictions = predict_pm25(models, kmeans, scaler, beijing_pm25)
    predictions['Beijing'] = beijing_pm25

    # 예측 데이터프레임 생성
    pred_df = pd.DataFrame({
        'City': list(predictions.keys()),
        'PM2.5 (µg/m³)': list(predictions.values()),
        'Latitude': [city_coords[city][0] for city in predictions.keys()],
        'Longitude': [city_coords[city][1] for city in predictions.keys()]
    })

    # 등급과 색상 추가
    pred_df[['Grade', 'Color']] = pred_df['PM2.5 (µg/m³)'].apply(lambda x: pd.Series(get_grade(x)))

    # 지도 시각화
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
        title=f"Beijing PM2.5 = {beijing_pm25} µg/m³일 때 예측"
        )

    fig.update_traces(textposition="top center")
    st.plotly_chart(fig)