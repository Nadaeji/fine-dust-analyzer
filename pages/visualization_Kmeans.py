import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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

# 군집 등급 매핑 딕셔너리
cluster_labels = {
    0: "좋음 (PM2.5 ≤ 10 µg/m³)",
    1: "보통 (10 < PM2.5 ≤ 25 µg/m³)",
    2: "나쁨 (25 < PM2.5 ≤ 50 µg/m³)",
    3: "매우 나쁨 (PM2.5 > 50 µg/m³)"
}

# KMeans + RandomForest 모델 학습
def train_model(data):
    pivot_data = data.pivot(index='Date', columns='City', values='PM2.5 (µg/m³)').reset_index().fillna(0)
    X = pivot_data[['Beijing']]  # 입력 변수: Beijing PM2.5만 사용
    y = pivot_data[['Seoul', 'Tokyo', 'Delhi', 'Bangkok']]  # 출력 변수

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, init=initial_centroids, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    pivot_data['Cluster'] = clusters

    models = {}
    X_tests = {}
    y_tests = {}

    for cluster in range(kmeans.n_clusters):
        cluster_data = pivot_data[pivot_data['Cluster'] == cluster]
        X_cluster = cluster_data[['Beijing']]
        y_cluster = cluster_data[['Seoul', 'Tokyo', 'Delhi', 'Bangkok']]

        if len(X_cluster) > 10:
            X_train, X_test, y_train, y_test = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=1000, random_state=42)
            model.fit(X_train, y_train)
            models[cluster] = model
            X_tests[cluster] = X_test
            y_tests[cluster] = y_test

    return models, kmeans, X_tests, y_tests, pivot_data, scaler 

# 예측 함수
def predict_pm25(models, kmeans, scaler, beijing_pm25):
    beijing_scaled = scaler.transform(np.array([[beijing_pm25]]))
    cluster = kmeans.predict(beijing_scaled)[0]
    model = models.get(cluster, None)

    if model:
        prediction = model.predict(np.array([[beijing_pm25]]))[0]
        city_names = ["Seoul", "Tokyo", "Delhi", "Bangkok"]
        return dict(zip(city_names, prediction))
    else:
        return None

# 이진 분류 평가 함수
def evaluate_binary_classification(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return cm, acc, prec, rec, f1

# 모델 스코어 및 이진 분류 성능 계산
def score_model(models, X_tests, y_tests, threshold=35):
    scores = {}
    binary_metrics = {}

    for cluster, model in models.items():
        X_test = X_tests[cluster]
        y_test = y_tests[cluster]

        # 회귀 모델 R² 스코어
        r2_score = model.score(X_test, y_test)
        scores[cluster] = r2_score

        # 이진 분류 성능 계산
        y_pred_cont = model.predict(X_test)
        y_test_binary = (y_test.mean(axis=1) > threshold).astype(int)  # 도시 평균 PM2.5로 이진화
        y_pred_binary = (y_pred_cont.mean(axis=1) > threshold).astype(int)
        cm, acc, prec, rec, f1 = evaluate_binary_classification(y_test_binary, y_pred_binary)
        binary_metrics[cluster] = {'confusion_matrix': cm, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1}

    return scores, binary_metrics

# 등급 계산 함수
def get_grade(pm25):
    if pm25 <= 10:
        return "좋음", "green"
    elif pm25 <= 25:
        return "보통", "blue"
    elif pm25 <= 50:
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

# 데이터 로드 및 모델 학습
data = load_data()

# 사용자가 선택할 수 있도록 특정 월 필터 추가
if data is not None:
    data['Month'] = data['Date'].dt.month  # 월 컬럼 추가
    unique_months = sorted(data['Month'].unique())  # 존재하는 월 리스트

    selected_month = st.selectbox("학습할 월을 선택하세요", unique_months, index=0)  # 기본값 첫 번째 월

    # 특정 월만 필터링
    filtered_data = data[data['Month'] == selected_month]

    # 모델 학습
    models, kmeans, X_tests, y_tests, pivot_data, scaler = train_model(filtered_data)

    # 모델 성능 평가
    scores, binary_metrics = score_model(models, X_tests, y_tests, threshold=35)

    # 예측 지도 탭
    st.subheader("베이징 PM2.5 값을 입력해 예측")
    st.write("군집별 회귀 모델 성능 (R² 스코어) 및 이진 분류 성능:")

    # 가로로 정렬하기 위해 st.columns 사용
    cols = st.columns(len(scores))  # 군집 수만큼 열 생성 (여기서는 4개: | | | |)
    
    # 각 군집별로 열에 expander 배치
    for idx, cluster in enumerate(scores.keys()):
        with cols[idx]:  # 각 열에 배치
            with st.expander(f"{cluster_labels[cluster]}"):
                st.write(f"R² 스코어: {scores[cluster]:.4f}")
                # st.write(f"혼동행렬:\n{binary_metrics[cluster]['confusion_matrix']}")
                st.write(f"정확도: {binary_metrics[cluster]['accuracy']:.4f}")
                st.write(f"정밀도: {binary_metrics[cluster]['precision']:.4f}")
                st.write(f"재현율: {binary_metrics[cluster]['recall']:.4f}")
                st.write(f"F1 스코어: {binary_metrics[cluster]['f1_score']:.5f}")

    beijing_pm25 = float(st.number_input("Beijing PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=100.0, step=1.0))

    if st.button("예측하기"):
        predictions = predict_pm25(models, kmeans, scaler, beijing_pm25)
        predictions['Beijing'] = beijing_pm25

        pred_df = pd.DataFrame({
            'City': list(predictions.keys()),
            'PM2.5 (µg/m³)': list(predictions.values()),
            'Latitude': [city_coords[city][0] for city in predictions.keys()],
            'Longitude': [city_coords[city][1] for city in predictions.keys()]
        })

        pred_df[['Grade', 'Color']] = pred_df['PM2.5 (µg/m³)'].apply(lambda x: pd.Series(get_grade(x)))

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