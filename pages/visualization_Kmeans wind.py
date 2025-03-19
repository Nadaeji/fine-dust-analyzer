import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px
import streamlit as st
import math

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("./data/pm25_pm10_merged_wind.csv")  # 파일 경로 확인
        data['Date'] = pd.to_datetime(data['Date'])
        # 계절 컬럼 추가
        data['Month'] = data['Date'].dt.month
        data['Season'] = data['Month'].apply(
            lambda x: '봄' if x in [3, 4, 5] else
                      '여름' if x in [6, 7, 8] else
                      '가을' if x in [9, 10, 11] else
                      '겨울'
        )
        print("데이터 열 이름:", data.columns.tolist())
        return data
    except FileNotFoundError:
        st.error("데이터 파일을 찾을 수 없습니다. 올바른 경로를 확인하세요.")
        return None

# PM2.5 군집 등급 지정
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

# 미세먼지 이동 경로 계산 함수
def calculate_movement(lat, lon, wind_speed, wind_direction, time_interval=24):
    wind_rad = math.radians(90 - wind_direction)
    distance = wind_speed * time_interval * 3600 / 1000  # km 단위
    delta_lat = (distance * math.cos(wind_rad)) / 111
    delta_lon = (distance * math.sin(wind_rad)) / (111 * math.cos(math.radians(lat)))
    new_lat = lat + delta_lat
    new_lon = lon + delta_lon
    return new_lat, new_lon

# 도시 좌표 딕셔너리
city_coords = {
    "Seoul": [126.9780, 37.5665],
    "Tokyo": [139.6917, 35.6895],
    "Beijing": [116.4074, 39.9042],
    "Delhi": [77.1025, 28.7041],
    "Bangkok": [100.5018, 13.7563],
    "Busan": [129.0756, 35.1796],
    "Incheon": [126.7052, 37.4563],
    "Daegu": [128.5911, 35.8704],
    "Osaka": [135.5023, 34.6937],
    "Nagoya": [136.9066, 35.1802],
    "Sapporo": [141.3545, 43.0618],
    "Fukuoka": [130.4017, 33.5904],
    "Kyoto": [135.7681, 35.0116],
    "Shanghai": [121.4737, 31.2304],
    "Guangzhou": [113.2644, 23.1291],
    "Chongqing": [106.5516, 29.5630],
    "Tianjin": [117.1902, 39.1256],
    "Wuhan": [114.3055, 30.5928],
    "Nanjing": [118.7969, 32.0603],
    "Hangzhou": [120.1551, 30.2741],
    "Chengdu": [104.0668, 30.5728],
    "Xi'an": [108.9355, 34.3416],
    "Shenyang": [123.4291, 41.7968],
    "Almaty": [76.8512, 43.2220],
    "Bishkek": [74.5698, 42.8746],
    "Dushanbe": [68.7864, 38.5481],
    "Kathmandu": [85.3240, 27.7172],
    "Yangon": [96.1951, 16.8409],
    "Guwahati": [91.7362, 26.1445],
    "Ulaanbaatar": [106.9057, 47.8864],
    "Irkutsk": [104.2964, 52.2869],
    "Vladivostok": [131.8869, 43.1155]
}

# KMeans + RandomForest 모델 학습
def train_model(data, season):
    seasonal_data = data[data['Season'] == season]
    
    expected_columns = ['PM2.5 (µg/m³)', 'Wind Speed (m/s)', 'Wind Direction (degrees)']
    missing_columns = [col for col in expected_columns if col not in seasonal_data.columns]
    if missing_columns:
        raise KeyError(f"다음 열이 데이터에 없습니다: {missing_columns}")

    pivot_data = seasonal_data.pivot_table(
        index='Date', 
        columns='City', 
        values=['PM2.5 (µg/m³)', 'Wind Speed (m/s)', 'Wind Direction (degrees)'], 
        aggfunc='mean'
    ).reset_index().fillna(0)
    
    X = pivot_data[[('PM2.5 (µg/m³)', 'Beijing'), ('Wind Speed (m/s)', 'Beijing'), ('Wind Direction (degrees)', 'Beijing')]]
    y_columns = [(f'PM2.5 (µg/m³)', city) for city in city_coords.keys() if city != 'Beijing']
    y = pivot_data[y_columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mean_wind_speed = seasonal_data[seasonal_data['City'] == 'Beijing']['Wind Speed (m/s)'].mean()
    mean_wind_direction = seasonal_data[seasonal_data['City'] == 'Beijing']['Wind Direction (degrees)'].mean()
    initial_centroids = np.array([
        [5, mean_wind_speed, mean_wind_direction],
        [17.5, mean_wind_speed, mean_wind_direction],
        [37.5, mean_wind_speed, mean_wind_direction],
        [60, mean_wind_speed, mean_wind_direction]
    ])

    kmeans = KMeans(n_clusters=4, init=initial_centroids, n_init=1, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    pivot_data['Cluster'] = clusters

    models = {}
    X_tests = {}
    y_tests = {}

    for cluster in range(kmeans.n_clusters):
        cluster_data = pivot_data[pivot_data['Cluster'] == cluster]
        X_cluster = cluster_data[[('PM2.5 (µg/m³)', 'Beijing'), ('Wind Speed (m/s)', 'Beijing'), ('Wind Direction (degrees)', 'Beijing')]]
        y_cluster = cluster_data[y_columns]

        if len(X_cluster) > 10:
            X_train, X_test, y_train, y_test = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=1000, random_state=42)
            model.fit(X_train, y_train)
            models[cluster] = model
            X_tests[cluster] = X_test
            y_tests[cluster] = y_test

    return models, kmeans, X_tests, y_tests, pivot_data, scaler, mean_wind_speed, mean_wind_direction

# 예측 함수
def predict_pm25(models, kmeans, scaler, beijing_pm25, wind_speed, wind_direction):
    beijing_input = np.array([[beijing_pm25, wind_speed, wind_direction]])
    beijing_scaled = scaler.transform(beijing_input)
    cluster = kmeans.predict(beijing_scaled)[0]
    model = models.get(cluster, None)

    if model:
        prediction = model.predict(beijing_input)[0]
        city_names = [city for city in city_coords.keys() if city != 'Beijing']
        return dict(zip(city_names, prediction))
    else:
        return None

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

# Streamlit 앱
st.title("베이징 PM2.5 기반 미세먼지 예측 및 이동 경로 시각화 (계절별, KMeans + RandomForest)")

# 데이터 로드
data = load_data()

if data is not None:
    # 계절 선택
    seasons = ['봄', '여름', '가을', '겨울']
    selected_season = st.selectbox("계절을 선택하세요", seasons)

    # 모델 학습 (계절별)
    try:
        models, kmeans, X_tests, y_tests, pivot_data, scaler, mean_wind_speed, mean_wind_direction = train_model(data, selected_season)
    except KeyError as e:
        st.error(f"모델 학습 중 오류 발생: {e}")
        st.stop()

    # 예측 및 이동 경로 시각화 탭
    st.subheader(f"{selected_season} 계절 기반 예측 및 이동 경로")
    
    # 계절별 평균 풍속과 풍향 표시
    st.write(f"{selected_season} 평균 풍속 (베이징): {mean_wind_speed:.2f} m/s")
    st.write(f"{selected_season} 평균 풍향 (베이징): {mean_wind_direction:.2f}°")

    # 입력값
    beijing_pm25 = float(st.number_input("베이징 PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=100.0, step=1.0))
    days_after = int(st.number_input("몇 일 뒤의 변화를 예측할까요? (일)", min_value=1, max_value=30, value=3, step=1))

    if st.button("예측 및 이동 경로 보기"):
        # 현재 예측 수행
        current_predictions = predict_pm25(models, kmeans, scaler, beijing_pm25, mean_wind_speed, mean_wind_direction)
        if current_predictions:
            current_predictions['Beijing'] = beijing_pm25

            # n일 뒤 예측 (단순히 동일 모델로 예측값을 재사용, 실제로는 시간 경과를 반영한 모델 필요)
            # 여기서는 단순화를 위해 현재 값을 기반으로 이동 경로만 반영
            future_predictions = predict_pm25(models, kmeans, scaler, beijing_pm25, mean_wind_speed, mean_wind_direction)
            future_predictions['Beijing'] = beijing_pm25

            # 변화량 계산
            change_dict = {city: future_predictions[city] - current_predictions[city] for city in current_predictions.keys()}

            # 예측 데이터프레임 생성 (현재)
            pred_df = pd.DataFrame({
                '도시': list(current_predictions.keys()),
                '현재 PM2.5 (µg/m³)': list(current_predictions.values()),
                f'{days_after}일 후 PM2.5 (µg/m³)': list(future_predictions.values()),
                '변화량 (µg/m³)': [change_dict[city] for city in current_predictions.keys()],
                '경도': [city_coords[city][0] for city in current_predictions.keys()],
                '위도': [city_coords[city][1] for city in current_predictions.keys()]
            })

            # 등급 추가
            pred_df[['등급', '색상']] = pred_df['현재 PM2.5 (µg/m³)'].apply(lambda x: pd.Series(get_grade(x)))

            # 이동 경로 계산 (Beijing 기준, 계절 평균값 사용, n일 뒤)
            beijing_lat, beijing_lon = city_coords['Beijing'][1], city_coords['Beijing'][0]
            new_lat, new_lon = calculate_movement(beijing_lat, beijing_lon, mean_wind_speed, mean_wind_direction, time_interval=24 * days_after)
            
            # 이동 경로 데이터프레임
            movement_df = pd.DataFrame({
                '위도': [beijing_lat, new_lat],
                '경도': [beijing_lon, new_lon],
                '도시': ['Beijing', f'Beijing ({days_after}일 후 예측 이동 경로)'],
                '현재 PM2.5 (µg/m³)': [beijing_pm25, beijing_pm25]
            })
            movement_df[['등급', '색상']] = movement_df['현재 PM2.5 (µg/m³)'].apply(lambda x: pd.Series(get_grade(x)))

            # 결과 표 출력
            st.write("### 도시별 현재 PM2.5 및 변화량")
            st.dataframe(pred_df[['도시', '현재 PM2.5 (µg/m³)', f'{days_after}일 후 PM2.5 (µg/m³)', '변화량 (µg/m³)', '등급']])

            # Plotly 지도 시각화
            fig = px.scatter_mapbox(
                pred_df, 
                lat="위도", 
                lon="경도", 
                size="현재 PM2.5 (µg/m³)", 
                color="등급", 
                color_discrete_map={"좋음": "green", "보통": "blue", "나쁨": "orange", "매우 나쁨": "red"},
                hover_name="도시", 
                hover_data={"현재 PM2.5 (µg/m³)": True, f"{days_after}일 후 PM2.5 (µg/m³)": True, "변화량 (µg/m³)": True, "등급": True, "위도": False, "경도": False},
                size_max=30,
                zoom=2,
                mapbox_style="open-street-map",
                title=f"{selected_season} - 베이징 PM2.5 = {beijing_pm25} µg/m³, 풍속 = {mean_wind_speed:.2f} m/s, 풍향 = {mean_wind_direction:.2f}°"
            )

            # 이동 경로 추가
            fig.add_trace(px.line_mapbox(
                movement_df,
                lat="위도",
                lon="경도",
                hover_name="도시",
                hover_data={"현재 PM2.5 (µg/m³)": True}
            ).data[0])

            fig.update_traces(textposition="top center")
            st.plotly_chart(fig)