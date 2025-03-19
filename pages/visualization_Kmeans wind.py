import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
        data = data[data['Date'].dt.year >= 2019]
        data['Month'] = data['Date'].dt.month
        data['Season'] = data['Month'].apply(
            lambda x: '봄' if x in [3, 4, 5] else
                      '여름' if x in [6, 7, 8] else
                      '가을' if x in [9, 10, 11] else
                      '겨울'
        )
        
        # 여름철 풍향을 남동풍(135도)으로 설정
        summer_mask = data['Season'] == '여름'
        data.loc[summer_mask, 'Wind Direction (degrees)'] = 135
        
        print("데이터 열 이름:", data.columns.tolist())
        return data
    except FileNotFoundError:
        st.error("데이터 파일을 찾을 수 없습니다. 올바른 경로를 확인하세요.")
        return None

# 군집 등급 지정 (PM2.5와 PM10 공통)
def assign_cluster(value, is_pm25=True):
    if is_pm25:
        if value <= 10:
            return "좋음", "green"
        elif value <= 25:
            return "보통", "blue"
        elif value <= 50:
            return "나쁨", "orange"
        else:
            return "매우 나쁨", "red"
    else:  # PM10 기준
        if value <= 30:
            return "좋음", "green"
        elif value <= 80:
            return "보통", "blue"
        elif value <= 150:
            return "나쁨", "orange"
        else:
            return "매우 나쁨", "red"

# 군집 설명 텍스트 생성
def get_cluster_description(cluster, is_pm25=True):
    if is_pm25:
        descriptions = {
            0: "0번 군집: 미세먼지 좋음 (PM2.5 ≤ 10 µg/m³)",
            1: "1번 군집: 미세먼지 보통 (10 < PM2.5 ≤ 25 µg/m³)",
            2: "2번 군집: 미세먼지 나쁨 (25 < PM2.5 ≤ 50 µg/m³)",
            3: "3번 군집: 미세먼지 매우 나쁨 (PM2.5 > 50 µg/m³)"
        }
    else:
        descriptions = {
            0: "0번 군집: 미세먼지 좋음 (PM10 ≤ 30 µg/m³)",
            1: "1번 군집: 미세먼지 보통 (30 < PM10 ≤ 80 µg/m³)",
            2: "2번 군집: 미세먼지 나쁨 (80 < PM10 ≤ 150 µg/m³)",
            3: "3번 군집: 미세먼지 매우 나쁨 (PM10 > 150 µg/m³)"
        }
    return descriptions.get(cluster, f"{cluster}번 군집: 정의되지 않음")

# 미세먼지 이동 경로 및 농도 변화 계산 함수
def calculate_movement_and_concentration(lat, lon, value, wind_speed, wind_direction, days=1, decay_rate=0.05):
    wind_rad = math.radians(90 - wind_direction)
    distance = wind_speed * days * 24 * 3600 / 1000  # 총 이동 거리 (km)
    delta_lat = (distance * math.cos(wind_rad)) / 111
    delta_lon = (distance * math.sin(wind_rad)) / (111 * math.cos(math.radians(lat)))
    new_lat = lat + delta_lat
    new_lon = lon + delta_lon
    
    total_decay = (1 - decay_rate) ** days
    new_value = value * total_decay
    return new_lat, new_lon, new_value

# 이진 분류 평가 함수
def evaluate_binary_classification(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return cm, acc, prec, rec, f1

# 도시 좌표 딕셔너리
city_coords = {
    "Seoul": [126.9780, 37.5665],
    "Tokyo": [139.6917, 35.6895],
    "Beijing": [116.4074, 39.9042],
    "Delhi": [77.1025, 28.7041],
    "Bangkok": [100.5018, 13.7563],
    "Busan": [129.0756, 35.1796],
    "Daegu": [128.5911, 35.8704],
    "Osaka": [135.5023, 34.6937],
    "Sapporo": [141.3545, 43.0618],
    "Fukuoka": [130.4017, 33.5904],
    "Kyoto": [135.7681, 35.0116],
    "Shanghai": [121.4737, 31.2304],
    "Guangzhou": [113.2644, 23.1291],
    "Chongqing": [106.5516, 29.5630],
    "Wuhan": [114.3055, 30.5928],
    "Nanjing": [118.7969, 32.0603],
    "Hangzhou": [120.1551, 30.2741],
    "Chengdu": [104.0668, 30.5728],
    "Almaty": [76.8512, 43.2220],
    "Bishkek": [74.5698, 42.8746],
    "Dushanbe": [68.7864, 38.5481],
    "Kathmandu": [85.3240, 27.7172],
    "Yangon": [96.1951, 16.8409],
    "Guwahati": [91.7362, 26.1445],
    "Ulaanbaatar": [106.9057, 47.8864],
    "Irkutsk": [104.2964, 52.2869],
}

# KMeans + RandomForest 모델 학습 및 성능 평가
def train_model(data, season, target='PM2.5 (µg/m³)'):
    seasonal_data = data[data['Season'] == season]
    
    expected_columns = [target, 'Wind Speed (m/s)', 'Wind Direction (degrees)']
    missing_columns = [col for col in expected_columns if col not in seasonal_data.columns]
    if missing_columns:
        raise KeyError(f"다음 열이 데이터에 없습니다: {missing_columns}")

    pivot_data = seasonal_data.pivot_table(
        index='Date', 
        columns='City', 
        values=[target, 'Wind Speed (m/s)', 'Wind Direction (degrees)'], 
        aggfunc='mean'
    ).reset_index().fillna(0)
    
    X = pivot_data[[(target, 'Beijing'), ('Wind Speed (m/s)', 'Beijing'), ('Wind Direction (degrees)', 'Beijing')]]
    y_columns = [(target, city) for city in city_coords.keys() if city != 'Beijing']
    y = pivot_data[y_columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mean_wind_speed = seasonal_data[seasonal_data['City'] == 'Beijing']['Wind Speed (m/s)'].mean()
    mean_wind_direction = seasonal_data[seasonal_data['City'] == 'Beijing']['Wind Direction (degrees)'].mean()
    initial_centroids = np.array([
        [5 if target == 'PM2.5 (µg/m³)' else 15, mean_wind_speed, mean_wind_direction],
        [17.5 if target == 'PM2.5 (µg/m³)' else 50, mean_wind_speed, mean_wind_direction],
        [37.5 if target == 'PM2.5 (µg/m³)' else 100, mean_wind_speed, mean_wind_direction],
        [60 if target == 'PM2.5 (µg/m³)' else 200, mean_wind_speed, mean_wind_direction]
    ])

    kmeans = KMeans(n_clusters=4, init=initial_centroids, n_init=1, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    pivot_data['Cluster'] = clusters

    models = {}
    X_tests = {}
    y_tests = {}
    metrics = {}

    for cluster in range(kmeans.n_clusters):
        cluster_data = pivot_data[pivot_data['Cluster'] == cluster]
        X_cluster = cluster_data[[(target, 'Beijing'), ('Wind Speed (m/s)', 'Beijing'), ('Wind Direction (degrees)', 'Beijing')]]
        y_cluster = cluster_data[y_columns]

        if len(X_cluster) > 10:
            X_train, X_test, y_train, y_test = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=500, random_state=0)
            model.fit(X_train, y_train)
            models[cluster] = model
            X_tests[cluster] = X_test
            y_tests[cluster] = y_test

            # R² 스코어 계산
            r2 = model.score(X_test, y_test)

            # 이진 분류 성능 평가
            y_pred_cont = model.predict(X_test)
            y_test_mean = y_test.mean(axis=1)
            y_pred_mean = y_pred_cont.mean(axis=1)
            threshold = 35 if target == 'PM2.5 (µg/m³)' else 100  # PM2.5: 35, PM10: 100
            y_test_binary = (y_test_mean > threshold).astype(int)
            y_pred_binary = (y_pred_mean > threshold).astype(int)
            cm, acc, prec, rec, f1 = evaluate_binary_classification(y_test_binary, y_pred_binary)
            metrics[cluster] = {'혼동 행렬': cm, '정확도': acc, '정밀도': prec, '재현율': rec, 'F1 점수': f1, 'R² 스코어': r2}

    return models, kmeans, X_tests, y_tests, pivot_data, scaler, mean_wind_direction, metrics

# 예측 함수
def predict_values(models, kmeans, scaler, beijing_value, wind_speed, wind_direction):
    beijing_input = np.array([[beijing_value, wind_speed, wind_direction]])
    beijing_scaled = scaler.transform(beijing_input)
    cluster = kmeans.predict(beijing_scaled)[0]
    model = models.get(cluster, None)

    if model:
        prediction = model.predict(beijing_input)[0]
        city_names = [city for city in city_coords.keys() if city != 'Beijing']
        return dict(zip(city_names, prediction))
    else:
        return None

# Streamlit 앱
st.title("베이징 기반 미세먼지 예측 및 이동 경로 시각화 (계절별, KMeans + RandomForest)")

# 데이터 로드
data = load_data()

if data is not None:
    # 계절 선택
    seasons = ['봄', '여름', '가을', '겨울']
    selected_season = st.selectbox("계절을 선택하세요", seasons)

    # 탭 생성
    tab1, tab2 = st.tabs(["PM2.5 예측", "PM10 예측"])

    # 공통 입력값
    wind_speed_options = [round(x * 0.5, 1) for x in range(0, 11)]  # 0부터 5까지 0.5 단위
    selected_wind_speed = st.selectbox("풍속을 선택하세요 (m/s)", wind_speed_options, index=4)  # 기본값 2.0
    days_after = int(st.number_input("몇 일 뒤의 변화를 예측할까요? (일)", min_value=1, max_value=30, value=3, step=1))

    # PM2.5 탭
    with tab1:
        st.subheader(f"{selected_season} 계절 기반 PM2.5 예측 및 이동 경로")
        
        # 모델 학습 (PM2.5)
        try:
            models_pm25, kmeans_pm25, X_tests_pm25, y_tests_pm25, pivot_data_pm25, scaler_pm25, mean_wind_direction, metrics_pm25 = train_model(data, selected_season, target='PM2.5 (µg/m³)')
        except KeyError as e:
            st.error(f"모델 학습 중 오류 발생: {e}")
            st.stop()

        st.write(f"{selected_season} 평균 풍향 (베이징): {mean_wind_direction:.2f}°")

        # 모델 성능 가로 출력 및 클릭 상세 정보
        st.write("### 모델 성능 (PM2.5)")
        cols = st.columns(4)  # 4개 군집을 가로로 배치
        for cluster, col in enumerate(cols):
            with col:
                with st.expander(get_cluster_description(cluster, is_pm25=True)):
                    st.write(f"혼동 행렬:\n{metrics_pm25[cluster]['혼동 행렬']}")
                    st.write(f"정확도: {metrics_pm25[cluster]['정확도']:.4f}")
                    st.write(f"정밀도: {metrics_pm25[cluster]['정밀도']:.4f}")
                    st.write(f"재현율: {metrics_pm25[cluster]['재현율']:.4f}")
                    st.write(f"F1 점수: {metrics_pm25[cluster]['F1 점수']:.4f}")
                    st.write(f"R² 스코어: {metrics_pm25[cluster]['R² 스코어']:.4f}")

        beijing_pm25 = float(st.number_input("베이징 PM2.5 (µg/m³)", min_value=0.0, max_value=300.0, value=30.0, step=1.0, key="pm25"))

        if st.button("예측 및 이동 경로 보기", key="pm25_button"):
            current_predictions = predict_values(models_pm25, kmeans_pm25, scaler_pm25, beijing_pm25, selected_wind_speed, mean_wind_direction)
            if current_predictions:
                current_predictions['Beijing'] = beijing_pm25

                future_predictions = {}
                for city, pm25 in current_predictions.items():
                    lat, lon = city_coords[city][1], city_coords[city][0]
                    new_lat, new_lon, new_pm25 = calculate_movement_and_concentration(
                        lat, lon, pm25, selected_wind_speed, mean_wind_direction, days=days_after
                    )
                    future_predictions[city] = max(new_pm25, 0)

                change_dict = {city: future_predictions[city] - current_predictions[city] for city in current_predictions.keys()}

                pred_df = pd.DataFrame({
                    '도시': list(current_predictions.keys()),
                    '현재 PM2.5 (µg/m³)': list(current_predictions.values()),
                    f'{days_after}일 후 PM2.5 (µg/m³)': list(future_predictions.values()),
                    '변화량 (µg/m³)': [change_dict[city] for city in current_predictions.keys()],
                    '경도': [city_coords[city][0] for city in current_predictions.keys()],
                    '위도': [city_coords[city][1] for city in current_predictions.keys()]
                })

                pred_df[['등급', '색상']] = pred_df['현재 PM2.5 (µg/m³)'].apply(lambda x: pd.Series(assign_cluster(x, is_pm25=True)))

                beijing_lat, beijing_lon = city_coords['Beijing'][1], city_coords['Beijing'][0]
                new_lat, new_lon, new_beijing_pm25 = calculate_movement_and_concentration(
                    beijing_lat, beijing_lon, beijing_pm25, selected_wind_speed, mean_wind_direction, days=days_after
                )

                movement_df = pd.DataFrame({
                    '위도': [beijing_lat, new_lat],
                    '경도': [beijing_lon, new_lon],
                    '도시': ['Beijing', f'Beijing ({days_after}일 후 예측 이동 경로)'],
                    '현재 PM2.5 (µg/m³)': [beijing_pm25, beijing_pm25],
                    f'{days_after}일 후 PM2.5 (µg/m³)': [beijing_pm25, new_beijing_pm25]
                })
                movement_df[['등급', '색상']] = movement_df['현재 PM2.5 (µg/m³)'].apply(lambda x: pd.Series(assign_cluster(x, is_pm25=True)))

                st.write("### 도시별 현재 PM2.5 및 변화량")
                st.dataframe(pred_df[['도시', '현재 PM2.5 (µg/m³)', f'{days_after}일 후 PM2.5 (µg/m³)', '변화량 (µg/m³)', '등급']])

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
                    title=f"{selected_season} - 베이징 PM2.5 = {beijing_pm25} µg/m³, 풍속 = {selected_wind_speed:.2f} m/s, 풍향 = {mean_wind_direction:.2f}°"
                )

                fig.add_trace(px.line_mapbox(
                    movement_df,
                    lat="위도",
                    lon="경도",
                    hover_name="도시",
                    hover_data={"현재 PM2.5 (µg/m³)": True, f"{days_after}일 후 PM2.5 (µg/m³)": True}
                ).data[0])

                fig.update_traces(textposition="top center")
                st.plotly_chart(fig)

    # PM10 탭
    with tab2:
        st.subheader(f"{selected_season} 계절 기반 PM10 예측 및 이동 경로")
        
        # 모델 학습 (PM10)
        try:
            models_pm10, kmeans_pm10, X_tests_pm10, y_tests_pm10, pivot_data_pm10, scaler_pm10, mean_wind_direction, metrics_pm10 = train_model(data, selected_season, target='PM10 (µg/m³)')
        except KeyError as e:
            st.error(f"모델 학습 중 오류 발생: {e}")
            st.stop()

        st.write(f"{selected_season} 평균 풍향 (베이징): {mean_wind_direction:.2f}°")

        # 모델 성능 가로 출력 및 클릭 상세 정보
        st.write("### 모델 성능 (PM10)")
        cols = st.columns(4)  # 4개 군집을 가로로 배치
        for cluster, col in enumerate(cols):
            with col:
                with st.expander(get_cluster_description(cluster, is_pm25=False)):
                    st.write(f"혼동 행렬:\n{metrics_pm10[cluster]['혼동 행렬']}")
                    st.write(f"정확도: {metrics_pm10[cluster]['정확도']:.4f}")
                    st.write(f"정밀도: {metrics_pm10[cluster]['정밀도']:.4f}")
                    st.write(f"재현율: {metrics_pm10[cluster]['재현율']:.4f}")
                    st.write(f"F1 점수: {metrics_pm10[cluster]['F1 점수']:.4f}")
                    st.write(f"R² 스코어: {metrics_pm10[cluster]['R² 스코어']:.4f}")

        beijing_pm10 = float(st.number_input("베이징 PM10 (µg/m³)", min_value=0.0, max_value=500.0, value=75.0, step=1.0, key="pm10"))

        if st.button("예측 및 이동 경로 보기", key="pm10_button"):
            current_predictions = predict_values(models_pm10, kmeans_pm10, scaler_pm10, beijing_pm10, selected_wind_speed, mean_wind_direction)
            if current_predictions:
                current_predictions['Beijing'] = beijing_pm10

                future_predictions = {}
                for city, pm10 in current_predictions.items():
                    lat, lon = city_coords[city][1], city_coords[city][0]
                    new_lat, new_lon, new_pm10 = calculate_movement_and_concentration(
                        lat, lon, pm10, selected_wind_speed, mean_wind_direction, days=days_after
                    )
                    future_predictions[city] = max(new_pm10, 0)

                change_dict = {city: future_predictions[city] - current_predictions[city] for city in current_predictions.keys()}

                pred_df = pd.DataFrame({
                    '도시': list(current_predictions.keys()),
                    '현재 PM10 (µg/m³)': list(current_predictions.values()),
                    f'{days_after}일 후 PM10 (µg/m³)': list(future_predictions.values()),
                    '변화량 (µg/m³)': [change_dict[city] for city in current_predictions.keys()],
                    '경도': [city_coords[city][0] for city in current_predictions.keys()],
                    '위도': [city_coords[city][1] for city in current_predictions.keys()]
                })

                pred_df[['등급', '색상']] = pred_df['현재 PM10 (µg/m³)'].apply(lambda x: pd.Series(assign_cluster(x, is_pm25=False)))

                beijing_lat, beijing_lon = city_coords['Beijing'][1], city_coords['Beijing'][0]
                new_lat, new_lon, new_beijing_pm10 = calculate_movement_and_concentration(
                    beijing_lat, beijing_lon, beijing_pm10, selected_wind_speed, mean_wind_direction, days=days_after
                )

                movement_df = pd.DataFrame({
                    '위도': [beijing_lat, new_lat],
                    '경도': [beijing_lon, new_lon],
                    '도시': ['Beijing', f'Beijing ({days_after}일 후 예측 이동 경로)'],
                    '현재 PM10 (µg/m³)': [beijing_pm10, beijing_pm10],
                    f'{days_after}일 후 PM10 (µg/m³)': [beijing_pm10, new_beijing_pm10]
                })
                movement_df[['등급', '색상']] = movement_df['현재 PM10 (µg/m³)'].apply(lambda x: pd.Series(assign_cluster(x, is_pm25=False)))

                st.write("### 도시별 현재 PM10 및 변화량")
                st.dataframe(pred_df[['도시', '현재 PM10 (µg/m³)', f'{days_after}일 후 PM10 (µg/m³)', '변화량 (µg/m³)', '등급']])

                fig = px.scatter_mapbox(
                    pred_df, 
                    lat="위도", 
                    lon="경도", 
                    size="현재 PM10 (µg/m³)", 
                    color="등급", 
                    color_discrete_map={"좋음": "green", "보통": "blue", "나쁨": "orange", "매우 나쁨": "red"},
                    hover_name="도시", 
                    hover_data={"현재 PM10 (µg/m³)": True, f"{days_after}일 후 PM10 (µg/m³)": True, "변화량 (µg/m³)": True, "등급": True, "위도": False, "경도": False},
                    size_max=30,
                    zoom=2,
                    mapbox_style="open-street-map",
                    title=f"{selected_season} - 베이징 PM10 = {beijing_pm10} µg/m³, 풍속 = {selected_wind_speed:.2f} m/s, 풍향 = {mean_wind_direction:.2f}°"
                )

                fig.add_trace(px.line_mapbox(
                    movement_df,
                    lat="위도",
                    lon="경도",
                    hover_name="도시",
                    hover_data={"현재 PM10 (µg/m³)": True, f"{days_after}일 후 PM10 (µg/m³)": True}
                ).data[0])

                fig.update_traces(textposition="top center")
                st.plotly_chart(fig)
