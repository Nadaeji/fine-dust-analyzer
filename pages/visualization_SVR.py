import pandas as pd
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import plotly.express as px
import streamlit as st
import math

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("./data/pm25_pm10_merged_wind.csv")
        data['Date'] = pd.to_datetime(data['Date'])
        data = data[data['Date'].dt.year >= 2019]
        data = data[(data['PM2.5 (µg/m³)'] > 0) & (data['PM10 (µg/m³)'] > 0)]
        data['Month'] = data['Date'].dt.month
        data['Season'] = data['Month'].apply(
            lambda x: '봄' if x in [3, 4, 5] else
                      '여름' if x in [6, 7, 8] else
                      '가을' if x in [9, 10, 11] else
                      '겨울'
        )
        summer_mask = data['Season'] == '여름'
        data.loc[summer_mask, 'Wind Direction (degrees)'] = 135
        print("데이터 열 이름:", data.columns.tolist())
        return data
    except FileNotFoundError:
        st.error("데이터 파일을 찾을 수 없습니다. 올바른 경로를 확인하세요.")
        return None

# 군집 등급 지정
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
    else:
        if value <= 30:
            return "좋음", "green"
        elif value <= 80:
            return "보통", "blue"
        elif value <= 150:
            return "나쁨", "orange"
        else:
            return "매우 나쁨", "red"

# 미세먼지 이동 경로 및 농도 변화 계산
def calculate_movement_and_concentration(lat, lon, value, wind_speed, wind_direction, days=1, decay_rate=0.05):
    wind_rad = math.radians(90 - wind_direction)
    distance = wind_speed * days * 24 * 3600 / 1000
    delta_lat = (distance * math.cos(wind_rad)) / 111
    delta_lon = (distance * math.sin(wind_rad)) / (111 * math.cos(math.radians(lat)))
    new_lat = lat + delta_lat
    new_lon = lon + delta_lon
    total_decay = (1 - decay_rate) ** days
    new_value = value * total_decay
    return new_lat, new_lon, new_value

# 회귀 평가 함수
def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

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

# 계절별 SVR 모델 학습 (MultiOutputRegressor 사용)
def train_model_svr(data, season, target='PM2.5 (µg/m³)'):
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
    ).dropna().reset_index()
    
    X = pivot_data[[(target, 'Beijing'), ('Wind Speed (m/s)', 'Beijing'), ('Wind Direction (degrees)', 'Beijing')]]
    y_columns = [(target, city) for city in city_coords.keys() if city != 'Beijing']
    y = pivot_data[y_columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # MultiOutputRegressor로 SVR 학습
    svr = SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale')
    svr_model = MultiOutputRegressor(svr)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    svr_model.fit(X_train, y_train)

    # 회귀 평가 지표 계산
    y_pred = svr_model.predict(X_test)
    mse, rmse, mae, r2 = evaluate_regression(y_test, y_pred)
    metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

    mean_wind_direction = seasonal_data[seasonal_data['City'] == 'Beijing']['Wind Direction (degrees)'].mean()

    return svr_model, X_test, y_test, pivot_data, scaler, mean_wind_direction, metrics

# 예측 함수
def predict_values_svr(model, scaler, beijing_value, wind_speed, wind_direction):
    beijing_input = np.array([[beijing_value, wind_speed, wind_direction]])
    beijing_scaled = scaler.transform(beijing_input)
    prediction = model.predict(beijing_scaled)[0]  # MultiOutputRegressor는 2D 배열 반환
    city_names = [city for city in city_coords.keys() if city != 'Beijing']
    return dict(zip(city_names, prediction))

# Streamlit 앱
st.title("베이징 기반 미세먼지 예측 및 이동 경로 시각화 (계절별 SVR)")

# 데이터 로드
data = load_data()

if data is not None:
    seasons = ['봄', '여름', '가을', '겨울']
    selected_season = st.selectbox("계절을 선택하세요", seasons)

    tab1, tab2 = st.tabs(["PM2.5 예측", "PM10 예측"])

    wind_speed_options = [round(x * 0.5, 1) for x in range(0, 11)]
    selected_wind_speed = st.selectbox("풍속을 선택하세요 (m/s)", wind_speed_options, index=4)
    days_after = int(st.number_input("몇 일 뒤의 변화를 예측할까요? (일)", min_value=1, max_value=30, value=3, step=1))

    # PM2.5 탭
    with tab1:
        st.subheader(f"{selected_season} 계절 기반 PM2.5 예측 및 이동 경로")
        
        try:
            svr_pm25, X_test_pm25, y_test_pm25, pivot_data_pm25, scaler_pm25, mean_wind_direction, metrics_pm25 = train_model_svr(data, selected_season, target='PM2.5 (µg/m³)')
        except KeyError as e:
            st.error(f"모델 학습 중 오류 발생: {e}")
            st.stop()

        st.write(f"{selected_season} 평균 풍향 (베이징): {mean_wind_direction:.2f}°")
        st.write("### 모델 성능 (PM2.5)")
        st.write(f"MSE: {metrics_pm25['MSE']:.4f}")
        st.write(f"RMSE: {metrics_pm25['RMSE']:.4f}")
        st.write(f"MAE: {metrics_pm25['MAE']:.4f}")
        st.write(f"R² 스코어: {metrics_pm25['R2']:.4f}")

        beijing_pm25 = float(st.number_input("베이징 PM2.5 (µg/m³)", min_value=0.0, max_value=300.0, value=30.0, step=1.0, key="pm25"))

        if st.button("예측 및 이동 경로 보기", key="pm25_button"):
            current_predictions = predict_values_svr(svr_pm25, scaler_pm25, beijing_pm25, selected_wind_speed, mean_wind_direction)
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
        
        try:
            svr_pm10, X_test_pm10, y_test_pm10, pivot_data_pm10, scaler_pm10, mean_wind_direction, metrics_pm10 = train_model_svr(data, selected_season, target='PM10 (µg/m³)')
        except KeyError as e:
            st.error(f"모델 학습 중 오류 발생: {e}")
            st.stop()

        st.write(f"{selected_season} 평균 풍향 (베이징): {mean_wind_direction:.2f}°")
        st.write("### 모델 성능 (PM10)")
        st.write(f"MSE: {metrics_pm10['MSE']:.4f}")
        st.write(f"RMSE: {metrics_pm10['RMSE']:.4f}")
        st.write(f"MAE: {metrics_pm10['MAE']:.4f}")
        st.write(f"R² 스코어: {metrics_pm10['R2']:.4f}")

        beijing_pm10 = float(st.number_input("베이징 PM10 (µg/m³)", min_value=0.0, max_value=500.0, value=75.0, step=1.0, key="pm10"))

        if st.button("예측 및 이동 경로 보기", key="pm10_button"):
            current_predictions = predict_values_svr(svr_pm10, scaler_pm10, beijing_pm10, selected_wind_speed, mean_wind_direction)
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