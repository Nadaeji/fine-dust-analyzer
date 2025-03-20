import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from utils.model_cache import ModelCache

# 캐싱된 모델 가져오기
try:
    scaler_pm25 = ModelCache.get_model('rf_Kmean_ensembel/scaler_pm25')
    scaler_pm10 = ModelCache.get_model('rf_Kmean_ensembel/scaler_pm10')
    kmeans_pm25 = ModelCache.get_model('rf_Kmean_ensembel/kmeans_pm25')
    kmeans_pm10 = ModelCache.get_model('rf_Kmean_ensembel/kmeans_pm10')
    season_wind = ModelCache.get_model('rf_Kmean_ensembel/season_wind')
    evaluation_scores_pm25 = ModelCache.get_model('rf_Kmean_ensembel/evaluation_scores_pm25')
    evaluation_scores_pm10 = ModelCache.get_model('rf_Kmean_ensembel/evaluation_scores_pm10')
except KeyError as e:
    st.error(f"필요한 모델이 캐시에 없습니다: {e}")
    st.write("현재 캐시된 모델 목록:", ModelCache.list_cached_models())
    st.stop()

# 앙상블 모델 로드 (PM2.5와 PM10)
seasons = ['봄', '여름', '가을', '겨울']
nearby_cities = ['Seoul', 'Tokyo', 'Delhi', 'Bangkok', 'Busan', 'Daegu', 'Osaka', 
                 'Sapporo', 'Fukuoka', 'Kyoto', 'Almaty', 'Bishkek', 'Dushanbe', 
                 'Kathmandu', 'Yangon', 'Guwahati', 'Ulaanbaatar', 'Irkutsk']
ensemble_models_pm25 = {season: {} for season in seasons}
ensemble_models_pm10 = {season: {} for season in seasons}

for season in seasons:
    for city in nearby_cities:
        try:
            ensemble_models_pm25[season][city] = {
                'Voting': ModelCache.get_model(f'rf_Kmean_ensembel/voting_pm25_{season}_{city}'),
                'Stacking': ModelCache.get_model(f'rf_Kmean_ensembel/stacking_pm25_{season}_{city}')
            }
            ensemble_models_pm10[season][city] = {
                'Voting': ModelCache.get_model(f'rf_Kmean_ensembel/voting_pm10_{season}_{city}'),
                'Stacking': ModelCache.get_model(f'rf_Kmean_ensembel/stacking_pm10_{season}_{city}')
            }
        except KeyError:
            continue  # 캐시에 없는 모델은 건너뜀

# 도시 이름 매핑 (영어 → 한국어)
city_names_kr = {
    "Seoul": "서울", "Tokyo": "도쿄", "Beijing": "베이징", "Delhi": "델리", 
    "Bangkok": "방콕", "Busan": "부산", "Daegu": "대구", "Osaka": "오사카", 
    "Sapporo": "삿포로", "Fukuoka": "후쿠오카", "Kyoto": "교토", "Shanghai": "상하이", 
    "Guangzhou": "광저우", "Chongqing": "충칭", "Wuhan": "우한", "Nanjing": "난징", 
    "Hangzhou": "항저우", "Chengdu": "청두", "Almaty": "알마티", "Bishkek": "비슈케크", 
    "Dushanbe": "두샨베", "Kathmandu": "카트만두", "Yangon": "양곤", "Guwahati": "구와하티", 
    "Ulaanbaatar": "울란바토르", "Irkutsk": "이르쿠츠크"
}

# 도시 좌표 (위도, 경도)
city_coords = {
    "Seoul": [37.5665, 126.9780], "Tokyo": [35.6895, 139.6917], "Beijing": [39.9042, 116.4074],
    "Delhi": [28.7041, 77.1025], "Bangkok": [13.7563, 100.5018], "Busan": [35.1796, 129.0756],
    "Daegu": [35.8704, 128.5911], "Osaka": [34.6937, 135.5023], "Sapporo": [43.0618, 141.3545],
    "Fukuoka": [33.5904, 130.4017], "Kyoto": [35.0116, 135.7681], "Shanghai": [31.2304, 121.4737],
    "Guangzhou": [23.1291, 113.2644], "Chongqing": [29.5630, 106.5516], "Wuhan": [30.5928, 114.3055],
    "Nanjing": [32.0603, 118.7969], "Hangzhou": [30.2741, 120.1551], "Chengdu": [30.5728, 104.0668],
    "Almaty": [43.2220, 76.8512], "Bishkek": [42.8746, 74.5698], "Dushanbe": [38.5481, 68.7864],
    "Kathmandu": [27.7172, 85.3240], "Yangon": [16.8409, 96.1951], "Guwahati": [26.1445, 91.7362],
    "Ulaanbaatar": [47.8864, 106.9057], "Irkutsk": [52.2869, 104.2964],
}

# 예측 함수
def predict_all_cities(season, china_value, pollutant='PM2.5', ensemble_method='Voting'):
    if pollutant == 'PM2.5':
        ensemble_models = ensemble_models_pm25
        scaler = scaler_pm25
        kmeans = kmeans_pm25
    else:
        ensemble_models = ensemble_models_pm10
        scaler = scaler_pm10
        kmeans = kmeans_pm10
    
    if season not in ensemble_models:
        return None, None
    
    wind_x = season_wind['Wind_X'][season]
    wind_y = season_wind['Wind_Y'][season]
    
    # KMeans 클러스터링 예측
    kmeans_input = np.array([[china_value]])
    cluster = kmeans.predict(kmeans_input)[0]
    
    # 앙상블 모델 예측
    input_data = pd.DataFrame([[china_value, wind_x, wind_y]], 
                              columns=[f'{pollutant} (µg/m³)', 'Wind_X', 'Wind_Y'])
    input_scaled = scaler.transform(input_data)
    
    predictions = {}
    for city in nearby_cities:
        if city in ensemble_models[season]:
            model = ensemble_models[season][city][ensemble_method]
            prediction = model.predict(input_scaled)[0]
            predictions[city] = prediction
    return predictions, cluster

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

# Streamlit 인터페이스
st.title("중국 미세먼지가 주변국에 미치는 영향 (KMeans + 앙상블 모델)")

# 탭 생성
tab1, tab2 = st.tabs(["PM2.5 예측", "PM10 예측"])

# PM2.5 탭
with tab1:
    st.header("PM2.5 예측")

    # 계절 선택
    season = st.selectbox("계절 선택 (PM2.5)", seasons, key="pm25_season")

    # 앙상블 방법 선택
    ensemble_method = st.selectbox("앙상블 방법 선택 (PM2.5)", ["Voting", "Stacking"], key="pm25_ensemble")

    # 입력값 수집
    china_pm25 = st.slider("중국 PM2.5 (µg/m³)", 0.0, 200.0, 50.0, key="pm25_input")

    # 예측 및 지도 표시
    predictions, cluster = predict_all_cities(season, china_pm25, pollutant='PM2.5', ensemble_method=ensemble_method)

    if predictions:
        # 예측 결과 섹션
        with st.expander(f"{season}의 주변국 PM2.5 예측 ({ensemble_method})"):
            pred_table_data = {
                "도시": [city_names_kr[city] for city in predictions.keys()],
                "PM2.5 (µg/m³)": [f"{pm25:.2f}" for pm25 in predictions.values()],
                "상태": [get_grade(pm25, 'PM2.5')[0] for pm25 in predictions.values()]
            }
            pred_df = pd.DataFrame(pred_table_data)
            st.dataframe(pred_df, use_container_width=True)
            st.write(f"입력값의 클러스터: {cluster}")

        # 모델 평가 점수 섹션
        with st.expander(f"{season}의 모델 평가 점수 (PM2.5, {ensemble_method})"):
            eval_table_data = {
                "도시": [city_names_kr[city] for city in predictions.keys()],
                "MSE": [evaluation_scores_pm25[season][city][ensemble_method]['MSE'] if city in evaluation_scores_pm25[season] and ensemble_method in evaluation_scores_pm25[season][city] else None for city in predictions.keys()],
                "RMSE": [evaluation_scores_pm25[season][city][ensemble_method]['RMSE'] if city in evaluation_scores_pm25[season] and ensemble_method in evaluation_scores_pm25[season][city] else None for city in predictions.keys()],
                "MAE": [evaluation_scores_pm25[season][city][ensemble_method]['MAE'] if city in evaluation_scores_pm25[season] and ensemble_method in evaluation_scores_pm25[season][city] else None for city in predictions.keys()],
                "R² 스코어": [evaluation_scores_pm25[season][city][ensemble_method]['R²'] if city in evaluation_scores_pm25[season] and ensemble_method in evaluation_scores_pm25[season][city] else None for city in predictions.keys()]
            }
            eval_df = pd.DataFrame(eval_table_data)
            st.dataframe(eval_df, use_container_width=True)

        # 지도 생성
        st.subheader(f"지도 (PM2.5, {ensemble_method})")
        m = folium.Map(location=[35, 120], zoom_start=4)
        
        # 예측값을 지도에 표시
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
        
        st_folium(m, width=700, height=500, key=f"map_pm25_{season}_{ensemble_method}")
    else:
        st.error(f"{season}에 대한 {ensemble_method} 모델이 없습니다 (PM2.5).")

# PM10 탭
with tab2:
    st.header("PM10 예측")

    # 계절 선택
    season = st.selectbox("계절 선택 (PM10)", seasons, key="pm10_season")

    # 앙상블 방법 선택
    ensemble_method = st.selectbox("앙상블 방법 선택 (PM10)", ["Voting", "Stacking"], key="pm10_ensemble")

    # 입력값 수집
    china_pm10 = st.slider("중국 PM10 (µg/m³)", 0.0, 300.0, 75.0, key="pm10_input")

    # 예측 및 지도 표시
    predictions, cluster = predict_all_cities(season, china_pm10, pollutant='PM10', ensemble_method=ensemble_method)

    if predictions:
        # 예측 결과 섹션
        with st.expander(f"{season}의 주변국 PM10 예측 ({ensemble_method})"):
            pred_table_data = {
                "도시": [city_names_kr[city] for city in predictions.keys()],
                "PM10 (µg/m³)": [f"{pm10:.2f}" for pm10 in predictions.values()],
                "상태": [get_grade(pm10, 'PM10')[0] for pm10 in predictions.values()]
            }
            pred_df = pd.DataFrame(pred_table_data)
            st.dataframe(pred_df, use_container_width=True)
            st.write(f"입력값의 클러스터: {cluster}")

        # 모델 평가 점수 섹션
        with st.expander(f"{season}의 모델 평가 점수 (PM10, {ensemble_method})"):
            eval_table_data = {
                "도시": [city_names_kr[city] for city in predictions.keys()],
                "MSE": [evaluation_scores_pm10[season][city][ensemble_method]['MSE'] if city in evaluation_scores_pm10[season] and ensemble_method in evaluation_scores_pm10[season][city] else None for city in predictions.keys()],
                "RMSE": [evaluation_scores_pm10[season][city][ensemble_method]['RMSE'] if city in evaluation_scores_pm10[season] and ensemble_method in evaluation_scores_pm10[season][city] else None for city in predictions.keys()],
                "MAE": [evaluation_scores_pm10[season][city][ensemble_method]['MAE'] if city in evaluation_scores_pm10[season] and ensemble_method in evaluation_scores_pm10[season][city] else None for city in predictions.keys()],
                "R² 스코어": [evaluation_scores_pm10[season][city][ensemble_method]['R²'] if city in evaluation_scores_pm10[season] and ensemble_method in evaluation_scores_pm10[season][city] else None for city in predictions.keys()]
            }
            eval_df = pd.DataFrame(eval_table_data)
            st.dataframe(eval_df, use_container_width=True)

        # 지도 생성
        st.subheader(f"지도 (PM10, {ensemble_method})")
        m = folium.Map(location=[35, 120], zoom_start=4)
        
        # 예측값을 지도에 표시
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
        
        st_folium(m, width=700, height=500, key=f"map_pm10_{season}_{ensemble_method}")
    else:
        st.error(f"{season}에 대한 {ensemble_method} 모델이 없습니다 (PM10).")