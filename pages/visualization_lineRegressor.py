import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import numpy as np

# 회귀 평가 지표 계산 함수
def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

# 데이터 로드 및 전처리
@st.cache_data
def load_data():
    data = pd.read_csv("./data/pm25_pm10_merged.csv")  # 파일 경로 수정 필요
    data['Date'] = pd.to_datetime(data['Date'])
    # 2015년 ~ 2018년 데이터 삭제
    data = data[data['Date'].dt.year >= 2019]
    # 월과 계절 컬럼 추가
    data['Month'] = data['Date'].dt.month
    def get_season(month):
        if month in [3, 4, 5]:
            return "봄"
        elif month in [6, 7, 8]:
            return "여름"
        elif month in [9, 10, 11]:
            return "가을"
        else:
            return "겨울"
    data['Season'] = data['Month'].apply(get_season)
    return data

# 계절별 이상치 제거 함수 (IQR 기반)
def remove_outliers_by_season(df, column, factor=1.5):
    # 각 계절 그룹별로 이상치 제거
    def filter_outliers(group):
        Q1 = group[column].quantile(0.25)
        Q3 = group[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return group[(group[column] >= lower_bound) & (group[column] <= upper_bound)]
    return df.groupby("Season", group_keys=False).apply(filter_outliers)

# 모델 학습
def train_model(data):
    # 피벗 테이블 생성 후 결측치 제거
    pivot_data = data.pivot(index='Date', columns='City', values='PM2.5 (µg/m³)').reset_index().dropna()
    # 피벗 데이터에서 0 값 제거 (모든 주요 도시가 0이면 제거)
    pivot_data = pivot_data[(pivot_data[['Beijing', 'Seoul', 'Tokyo', 'Delhi', 'Bangkok']] != 0).all(axis=1)]
    X = pivot_data[['Beijing']]
    y = pivot_data[['Seoul', 'Tokyo', 'Delhi', 'Bangkok']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test, pivot_data

# 예측 함수
def predict_pm25(model, beijing_pm25):
    input_value = [[beijing_pm25]]
    predicted_pm25 = model.predict(input_value)
    cities = ['Seoul', 'Tokyo', 'Delhi', 'Bangkok']
    return dict(zip(cities, predicted_pm25[0]))

# 도시 좌표 딕셔너리
city_coords = {
    'Seoul': (37.5665, 126.978),
    'Tokyo': (35.6895, 139.6917),
    'Beijing': (39.9042, 116.4074),
    'Delhi': (28.7041, 77.1025),
    'Bangkok': (13.7563, 100.5018)
}

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

# Streamlit 앱 제목
st.title("Beijing PM2.5 기반 도시별 미세먼지 예측 및 시간별 지도")

# 데이터 로드
data = load_data()

# 계절 선택 (봄, 여름, 가을, 겨울)
selected_season = st.selectbox("계절을 선택하세요", ["봄", "여름", "가을", "겨울"])
# 선택된 계절에 해당하는 데이터만 필터링
filtered_data = data[data['Season'] == selected_season]

# 0 값 제거 (해당 계절 내에서)
filtered_data = filtered_data[filtered_data["PM2.5 (µg/m³)"] != 0]

# 이상치 제거 여부 선택: 먼저 0 값을 제거한 후 계절별로 이상치 제거 적용
if st.checkbox("이상치 제거 적용"):
    filtered_data = remove_outliers_by_season(filtered_data, "PM2.5 (µg/m³)")

# 모델 학습 (선택한 계절의 데이터로 학습)
model, X_test, y_test, pivot_data = train_model(filtered_data)

st.subheader("베이징 PM2.5 값을 입력해 예측")
beijing_pm25 = st.number_input("Beijing PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)

# 계절별 이상치/노이즈 확인 (추가 기능)
if st.checkbox("계절별 PM2.5 이상치 확인"):
    fig_box = px.box(data, x="Season", y="PM2.5 (µg/m³)", color="Season",
                     title="계절별 PM2.5 분포 및 이상치 확인")
    st.plotly_chart(fig_box)

if st.button("예측하기"):
    predictions = predict_pm25(model, beijing_pm25)
    predictions['Beijing'] = beijing_pm25  # 입력값 포함

    # 예측 데이터프레임 생성
    pred_df = pd.DataFrame({
        'City': list(predictions.keys()),
        'PM2.5 (µg/m³)': list(predictions.values()),
        'Latitude': [city_coords[city][0] for city in predictions.keys()],
        'Longitude': [city_coords[city][1] for city in predictions.keys()]
    })

    # 등급과 색상 추가
    pred_df[['Grade', 'Color']] = pred_df['PM2.5 (µg/m³)'].apply(lambda x: pd.Series(get_grade(x)))

    # 지도 시각화 (Plotly Mapbox)
    fig = px.scatter_mapbox(
        pred_df, 
        lat="Latitude", 
        lon="Longitude", 
        size="PM2.5 (µg/m³)", 
        color="Grade", 
        color_discrete_map={"좋음": "green", "보통": "blue", "나쁨": "orange", "매우 나쁨": "red"},
        hover_name="City", 
        hover_data={"PM2.5 (µg/m³)": True, "Grade": True, "Latitude": False, "Longitude": False},
        text="Grade",
        size_max=30,
        zoom=2,
        mapbox_style="open-street-map",
        title=f"Beijing PM2.5 = {beijing_pm25} µg/m³일 때 예측 ({selected_season})"
    )
    fig.update_traces(textposition="top center")

    # 테스트 데이터에 대한 예측값 계산
    y_pred_full = model.predict(X_test)

    # 서울에 대한 회귀 평가 지표 계산 (다른 도시도 추가 가능)
    y_test_seoul = y_test['Seoul']
    y_pred_seoul = y_pred_full[:, 0]  # 서울 예측값 (첫 번째 열)
    mse, rmse, mae, r2 = evaluate_regression(y_test_seoul, y_pred_seoul)

    # 평가 결과 텍스트
    eval_text = (
        f"- Mean Squared Error (MSE): {mse:.2f}  \n"
        f"- Root Mean Squared Error (RMSE): {rmse:.2f}  \n"
        f"- Mean Absolute Error (MAE): {mae:.2f}  \n"
        f"- R² Score: {r2:.2f}"
    )

    st.markdown(eval_text)
    st.plotly_chart(fig)
