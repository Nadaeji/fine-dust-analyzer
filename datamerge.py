import pandas as pd
import glob
import os

# 저장된 CSV 파일이 있는 폴더 경로
save_dir = "pm25_pm10_data"
output_excel_file = "pm25_pm10_merged.csv"

# 모든 CSV 파일 가져오기
csv_files = sorted(glob.glob(os.path.join(save_dir, "pm25_pm10_*.csv")))

# 빈 리스트 생성 (모든 데이터를 저장)
all_data = []

# 모든 CSV 파일 읽어서 합치기
for file in csv_files:
    df = pd.read_csv(file)
    all_data.append(df)

# 하나의 DataFrame으로 병합
merged_df = pd.concat(all_data, ignore_index=True)

# Excel 파일로 저장
merged_df.to_csv(output_excel_file, index=False)

print(f"✅ 모든 CSV 파일이 {output_excel_file}로 합쳐졌습니다!")
