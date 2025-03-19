import os
import subprocess

def run_train_scripts():
    # 현재 실행 중인 파일명
    current_script = os.path.basename(__file__)
    
    # 현재 폴더 내 모든 .py 파일 가져오기 (현재 파일 제외)
    script_files = sorted([f for f in os.listdir() if f.endswith(".py") and f != current_script])
    
    if not script_files:
        print("No training scripts found in the directory.")
        return
    
    for script in script_files:
        print(f"Running: {script}")
        result = subprocess.run(["python", script], capture_output=True, text=True)
        
        # 실행 결과 출력
        print("Output:")
        print(result.stdout)
        print("Error:")
        print(result.stderr)
        print("-" * 50)

if __name__ == "__main__":
    run_train_scripts()