import os
import subprocess

def run_train_scripts(directory="train"):
    # 현재 실행 중인 파일명
    current_script = os.path.basename(__file__)
    
    # train 폴더 내 모든 .py 파일 가져오기 (현재 파일 제외)
    script_files = sorted([f for f in os.listdir(directory) if f.endswith(".py") and f != current_script])
    
    if not script_files:
        print("No training scripts found in the directory.")
        return
    
    for script in script_files:
        script_path = os.path.join(directory, script)
        print(f"Running: {script_path}")
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        
        # 실행 결과 출력
        print("Output:")
        print(result.stdout)
        print("Error:")
        print(result.stderr)
        print("-" * 50)

if __name__ == "__main__":
    run_train_scripts()