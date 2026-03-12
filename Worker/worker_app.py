import os
import subprocess
import ray

BASE_DIR = "Z:\\"
INPUT_DIR = os.path.join(BASE_DIR, "Input_video")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output_video")
#RESULT_DIR = os.path.join(BASE_DIR, "result")
#MODEL_DIR = os.path.join(BASE_DIR, "ML_Models")

# Ensure shared folders exist (mounted or NFS-shared)
#os.makedirs(INPUT_DIR, exist_ok=True)
#os.makedirs(OUTPUT_DIR, exist_ok=True)
#os.makedirs(RESULT_DIR, exist_ok=True)
#os.makedirs(MODEL_DIR, exist_ok=True)

# Connect to the Ray head node
ray.init(address="auto", ignore_reinit_error=True)

@ray.remote(resources={"heavy": 2, "GPU": 1})
def run_ml_pipeline(video_path):
    """
    Executes main.py on the worker node for the given video file.
    """
    try:
        print(f"?? Worker processing: {video_path}")
        subprocess.run(["python", "C:\\Users\\Window\\Desktop\\Ray_Project\\main.py"], check=True)
        return {"status": "success", "file": os.path.basename(video_path)}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print("? Worker node connected and ready to receive Ray tasks.")
    print(f"?? Shared folders: {INPUT_DIR}, {OUTPUT_DIR}")
    while True:
        pass  # Keep process alive for Ray task reception
