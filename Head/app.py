from flask import Flask, render_template, request, jsonify, send_from_directory, Response, send_file, abort
import os
import ray

from main import process_video  # Import the processing function

BASE_DIR = "Z:\\"
INPUT_DIR = os.path.join(BASE_DIR, "Input_video")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output_video")
CHALAN_DIR = os.path.join(BASE_DIR, "generated_challans")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHALAN_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates")

# Initialize Ray
ray.init(address="auto", ignore_reinit_error=True, include_dashboard=True)

# Progress Actor
try:
    progress = ray.get_actor("progress")
except Exception:
    @ray.remote
    class ProgressTracker:
        def __init__(self):
            self.progress = {}

        def update(self, filename, percent):
            try:
                key = str(filename)
                p = int(percent)
                self.progress[key] = p
                print(f"[ProgressTracker] Updated: {key} = {p}%")
            except:
                pass

        def get(self):
            return self.progress

    progress = ProgressTracker.options(name="progress").remote()

@ray.remote(resources={"normal": 1})
def run_ml_pipeline(video_path):
    try:
        filename = os.path.basename(video_path)
        # Initialize progress at 0%
        ray.get(progress.update.remote(filename, 0))

        # Run processing directly, pass the progress actor
        process_video(video_path, progress)

        # Mark progress complete
        ray.get(progress.update.remote(filename, 100))
        print(f"[run_ml_pipeline] Completed: {filename}")
        return {"status": "success", "file": filename}

    except Exception as e:
        try:
            ray.get(progress.update.remote(filename, -1))
        except:
            pass
        return {"status": "error", "message": str(e)}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file found"})

    file = request.files["file"]
    filename = file.filename
    save_path = os.path.join(INPUT_DIR, filename)
    file.save(save_path)

    # Set progress to 0 and launch ML pipeline in background
    ray.get(progress.update.remote(filename, 0))
    run_ml_pipeline.remote(save_path)

    return jsonify({"status": "success", "message": f"Uploaded {filename} successfully!"})


@app.route("/progress")
def get_progress():
    try:
        p = ray.get(progress.get.remote())
        return jsonify(p)
    except:
        return jsonify({})


# -------------------------------------------------------------------
# INPUT VIDEOS
# -------------------------------------------------------------------
@app.route("/list_input_videos")
def list_input_videos():
    files = [f for f in os.listdir(INPUT_DIR)
             if f.lower().endswith((".mp4", ".webm", ".avi", ".mov"))]
    return jsonify({"videos": files})


@app.route("/input_videos/<path:filename>")
def serve_input_video(filename):
    return send_from_directory(INPUT_DIR, filename)


@app.route("/delete_input_video", methods=["POST"])
def delete_input_video():
    data = request.get_json()
    filename = data.get("filename")
    file_path = os.path.join(INPUT_DIR, filename)

    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "File not found"})


# -------------------------------------------------------------------
# OUTPUT VIDEOS
# -------------------------------------------------------------------
@app.route("/list_output_videos")
def list_output_videos():
    files = [f for f in os.listdir(OUTPUT_DIR)
             if f.lower().endswith((".mp4", ".webm", ".avi", ".mov"))]
    return jsonify({"videos": files})


# Serve output videos with support for range requests
@app.route('/output_video/<path:filename>')
def output_video(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return "File not found", 404

    range_header = request.headers.get('Range', None)
    if not range_header:
        return send_file(path, mimetype='video/mp4')

    size = os.path.getsize(path)
    byte1, byte2 = 0, None
    m = range_header.replace("bytes=", "").split("-")

    if len(m) == 2:
        byte1 = int(m[0]) if m[0] else 0
        if m[1]:
            byte2 = int(m[1])

    length = size - byte1 if byte2 is None else byte2 - byte1 + 1

    with open(path, 'rb') as f:
        f.seek(byte1)
        data = f.read(length)

    rv = Response(data, status=206, mimetype="video/mp4", direct_passthrough=True)
    rv.headers.add('Content-Range', f'bytes {byte1}-{byte1 + length - 1}/{size}')
    rv.headers.add('Accept-Ranges', 'bytes')
    rv.headers.add('Content-Length', str(length))

    return rv


# **IMPORTANT: Add alias route to fix your frontend URL**:
@app.route('/output_videos/<path:filename>')
def output_videos(filename):
    return output_video(filename)


@app.route("/delete_output_video", methods=["POST"])
def delete_output_video():
    data = request.get_json()
    filename = data.get("filename")
    file_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "File not found"})


# -------------------------------------------------------------------
# CHALAN PDF SECTION
# -------------------------------------------------------------------
@app.route("/list_chalans")
def list_chalans():
    try:
        files = [f for f in os.listdir(CHALAN_DIR) if f.lower().endswith(".pdf")]
        return jsonify({"chalans": files})
    except Exception as e:
        return jsonify({"chalans": [], "error": str(e)})


@app.route("/chalans/<path:filename>")
def serve_chalan(filename):
    safe_filename = os.path.basename(filename)
    full_path = os.path.join(CHALAN_DIR, safe_filename)

    if os.path.exists(full_path):
        return send_from_directory(CHALAN_DIR, safe_filename)
    else:
        abort(404, "Chalan not found")


# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
