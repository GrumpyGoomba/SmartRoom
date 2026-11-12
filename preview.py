
import cv2
from flask import Flask, Response, jsonify, request
from flask import render_template_string

from config import load_config


HTML = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SmartRoom</title>
<style>
body {
    background: #111827;
    color: #f9fafb;
    font-family: Arial, sans-serif;
    max-width: 850px;
    padding: 20px;
    margin: auto;
}
section {
    background: #1f2937;
    padding: 20px;
    margin: 16px 0;
    border-radius: 12px;
}
label { display: block; margin: 12px 0 5px; }
input, select {
    box-sizing: border-box;
    background: #374151;
    color: white;
    border: 1px solid #6b7280;
    padding: 10px;
    border-radius: 6px;
    width: 100%;
}
input[type=checkbox] { width: auto; }
button {
    background: #7c3aed;
    color: white;
    border: 0;
    padding: 13px 24px;
    border-radius: 8px;
    cursor: pointer;
}
img { width: 100%; border-radius: 8px; }
.row { display: flex; gap: 12px; }
.row > div { flex: 1; min-width: 0; }
#message { color: #a5b4fc; }
</style>
</head>
<body>
<h1>SmartRoom Control Panel</h1>
<p id="status">Connecting...</p>

<section>
<h2>Live Camera</h2>
<label>
<input id="enabled" type="checkbox" onchange="togglePreview()">
 Enable camera preview
</label>
<div id="videoArea" style="display:none">
<img id="video">
</div>
<p id="previewMessage">Preview disabled</p>
</section>

<section>
<h2>Configuration</h2>

<label>Camera resolution</label>
<select id="resolution" onchange="resolutionChanged()">
<option value="1280x720">1280 × 720</option>
<option value="960x540">960 × 540</option>
<option value="640x360">640 × 360</option>
<option value="custom">Custom</option>
</select>

<div class="row" id="customFields" style="display:none">
<div><label>Width</label><input id="width" type="number"></div>
<div><label>Height</label><input id="height" type="number"></div>
</div>

<div class="row">
<div>
<label>Camera FPS</label>
<input id="cameraFps" type="number" min="1" max="60">
</div>
<div>
<label>Preview FPS cap</label>
<input id="previewFps" type="number" min="1" max="30">
</div>
</div>

<label>YOLO image size</label>
<select id="imgsz">
<option value="320">320</option>
<option value="416">416</option>
<option value="512">512</option>
<option value="640">640</option>
</select>

<label>YOLO confidence</label>
<input id="confidence" type="number" min="0" max="1" step=".05">

<label>Keypoint confidence</label>
<input id="kpConfidence" type="number" min="0" max="1" step=".05">

<label>Bed region (x1, y1, x2, y2)</label>
<input id="region" type="text">

<label>Transition frames</label>
<input id="transition" type="number" min="1">

<label>Lying-down angle (degrees)</label>
<input id="angle" type="number" min="0" max="90">

<p><button onclick="updateConfig()">UPDATE</button></p>
<p id="message"></p>
</section>

<script>
let config;

async function api(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || "Request failed");
    }
    return data;
}

function resolutionChanged() {
    document.getElementById("customFields").style.display =
        document.getElementById("resolution").value === "custom"
        ? "flex" : "none";
}

function showPreview(enabled) {
    const img = document.getElementById("video");
    document.getElementById("videoArea").style.display =
        enabled ? "block" : "none";
    document.getElementById("previewMessage").textContent =
        enabled ? "" : "Preview disabled";

    if (enabled) {
        if (!img.getAttribute("src")) {
            img.src = "/video_feed";
        }
    } else {
        img.removeAttribute("src");
    }
}

async function togglePreview() {
    const enabled = document.getElementById("enabled").checked;
    try {
        await api("/api/preview", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({enabled})
        });
        showPreview(enabled);
    } catch (error) {
        alert(error.message);
        document.getElementById("enabled").checked = !enabled;
    }
}

async function loadSettings() {
    config = await api("/api/config");

    const c = config.camera;
    const y = config.yolo;
    const b = config.bed;

    const size = c.width + "x" + c.height;
    const presets = ["1280x720", "960x540", "640x360"];

    document.getElementById("resolution").value =
        presets.includes(size) ? size : "custom";

    document.getElementById("width").value = c.width;
    document.getElementById("height").value = c.height;
    document.getElementById("cameraFps").value = c.fps;
    document.getElementById("previewFps").value = config.preview.fps;
    document.getElementById("imgsz").value = y.imgsz;
    document.getElementById("confidence").value = y.confidence;
    document.getElementById("kpConfidence").value =
        y.keypoint_confidence;
    document.getElementById("region").value = b.region.join(", ");
    document.getElementById("transition").value = b.transition_frames;
    document.getElementById("angle").value = b.lying_angle_max;

    resolutionChanged();
}

async function updateConfig() {
    const message = document.getElementById("message");
    message.textContent = "Saving configuration...";

    try {
        const selected = document.getElementById("resolution").value;
        let width, height;

        if (selected === "custom") {
            width = Number(document.getElementById("width").value);
            height = Number(document.getElementById("height").value);
        } else {
            [width, height] = selected.split("x").map(Number);
        }

        const newConfig = {
            camera: {
                width,
                height,
                fps: Number(document.getElementById("cameraFps").value)
            },
            yolo: {
                imgsz: Number(document.getElementById("imgsz").value),
                confidence: Number(
                    document.getElementById("confidence").value
                ),
                keypoint_confidence: Number(
                    document.getElementById("kpConfidence").value
                )
            },
            preview: {
                fps: Number(
                    document.getElementById("previewFps").value
                )
            },
            bed: {
                region: document.getElementById("region")
                    .value.split(",").map(Number),
                transition_frames: Number(
                    document.getElementById("transition").value
                ),
                lying_angle_max: Number(
                    document.getElementById("angle").value
                )
            }
        };

        await api("/api/config", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(newConfig)
        });

        message.textContent = "Configuration saved. Applying...";
    } catch (error) {
        message.textContent = "Error: " + error.message;
    }
}

async function refreshStatus() {
    try {
        const state = await api("/api/status");
        document.getElementById("status").textContent =
            "Detection: " + state.detection +
            " | Camera: " + state.camera;

        if (state.error) {
            document.getElementById("message").textContent =
                "Error: " + state.error;
        } else if (state.detection === "running") {
            const message = document.getElementById("message");
            if (message.textContent.includes("Applying")) {
                message.textContent = "Configuration applied!";
            }
        }

        document.getElementById("enabled").checked =
            state.preview.enabled;

    } catch (error) {
        document.getElementById("status").textContent =
            "Connection unavailable";
    }
}

async function initialize() {
    await loadSettings();
    const state = await api("/api/status");
    document.getElementById("enabled").checked =
        state.preview.enabled;
    showPreview(state.preview.enabled);
    await refreshStatus();
}

initialize().catch(error => {
    document.getElementById("status").textContent =
        "Error: " + error.message;
});
setInterval(refreshStatus, 2000);
</script>
</body>
</html>
"""


def create_app(controller, frames):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML)

    @app.get("/api/config")
    def get_config():
        return jsonify(load_config())

    @app.post("/api/config")
    def update_config():
        try:
            data = request.get_json()
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object")

            controller.request_update(data)
            return jsonify({"status": "applying"}), 202

        except (ValueError, KeyError, TypeError) as error:
            return jsonify({"error": str(error)}), 400

        except RuntimeError as error:
            return jsonify({"error": str(error)}), 409

    @app.get("/api/status")
    def status():
        result = controller.status()
        result["preview"] = frames.status()
        return jsonify(result)

    @app.post("/api/preview")
    def toggle_preview():
        data = request.get_json(silent=True) or {}
        if type(data.get("enabled")) is not bool:
            return jsonify({"error": "Expected boolean enabled"}), 400

        frames.set_enabled(data["enabled"])
        return jsonify(frames.status())

    def stream():
        frames.add_viewer()
        version = -1

        try:
            while True:
                frame, version, enabled = frames.get_next(version)

                if not enabled:
                    break

                if frame is None or version == previous_version:
                    continue

                image = frame.copy()
                h, w = image.shape[:2]

                region = controller.current_config()["bed"]["region"]
                x1, y1, x2, y2 = region

                p1 = (int(x1 * w), int(y1 * h))
                p2 = (
                    min(w - 1, int(x2 * w)),
                    min(h - 1, int(y2 * h))
                )

                cv2.rectangle(image, p1, p2, (255, 0, 255), 3)
                cv2.putText(
                    image,
                    "BED REGION",
                    (p1[0] + 5, p1[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),
                    2
                )

                ok, jpeg = cv2.imencode(
                    ".jpg", image,
                    [cv2.IMWRITE_JPEG_QUALITY, 65]
                )

                if ok:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg.tobytes()
                        + b"\r\n"
                    )

        finally:
            frames.remove_viewer()

    @app.get("/video_feed")
    def video_feed():
        if not frames.status()["enabled"]:
            return Response(status=204)

        return Response(
            stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    return app
