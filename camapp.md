# Headless Android IP Camera for Robotic Visual Servoing

An embedded, headless Android IP Camera application written in Kotlin designed specifically for robotic visual servoing dataset collection.

The application turns an Android smartphone or single-board computer into a deterministic, zero-latency network vision sensor. It runs an embedded HTTP server on port `8080`, forces the Camera2 HAL into strict manual control (bypassing all automated 3A algorithms and ISP image enhancements), and serves 1080p JPEG frames directly from an `ImageReader` pipeline.

---

## Key Features

- **Strict Determinism**: Disables all OEM 3A algorithms (Auto-Focus, Auto-Exposure, Auto-White Balance) to ensure repeatable illumination and focal planes across datasets.
- **Zero Image Post-Processing**: Disables optical image stabilization (OIS), electronic stabilization (EIS), spatial/temporal noise reduction, edge sharpening, and chromatic aberration correction for unadulterated sensor output.
- **Embedded HTTP Server**: Runs [NanoHTTPD](https://github.com/NanoHttpd/nanohttpd) (`2.3.1`) directly on the device on port `8080`.
- **Low-Latency Image Pipeline**: Uses a dedicated `ImageReader` configured for `ImageFormat.JPEG` at 1080p (1920x1080) with blocking thread synchronization.
- **Runtime Reconfiguration**: Dynamically adjust focus distance, ISO sensitivity, and exposure time on the live capture session via `POST /config`.
- **Robot Arm Mounting & Headless Operation**: Enforces `FLAG_KEEP_SCREEN_ON` and holds a `PowerManager.PARTIAL_WAKE_LOCK` to prevent device sleep during long data collection runs.
- **On-Screen Telemetry HUD**: Displays local IP address, port, live sensor parameters (ISO, exposure ms, focus diopters), and total captures served.

---

## Camera2 Manual Hardware Configuration

The camera pipeline is configured in [`CameraController.kt`](app/src/main/java/com/robotics/camera/CameraController.kt) with absolute manual overrides:

| Parameter | Configuration | Purpose |
| :--- | :--- | :--- |
| `CONTROL_MODE` | `CONTROL_MODE_OFF` | Disables automated scene modes and overrides OEM HAL algorithms. |
| `CONTROL_AF_MODE` | `CONTROL_AF_MODE_OFF` | Locks the physical lens actuator at a fixed focus distance (`LENS_FOCUS_DISTANCE`). |
| `CONTROL_AE_MODE` | `CONTROL_AE_MODE_OFF` | Disables auto-exposure; locks shutter time (`SENSOR_EXPOSURE_TIME`) and gain (`SENSOR_SENSITIVITY`). |
| `CONTROL_AWB_MODE` | `CONTROL_AWB_MODE_OFF` | Disables dynamic white balance calculations to prevent chromatic shifts. |
| `COLOR_CORRECTION_MODE` | `TRANSFORM_MATRIX` | Sets neutral daylight gains (`COLOR_CORRECTION_GAINS = 1.8, 1.0, 1.0, 1.6`) to prevent OEM HAL tint bugs when AWB is off. |
| `LENS_OPTICAL_STABILIZATION_MODE` | `OFF` | Prevents voice-coil physical motion caused by robot arm vibrations. |
| `CONTROL_VIDEO_STABILIZATION_MODE` | `OFF` | Prevents electronic image warping and dynamic frame cropping. |
| `NOISE_REDUCTION_MODE` | `OFF` | Preserves raw spatial noise patterns; eliminates frame-to-frame temporal smearing. |
| `EDGE_MODE` | `OFF` | Disables ISP edge sharpening filters for raw pixel fidelity. |
| `COLOR_CORRECTION_ABERRATION_MODE`| `OFF` | Disables computational chromatic aberration correction. |

---

## HTTP Endpoints

The server listens on **port 8080**. The device IP is displayed on screen when the app opens.

### 1. `GET /capture`
Triggers an immediate frame capture from the active sensor stream and returns the raw JPEG byte stream.

- **Method**: `GET`
- **Path**: `/capture`
- **Response**: `200 OK`, `Content-Type: image/jpeg`
- **Response Headers**:
  - `Cache-Control: no-cache, no-store, must-revalidate`
  - `X-Camera-Focus-Distance`: Current focus distance in diopters ($1 / \text{meters}$)
  - `X-Camera-ISO`: Current ISO sensitivity
  - `X-Camera-Exposure-Time-Ns`: Current integration time in nanoseconds
  - `X-Camera-Exposure-Time-Ms`: Current integration time in milliseconds

#### cURL:
```bash
curl http://<DEVICE_IP>:8080/capture -o frame.jpg
```

---

### 2. `POST /config`
Dynamically updates the manual camera parameters on the fly without closing the capture session.

- **Method**: `POST`
- **Path**: `/config`
- **Content-Type**: `application/json`

#### Request Body:
```json
{
  "focus_distance": 2.5,
  "iso": 200,
  "exposure_time_ns": 10000000
}
```
*Note: `focus_distance` is in diopters ($0.0$ = infinity, $10.0$ = 10cm). `exposure_time_ns` is in nanoseconds ($10{,}000{,}000 = 10\text{ms} = 1/100\text{s}$). If passing milliseconds, use `"exposure_time": 10.0`.*

#### cURL:
```bash
curl -X POST http://<DEVICE_IP>:8080/config \
     -H "Content-Type: application/json" \
     -d '{"focus_distance": 2.5, "iso": 200, "exposure_time_ns": 10000000}'
```

#### Response:
```json
{
  "status": "success",
  "message": "Camera parameters updated",
  "applied_settings": {
    "focus_distance": 2.5,
    "iso": 200,
    "exposure_time_ns": 10000000,
    "exposure_time_ms": 10.0
  }
}
```

---

### 3. `GET /status`
Returns live telemetry, hardware information, and session capture statistics.

- **Method**: `GET`
- **Path**: `/status`
- **Response**: `200 OK`, `Content-Type: application/json`

#### Response:
```json
{
  "status": "online",
  "device_model": "Pixel 7 Pro",
  "android_version": "14",
  "captures_served": 184,
  "current_settings": {
    "focus_distance": 2.5,
    "iso": 200,
    "exposure_time_ns": 10000000,
    "exposure_time_ms": 10.0
  }
}
```

---

## Python Robotic Servoing Client Example

Here is a minimal Python example demonstrating how to stream frames directly into OpenCV matrices without saving to disk:

```python
import cv2
import numpy as np
import requests

CAMERA_URL = "http://192.168.1.150:8080/capture"
CONFIG_URL = "http://192.168.1.150:8080/config"

# Optional: Set manual exposure & focus before starting
requests.post(CONFIG_URL, json={
    "focus_distance": 2.0,       # 0.5 meters
    "iso": 100,                  # Clean low-noise baseline
    "exposure_time_ns": 10000000 # 10 ms (1/100 sec)
})

# Visual Servoing Capture Loop
while True:
    try:
        response = requests.get(CAMERA_URL, timeout=1.0)
        if response.status_code == 200:
            # Decode JPEG buffer directly to OpenCV BGR array
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Metadata from headers
            exp_ms = response.headers.get("X-Camera-Exposure-Time-Ms")
            iso = response.headers.get("X-Camera-ISO")

            # Visual servoing / inference logic here
            cv2.putText(frame, f"ISO: {iso} Exp: {exp_ms}ms", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Robot Arm View", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")

cv2.destroyAllWindows()
```

---

## Project Structure

```
Cam-App/
├── app/
│   ├── build.gradle.kts           # Module build configuration (NanoHTTPD 2.3.1)
│   └── src/main/
│       ├── AndroidManifest.xml    # Permissions, hardware features, cleartext HTTP
│       ├── java/com/robotics/camera/
│       │   ├── MainActivity.kt    # Headless Activity, WakeLock, HUD
│       │   ├── CameraController.kt# Camera2 hardware controller & ImageReader pipeline
│       │   ├── HttpServer.kt      # NanoHTTPD server (port 8080)
│       │   ├── CameraConfig.kt    # Parameter data model & JSON serialization
│       │   └── NetworkUtils.kt    # Local IPv4 network resolver
│       └── res/
│           ├── layout/
│           │   └── activity_main.xml # TextureView & telemetry overlay layout
│           └── values/
│               ├── colors.xml
│               ├── strings.xml
│               └── themes.xml
├── build.gradle.kts               # Root Gradle build script
├── settings.gradle.kts            # Project repositories & module definitions
└── README.md
```

---

## Building and Installing

### Prerequisites
- Android Studio Hedgehog (2023.1.1) or newer
- Android SDK 34 (`minSdk = 24`, `targetSdk = 34`)
- JDK 17 or higher
- Android device with Camera2 `FULL` hardware level support

### Build via Gradle
```bash
# Debug APK
./gradlew assembleDebug

# Install to connected device via ADB
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### Required Permissions
The app requests runtime camera permission upon first launch:
- `android.permission.CAMERA`
- `android.permission.INTERNET`
- `android.permission.ACCESS_WIFI_STATE`
- `android.permission.ACCESS_NETWORK_STATE`
- `android.permission.WAKE_LOCK`

---

## License
Apache 2.0
