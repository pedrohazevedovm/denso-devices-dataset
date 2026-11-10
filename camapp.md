# Headless Android IP Camera for Robotic Visual Servoing

An embedded, headless Android IP Camera application written in Kotlin designed specifically for robotic visual servoing dataset collection.

The application turns an Android smartphone or single-board computer into a deterministic, zero-latency network vision sensor. It runs an embedded HTTP server on port `8080`, forces the Camera2 HAL into strict manual control (bypassing all automated 3A algorithms and ISP image enhancements), and serves 1080p JPEG frames directly from an `ImageReader` pipeline.

---

## Key Features

- **Strict Determinism**: Disables all OEM 3A algorithms (Auto-Focus, Auto-Exposure, Auto-White Balance) to ensure repeatable illumination and focal planes across datasets.
- **Zero Image Post-Processing**: Disables optical image stabilization (OIS), electronic stabilization (EIS), spatial/temporal noise reduction, edge sharpening, and chromatic aberration correction for unadulterated sensor output.
- **Embedded HTTP Server**: Runs [NanoHTTPD](https://github.com/NanoHttpd/nanohttpd) (`2.3.1`) directly on the device on port `8080`.
- **Low-Latency Image Pipeline**: Uses a dedicated `ImageReader` configured for `ImageFormat.JPEG` at 1080p (1920x1080) with blocking thread synchronization.
- **Runtime Reconfiguration**: Starts in `"auto"` mode (3A + OIS/EIS) by default and supports dynamically switching to `"manual"` mode (focus distance, ISO sensitivity, exposure time, OIS/EIS) on the live capture session via `POST /config`.
- **Robot Arm Mounting & Headless Operation**: Enforces `FLAG_KEEP_SCREEN_ON` and holds a `PowerManager.PARTIAL_WAKE_LOCK` to prevent device sleep during long data collection runs.
- **On-Screen Telemetry HUD**: Displays local IP address, port, live sensor parameters (ISO, exposure ms, focus diopters), and total captures served.

---

## Camera2 Manual Hardware Configuration

The camera pipeline is configured in `CameraController.kt` with absolute manual overrides when `control_mode` is `"manual"`:

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
  - `X-Camera-Metadata`: Compact JSON string containing the frame-synchronized hardware metadata keys from `TotalCaptureResult`:
    - `sensor_exposure_time` (ns)
    - `sensor_sensitivity` (ISO)
    - `sensor_timestamp` (hardware monotonic ns)
    - `sensor_rolling_shutter_skew` (ns duration from first row to last row readout)
    - `lens_focus_distance` (diopters)
    - `lens_focal_length` (mm)
    - `lens_aperture` (f-number)
    - `lens_state` (`0` = `STATIONARY`, `1` = `MOVING`)
    - `lens_intrinsic_calibration` (`[fx, fy, cx, cy, skew]` in active-array pixels)
    - `camera_matrix_k` (9 values in ROS row-major order `[fx, 0, cx, 0, fy, cy, 0, 0, 1]`, scaled to the JPEG output)
    - `camera_matrix_width`, `camera_matrix_height`
    - `control_ae_mode`, `control_af_mode`, `control_awb_mode`
    - `color_correction_mode`
    - `color_correction_gains` (`{"red", "green_even", "green_odd", "blue"}`)
    - `color_correction_transform` (3x3 matrix)
    - `lens_optical_stabilization_mode`, `control_video_stabilization_mode`
    - `noise_reduction_mode`, `edge_mode`, `color_correction_aberration_mode`
    - `scaler_crop_region` (`{"left", "top", "right", "bottom"}`)
    - `jpeg_quality`
  - Individual convenience headers: `X-Camera-Focus-Distance`, `X-Camera-ISO`, `X-Camera-Exposure-Time-Ns`, `X-Camera-Exposure-Time-Ms`, `X-Camera-Sensor-Timestamp`.

#### Optional Query Parameter:
- `GET /capture?format=json`: Returns an `application/json` payload containing the metadata object and the JPEG image encoded in Base64 (`image_base64`).

---

### 2. `POST /config`
Dynamically updates the camera parameters on the fly without closing the capture session. Supports switching between strict deterministic `"manual"` mode and standard Android `"auto"` mode (3A + ISP filters), as well as toggling Optical (`ois_enabled`) and Electronic (`eis_enabled`) stabilization.

- **Method**: `POST`
- **Path**: `/config`
- **Content-Type**: `application/json`

#### Request Body (Manual Calibrado):
```json
{
  "control_mode": "manual",
  "focus_distance": 1.8621974,
  "iso": 1141,
  "exposure_time_ns": 16666660,
  "ois_enabled": false,
  "eis_enabled": false
}
```

#### Request Body (Auto):
```json
{
  "control_mode": "auto",
  "ois_enabled": true,
  "eis_enabled": true
}
```

#### Response:
```json
{
  "status": "success",
  "message": "Camera parameters updated",
  "applied_settings": {
    "control_mode": "manual",
    "auto_mode": false,
    "focus_distance": 1.8621974,
    "iso": 1141,
    "exposure_time_ns": 16666660,
    "exposure_time_ms": 16.66666,
    "ois_enabled": false,
    "eis_enabled": false
  }
}
```

---

### 3. `GET /status`
Returns live telemetry, hardware information, and session capture statistics.

- **Method**: `GET`
- **Path**: `/status`
- **Response**: `200 OK`, `Content-Type: application/json`
