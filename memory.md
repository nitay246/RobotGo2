# RobotGo2 Agent Memory

## Project Purpose
Behavior-based control stack for Unitree Go2:
- Default `FOLLOW` mode tracks a UWB tag.
- Vision loop (YOLOv8) runs in parallel.
- On target detection, transitions to `APPROACH`.
- On proximity convergence, transitions to `HOLD` (stop + feedback), then returns to `FOLLOW` after cooldown.

This repository is Python-first and hardware-coupled (Go2 SDK, UWB stream, robot camera, obstacle-avoid API).

## Current Repository Snapshot
Top-level Python modules:
- `yolo_follow.py` (main runtime entrypoint)
- `system_init.py` (centralized subsystem initialization)
- `AppConfig.py` (all tuning constants)
- `follow_controller.py` (UWB follow velocity producer thread)
- `target_lock.py` (single-target lock and candidate filters)
- `camera.py` (Unitree camera wrapper)
- `uwb_state_manager.py` (latest UWB message holder)
- `uwb_button_monitor.py` (X-button shutdown monitor)
- `music_player.py` (background audio helper via WebRTC)
- `play_music.py` (standalone audio experiment script)

Other key files:
- `README.md`
- `yolov8n.pt` (YOLO model weights)
- `.gitignore`

Note: `test_best_path_model.py` is not present in the current branch snapshot.

## Architecture (High Level)
Runtime in `yolo_follow.py` is a multi-thread design with shared mutable state (`behavior` dict):

## Architecture Diagram
```mermaid
flowchart TD
  UWB[UWB DDS Topic\nrt/uwbstate] --> SUB[ChannelSubscriber\nUwbButtonMonitor callback]
  SUB --> SM[UwbStateManager\nremote_state]

  MAIN[yolo_follow.py\nMain Loop + State Machine] --> BEH[Shared behavior dict\nmode, vx, wz, target_box, roi_px]
  MAIN --> CAM[Camera wrapper\nVideoClient -> OpenCV frame]
  CAM --> YOLO[Ultralytics YOLO\nmodel.predict]
  YOLO --> DET[Candidate Finder\nPERSON_ONLY / PERSON_ON_BENCH]
  DET --> LOCK[TargetLock\nacquire/update/reset]
  LOCK --> MAIN

  SM --> FC[FollowController Thread\nUWB follow law]
  FC --> BEH

  BEH --> MOT[Motion Executor Thread\n50 Hz avoid.Move(vx, 0, wz)]
  MOT --> ROBOT[Unitree Go2 Motion API\nObstaclesAvoidClient]

  MAIN --> SPORT[SportClient\nHello() in HOLD]

  BTN[UWB X Button] --> SUB
  SUB --> EXIT[Force exit\nos._exit(0)]
```

1. Main thread:
- Signal handling (`SIGINT`)
- System initialization (`SystemInit`)
- State machine loop (`FOLLOW` / `APPROACH` / `HOLD`)
- Camera frame acquisition + YOLO inference
- ROI logic and lock management
- UI display with OpenCV

2. `FollowController` thread (`follow_controller.py`):
- Reads latest UWB estimates from `UwbStateManager`.
- Computes follow velocities when mode is `FOLLOW`.
- Writes `behavior["vx"]` and `behavior["wz"]`.

3. Motion executor thread (`motion_executor` in `yolo_follow.py`):
- Single writer to robot motion command API (`avoid.Move`).
- Sends velocity commands at fixed rate (`rate_hz=50`) from `behavior` dict.

4. UWB callback path:
- `ChannelSubscriber("rt/uwbstate", UwbState_)` in `system_init.py`.
- Callback updates state manager and watches button transitions.
- UWB X-button triggers callback then hard exits via `os._exit(0)`.

5. Optional audio background loop:
- `start_audio_service()` in `yolo_follow.py` (currently commented out in main startup).
- Alternate class in `music_player.py`.

## Main Data Flow
1. UWB DDS message arrives -> `UwbStateManager.remote_state` updated.
2. `FollowController` computes UWB-based velocity targets in `FOLLOW`.
3. Main loop reads camera frame -> YOLO prediction -> target candidate extraction.
4. `TargetLock` acquires/updates a single target box.
5. State machine decides mode transitions and desired velocities.
6. Motion thread continuously pushes `behavior` velocities to `avoid.Move(vx, 0, wz)`.

## Behavior State Machine
Shared state in `behavior` dict:
- `mode`: `FOLLOW` | `APPROACH` | `HOLD`
- `vx`, `wz`: commanded velocities
- `target_box`: locked box or `None`
- `roi_px`: active ROI rectangle in pixels
- `until`, `cooldown_until`: timers/cooldowns

Transitions:
- `FOLLOW` -> `APPROACH`: valid target acquired and cooldown expired.
- `APPROACH` -> `HOLD`: target centered and size error within tolerance.
- `APPROACH` -> `FOLLOW`: lock lost.
- `HOLD` -> `FOLLOW`: hold timer expires, then cooldown starts.

## Core Modules and Responsibilities
`AppConfig.py`
- Central constants for camera timeout, YOLO thresholds, ROI, control gains, lock params, timing.
- `DETECTION_MODE` strategy switch:
  - `"PERSON_ONLY"` (testing)
  - `"PERSON_ON_BENCH"` (production intent)

`system_init.py`
- Encapsulates setup for Unitree comms, UWB subscriber, sport client, obstacle avoid client, camera, YOLO model, lock config.
- Returns initialized handles for main runtime.

`follow_controller.py`
- Pure UWB follow control law with deadbands and slowdown zones.
- Writes to shared `behavior` only when mode is `FOLLOW`.

`target_lock.py`
- Candidate shape: `(confidence, (x1, y1, x2, y2))`.
- IoU-based tracking with miss budget (`lock_max_miss_fr`).
- Optional ROI preference during initial acquire.
- Provides two candidate-finder strategies:
  - `find_person_candidates`
  - `find_person_on_bench_candidates`

`camera.py`
- Wrapper over Unitree `VideoClient`.
- Converts JPEG bytes from SDK to OpenCV BGR frame.

`uwb_state_manager.py`
- Stores latest `UwbState_` sample as `remote_state`.

`uwb_button_monitor.py`
- Detects X-button edge via bitmask (`1 << 2`).
- Invokes callback and then force exits process.

`yolo_follow.py`
- Main orchestrator and runtime state machine.
- Creates motion thread and follow controller thread.
- Handles visualization, FPS display, and cleanup.

## External Dependencies
Python stdlib:
- `asyncio`, `signal`, `threading`, `time`, `math`, `logging`, `os`, `sys`, `dataclasses`, `typing`

Third-party / robotics:
- `opencv-python` (`cv2`)
- `numpy`
- `ultralytics` (YOLO)
- `unitree_sdk2py`
- `go2_webrtc_driver`
- `aiortc`

Model/artifacts:
- `yolov8n.pt` in project root (loaded by literal path).

## Runtime Requirements
- Unitree Go2 reachable and configured for local communication.
- UWB topic available: `rt/uwbstate`.
- Camera stream available through Unitree SDK video client.
- YOLO model file present in working directory.
- OpenCV GUI available (desktop/X forwarding if remote).

## Launch
Primary entrypoint:
- `python yolo_follow.py`

Operational behavior:
- Starts SDK comms, subscriber, control threads, camera, YOLO, and lock subsystem.
- Press `q` in OpenCV window or send Ctrl+C to stop.

## Important Configuration Knobs
From `AppConfig.py`:
- Detection filtering: `MIN_CONF`, `MIN_BOX_FRAC`, `DETECTION_MODE`
- ROI: `ROI_NORM`
- Approach control: `CENTER_TOL`, `SIZE_TOL`, `K_VX_FWD`, `K_VX_BACK`, `MAX_WZ`
- Follow control: `DEAD_BAND_D`, `DEAD_BAND_O`, `DIST_SLOWDOWN`, `SLOWDOWN_ANGLE`
- Safety/timing: `HOLD_SECONDS`, `COOLDOWN_SECONDS`

## Concurrency and Safety Notes
- `behavior` is shared mutable state across threads with no explicit lock.
- Pattern is simple and likely acceptable for coarse control, but race conditions are possible.
- Only motion executor should call `avoid.Move` (this invariant is documented in code and currently respected).
- `UwbButtonMonitor` uses `os._exit(0)`, which bypasses graceful teardown.

## Known Issues / Code Smells
- `yolo_follow.py` imports `ThreadPoolExecutor` and `TargetLockConfig` but does not use them.
- `system_init.py` imports `VideoClient` and `time` but does not use them directly.
- `play_music.py` contains duplicate track add logic and appears experimental.
- Audio service startup in `yolo_follow.py` is commented out, so bark/audio path is effectively disabled.
- Comment in `AppConfig.py` references `app_config.py` while file name is `AppConfig.py`.
- Hard-coded robot IP in audio code paths (`192.168.123.161`).

## Git and Artifact Strategy
Current `.gitignore` covers:
- Python caches, venvs, build artifacts, logs, local env files, editor files.
- ML artifact patterns (`*.pt`, `*.onnx`, `*.engine`) are present but commented.

If model weights should not be versioned, uncomment `*.pt` and consider moving model assets to release storage.

## Fast Context for Future Agents
If token budget is tight, read in this order:
1. `memory.md` (this file)
2. `AppConfig.py` (behavior constants)
3. `yolo_follow.py` (main flow + state machine)
4. `system_init.py` (initialization and external interfaces)
5. `target_lock.py` + `follow_controller.py` (decision/control internals)

## Suggested Next Technical Improvements
- Add `requirements.txt` or `pyproject.toml` for reproducible installs.
- Replace hard process exit with graceful shutdown path.
- Add thread-safe wrapper or lock around `behavior` mutations.
- Add smoke tests for state transitions using mocked UWB and YOLO outputs.
- Parameterize robot IP and model path via env vars or CLI.
