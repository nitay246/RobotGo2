# Unitree Go2 – UWB-Based Following with Vision-Based Chair Approach


## Overview

This project implements a behavior system for the Unitree Go2 robot that combines UWB-based following with real-time vision.

In its default FOLLOW mode, the robot continuously tracks a wearable UWB tag, using distance and orientation estimates to generate smooth motion commands. At the same time, a camera-based vision pipeline runs in parallel and searches for a specific visual target (a chair) using a YOLOv8 object detection model.

When a valid chair is detected, the system stabilizes the target selection using a lightweight target-locking mechanism and transitions to an APPROACH mode. In this mode, motion control is driven by visual feedback, allowing the robot to align itself and move toward the target in a controlled manner. Once the desired proximity is reached, the robot enters a HOLD state, stops its motion, and provides feedback through a gesture and audio cue. After a short delay, the system returns to FOLLOW mode and resumes UWB-based tracking.

An emergency shutdown mechanism is available via the UWB controller’s X button, allowing the system to be stopped immediately at any time. All major thresholds, motion limits, and timing parameters are centralized in a configuration module, enabling fast tuning without modifying core logic.
