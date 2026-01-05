# Unitree Go2 – UWB-Based Following with Vision-Based Object Approach

## Overview

This project implements a behavior-based control system for the Unitree Go2 robot that combines UWB-based following with real-time vision perception.

In its default **FOLLOW** mode, the robot continuously tracks a wearable UWB tag, using distance and orientation estimates to generate smooth motion commands. In parallel, a camera-based vision pipeline runs continuously and searches for a specific visual object using a YOLOv8 object detection model.

When a valid object is detected, the system stabilizes the target selection using a lightweight target-locking mechanism and transitions to an **APPROACH** mode. In this mode, motion control is driven by visual feedback, allowing the robot to align itself and move toward the target in a controlled and stable manner. Once the desired proximity is reached, the robot enters a **HOLD** state, stops its motion, and provides user feedback through a gesture and an audio cue. After a short delay, the system returns to **FOLLOW** mode and resumes UWB-based tracking.

An emergency shutdown mechanism is available via the UWB controller’s X button, allowing the system to be stopped immediately at any time. All major thresholds, motion limits, and timing parameters are centralized in a configuration module, enabling fast tuning without modifying core logic.

## Motivation & Goal

Autonomous mobile robots operating in real-world environments must handle unreliable sensing, communication constraints, and dynamic human interaction. Relying on a single sensing modality is often insufficient, as UWB-based tracking lacks semantic awareness, while vision-based perception is sensitive to noise, occlusions, and environmental conditions.

The goal of this project is to design a robust behavior-driven system that fuses UWB and vision data to enable smooth following, reliable object approach, and safe state transitions. By combining complementary sensing modalities and clear behavioral states, the system aims to provide intuitive human–robot interaction while maintaining real-time responsiveness, safety, and extensibility.
