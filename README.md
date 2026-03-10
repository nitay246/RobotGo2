#  🤖 Unitree Go2 – UWB-Based Following with Vision-Based Object Approach 

## Motivation & Goal

The primary goal of this project is to emulate the experience of walking with a dog, where the dog naturally follows its owner, maintains smooth motion, and occasionally reacts to the environment or approaches objects of interest in a safe and intuitive manner.

Inspired by this behavior, the project aims to develop a behavior-driven control system for the Unitree Go2 robot that enables continuous user following, environment-aware object interaction, and controlled transitions between behavioral states. The system is designed to provide a natural companion-like interaction while maintaining real-time responsiveness, safety, and extensibility.

## Overview

This project implements a behavior-based control system for the Unitree Go2 robot that integrates UWB-based localization with real-time vision perception.

In its default **FOLLOW** state, the robot tracks a wearable UWB tag and generates smooth motion commands based on relative distance and orientation. A vision pipeline runs in parallel, continuously analyzing the camera feed to detect predefined objects using a YOLOv8-based object detection model.

Upon detecting a valid object, the system stabilizes the selection through a lightweight target-locking mechanism and transitions to an **APPROACH** state, where motion control is driven by visual feedback. Once the robot reaches the desired proximity, it enters a **HOLD** state, stops its motion, and provides user feedback. After a short delay, the system returns to **FOLLOW** and resumes UWB-based tracking.

An emergency stop mechanism is available via the UWB controller, allowing immediate system shutdown at any time. All motion thresholds, timing parameters, and behavioral constraints are centralized in a configuration module, enabling efficient tuning without modifying the core logic.

## Example Run

The following example demonstrates a typical system execution scenario:
- The robot starts in **FOLLOW** mode and tracks a wearable UWB tag.
- While following, the vision pipeline runs in parallel and monitors the environment.
- Upon detecting a predefined object, the system transitions to **APPROACH** mode.
- After reaching the desired proximity, the robot enters **HOLD**, provides user feedback, and then returns to **FOLLOW**.

## Demo

▶️ **System demonstration video:**

[System Demo](https://github.com/user-attachments/assets/7d9a5e49-e259-4a5c-9ea6-8031682adc1d)

For a detailed system description, architecture overview, and design rationale, please refer to the project [Wiki](../../wiki).

A guide for the routine operation of the Go2 robot can be found at this [link](https://docs.google.com/document/d/14_qHkVyDAV9-awSuEr1Pa4G2HZZrMQGCJ11ud12n6Hk/edit?usp=sharing).




