# RPS-Neural-Link: Cyber-Physical Game Theory System
### AISE 3350A Group Project | Western University

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-MediaPipe-orange)

## 🤖 Project Overview
**RPS-Neural-Link** is an Augmented Reality (AR) decision-support system designed to play the game of *Rock-Paper-Scissors-Minus-One* (RPS-1). 

Unlike standard random bots, this system utilizes **Computer Vision** to perceive the real world and **Adaptive Machine Learning (Markov Chains)** to analyze the opponent's psychological patterns in real-time. It provides the user with a tactical "Heads Up Display" (HUD) suggesting the mathematically optimal move to defeat the opponent.

## ✨ Key Features
* **Real-Time Computer Vision:** Uses **MediaPipe** and **OpenCV** to track up to 4 hands simultaneously at 30+ FPS.
* **Adaptive AI Brain:** Implements a **Markov Chain Predictor** that learns the opponent's move patterns (e.g., "Opponent plays Paper after Rock 80% of the time").
* **Game Theory Engine:** Calculates the **Nash Equilibrium** to determine the optimal defensive strategy when no clear pattern is detected.
* **Cyberpunk UI:** A responsive, web-based HUD featuring CRT scanlines, probability visualizations, and holographic controls.
* **Multi-Mode Support:**
    * **Single Player:** Human vs. Adaptive AI Simulation.
    * **Tournament Mode:** Best of 3/5/9 match tracking with post-game statistical analysis.

## 🛠️ Tech Stack
* **Backend:** Python, Flask (Web Server)
* **Computer Vision:** Google MediaPipe (Hand Landmark Detection), OpenCV
* **Frontend:** HTML5, CSS3 (Animations), JavaScript (Async Polling)
* **Math/Logic:** NumPy (Matrix Operations)

## 🚀 Installation & Setup

### Prerequisites
* Python 3.8 or higher
* A webcam

### Step 1: Clone the Repository
```bash
git clone [https://github.com/TylerLafond/RPS-Project.git](https://github.com/TylerLafond/RPS-Project.git)
cd RPS-Project
