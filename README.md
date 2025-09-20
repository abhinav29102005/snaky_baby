# 🐍 Snaky Baby - Enhanced Gesture-Controlled Snake Game 🎮

<div align="center">

![Snake Animation](https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake.svg)

<h3>🚀 Experience the classic Snake game like never before with cutting-edge hand gesture controls! 🚀</h3>

[![Python](https://img.shields.io/badge/Python-3.6+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8+-00D4FF?style=for-the-badge&logo=google&logoColor=white)](https://mediapipe.dev)
[![Tkinter](https://img.shields.io/badge/Tkinter-GUI-FF6B35?style=for-the-badge&logo=python&logoColor=white)](https://docs.python.org/3/library/tkinter.html)

![GitHub repo size](https://img.shields.io/github/repo-size/abhinav29102005/snaky_baby?style=flat-square&color=green)
![GitHub last commit](https://img.shields.io/github/last-commit/abhinav29102005/snaky_baby?style=flat-square&color=blue)
![GitHub stars](https://img.shields.io/github/stars/abhinav29102005/snaky_baby?style=flat-square&color=yellow)
![GitHub forks](https://img.shields.io/github/forks/abhinav29102005/snaky_baby?style=flat-square&color=orange)

</div>

---

## 🎯 **What Makes Snaky Baby Special?**

<div align="center">

```
     🎮 No Keyboard Needed    🖐️ Gesture Controls    🏆 Multiplayer Mode
          ↓                        ↓                       ↓
    ┌─────────────┐          ┌─────────────┐         ┌─────────────┐
    │    Just     │          │   Move Your │         │  Up to 4    │
    │   Use Your  │    ➜     │    Hand to  │   ➜     │   Players   │
    │   Webcam!   │          │  Control!   │         │   Compete!  │
    └─────────────┘          └─────────────┘         └─────────────┘
```

</div>

---

## ✨ **Features That Will Blow Your Mind**

<table>
<tr>
<td width="50%">

### 🎯 **Advanced Gesture Controls**
- 🖐️ **Open Palm**: Navigate your snake
- ✊ **Closed Fist**: Pause/Resume game
- 🔄 **Real-time calibration** for personalized controls
- 📐 **Adaptive hand size detection**

### 🎮 **Enhanced Multiplayer**
- 👥 **Up to 4 players** can compete
- 📊 **Dynamic score tracking**
- 🏆 **Championship results**
- 🔄 **Smooth turn transitions**

</td>
<td width="50%">

### 🎨 **Modern UI/UX**
- ✨ **Professional animated interface**
- 📱 **Scalable and responsive design**
- 🖥️ **Fullscreen mode support**
- 🎭 **Smooth visual transitions**

### ⚡ **Robust Performance**
- 🧵 **Multithreaded camera capture**
- 🎬 **Custom rendering pipeline**
- 📈 **Adaptive difficulty scaling**
- 🔧 **Optimized frame rates**

</td>
</tr>
</table>

---

## 🛠️ **Tech Stack & Architecture**

<div align="center">

### **Core Technologies**

| Technology | Version | Purpose | Badge |
|------------|---------|---------|--------|
| **Python** | 3.6+ | Core Language | ![Python](https://img.shields.io/badge/Python-FFD43B?style=flat&logo=python&logoColor=blue) |
| **OpenCV** | 4.0+ | Computer Vision | ![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=flat&logo=OpenCV&logoColor=white) |
| **MediaPipe** | 0.8+ | Hand Tracking | ![MediaPipe](https://img.shields.io/badge/MediaPipe-0093D7?style=flat&logo=google&logoColor=white) |
| **Tkinter** | Built-in | GUI Framework | ![Tkinter](https://img.shields.io/badge/Tkinter-306998?style=flat&logo=python&logoColor=white) |
| **NumPy** | Latest | Numerical Computing | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) |

### **Architecture Flow**

</div>

```mermaid
graph TD
    A[🎥 Webcam Input] --> B[📹 OpenCV Capture]
    B --> C[🖐️ MediaPipe Hand Detection]
    C --> D[📊 Gesture Analysis]
    D --> E[🎮 Game Logic Engine]
    E --> F[🎨 Tkinter UI Renderer]
    F --> G[🖥️ Display Output]
    
    H[👥 Multi-Player Manager] --> E
    I[⚙️ Calibration System] --> D
    J[🏆 Score Tracker] --> F
    
    style A fill:#ff6b6b
    style C fill:#4ecdc4
    style E fill:#45b7d1
    style F fill:#f9ca24
```

---

## 🚀 **Quick Start Guide**

<div align="center">

### **Step 1: Clone the Repository**

</div>

```bash
# 📥 Clone the amazing Snaky Baby repository
git clone https://github.com/abhinav29102005/snaky_baby.git

# 📂 Navigate to the project directory
cd snaky_baby
```

<div align="center">

### **Step 2: Install Dependencies**

</div>

```bash
# 🔧 Install all required packages with one command!
pip install opencv-python mediapipe numpy tkinter

# Or use requirements.txt if available
pip install -r requirements.txt
```

<div align="center">

### **Step 3: Launch the Game**

</div>

```bash
# 🎮 Start your gesture-controlled adventure!
python main.py
```

---

## 🎮 **How to Play Like a Pro**

<div align="center">

### **Game Flow Visualization**

</div>

```
🏠 Main Menu → 👥 Select Players → ✏️ Enter Names → 🤏 Calibration → 🎮 Game Start
     ↓              ↓                ↓              ↓              ↓
   Press 1-4    Enter player      Move hand      Open palm =    🐍 Control
   to choose      names for      slowly for       Navigate       your snake!
   players      multiplayer     calibration        game           
```

### **🖐️ Gesture Commands**

<table align="center">
<tr>
<th width="30%">Gesture</th>
<th width="35%">Action</th>
<th width="35%">Visual Cue</th>
</tr>
<tr>
<td align="center">🖐️ <strong>Open Palm</strong></td>
<td align="center">Move Snake Direction</td>
<td align="center">✅ Green indicator</td>
</tr>
<tr>
<td align="center">✊ <strong>Closed Fist</strong></td>
<td align="center">Pause/Resume Game</td>
<td align="center">⏸️ Pause icon</td>
</tr>
<tr>
<td align="center">👋 <strong>No Hand</strong></td>
<td align="center">Continue Current Direction</td>
<td align="center">➡️ Arrow indicator</td>
</tr>
</table>

### **⌨️ Keyboard Shortcuts**

| Key | Function | Description |
|-----|----------|-------------|
| `Q` | Quit Game | 🚪 Exit the application |
| `F` | Fullscreen | 🖥️ Toggle fullscreen mode |
| `ENTER` | Confirm | ✅ Advance to next screen |
| `BACKSPACE` | Delete | ❌ Remove character in name input |
| `ESC` | Back | ⬅️ Return to main menu |
| `1-4` | Players | 👥 Select number of players |

---

## 📋 **System Requirements**

<div align="center">

### **Minimum Requirements**

</div>

| Component | Requirement | Recommended |
|-----------|-------------|-------------|
| **🖥️ OS** | Windows 7+, macOS 10.12+, Ubuntu 16.04+ | Windows 10+, macOS 12+, Ubuntu 20.04+ |
| **🐍 Python** | 3.6+ | 3.8+ |
| **📹 Webcam** | Any USB/Built-in webcam | HD webcam (720p+) |
| **💾 RAM** | 4GB | 8GB+ |
| **💽 Storage** | 500MB free space | 1GB free space |
| **🌐 Internet** | Required for installation | Stable connection |

---

## 🏆 **Game Modes & Features**

<div align="center">

### **🎯 Single Player Mode**
Perfect your gesture control skills and achieve high scores!

### **👥 Multiplayer Mode (2-4 Players)**
Compete with friends in turn-based gameplay with comprehensive scoring!

### **🔧 Calibration System**
Personalized hand tracking for optimal control precision!

</div>

---

## 🤝 **Contributing**

We welcome contributions from the community! Here's how you can help make Snaky Baby even better:

<div align="center">

### **Ways to Contribute**

</div>

| Contribution Type | How to Help | Badge |
|-------------------|-------------|--------|
| 🐛 **Bug Reports** | Found a bug? Open an issue! | ![Issues](https://img.shields.io/github/issues/abhinav29102005/snaky_baby?style=flat-square) |
| ✨ **Feature Requests** | Have an idea? We'd love to hear it! | ![Ideas](https://img.shields.io/badge/Ideas-Welcome-brightgreen?style=flat-square) |
| 💻 **Code Contributions** | Submit a pull request! | ![PRs](https://img.shields.io/badge/PRs-Welcome-blue?style=flat-square) |
| 📚 **Documentation** | Help improve our docs! | ![Docs](https://img.shields.io/badge/Docs-Help%20Needed-orange?style=flat-square) |

### **Development Setup**

```bash
# 🔧 Set up development environment
git clone https://github.com/abhinav29102005/snaky_baby.git
cd snaky_baby

# 🐍 Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 📦 Install dependencies
pip install -r requirements.txt
```

---

## 🐛 **Troubleshooting**

<details>
<summary><strong>🔍 Common Issues & Solutions</strong></summary>

### **Camera Not Detected**
- ✅ Ensure webcam is connected and not used by other apps
- ✅ Try different USB ports
- ✅ Check camera permissions in system settings

### **Poor Hand Detection**
- ✅ Ensure good lighting conditions
- ✅ Keep hand clearly visible in camera frame
- ✅ Complete the calibration process properly

### **Performance Issues**
- ✅ Close other camera-using applications
- ✅ Lower camera resolution if possible
- ✅ Check system resources (CPU/RAM usage)

### **Installation Problems**
```bash
# If pip install fails, try:
pip install --upgrade pip
pip install --user opencv-python mediapipe numpy

# For macOS users:
brew install python-tk
```

</details>

---

## 🎖️ **Achievements & Stats**

<div align="center">

![GitHub Language Count](https://img.shields.io/github/languages/count/abhinav29102005/snaky_baby?style=for-the-badge&color=brightgreen)
![Top Language](https://img.shields.io/github/languages/top/abhinav29102005/snaky_baby?style=for-the-badge&color=blue)
![Code Size](https://img.shields.io/github/languages/code-size/abhinav29102005/snaky_baby?style=for-the-badge&color=orange)

### **🏆 Project Milestones**

- ✅ **Gesture Recognition** - Advanced hand tracking implementation
- ✅ **Multiplayer Support** - Up to 4 players competitive mode
- ✅ **Real-time Calibration** - Personalized control system
- ✅ **Modern UI** - Professional animated interface
- 🎯 **Future**: AI opponent mode, online multiplayer, custom themes

</div>

---

## 📞 **Get in Touch**

<div align="center">

### **Connect with the Developer**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-bigboyaks-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/bigboyaks)
[![Email](https://img.shields.io/badge/Email-asingh3__be24@thapar.edu-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:asingh3_be24@thapar.edu)
[![GitHub](https://img.shields.io/badge/GitHub-abhinav29102005-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abhinav29102005)

### **💝 Show Your Support**

If you enjoyed this project, please consider:

[![Star this repo](https://img.shields.io/badge/⭐-Star%20this%20repo-yellow?style=for-the-badge)](https://github.com/abhinav29102005/snaky_baby)
[![Fork](https://img.shields.io/badge/🍴-Fork%20this%20repo-orange?style=for-the-badge)](https://github.com/abhinav29102005/snaky_baby/fork)
[![Follow](https://img.shields.io/badge/👤-Follow%20@abhinav29102005-blue?style=for-the-badge)](https://github.com/abhinav29102005)

</div>

---

## 📜 **License**

<div align="center">

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

![License](https://img.shields.io/github/license/abhinav29102005/snaky_baby?style=for-the-badge&color=green)

</div>

---

<div align="center">

### **🎉 Thank you for checking out Snaky Baby! 🎉**

**Made with ❤️ and lots of ☕ by [Abhinav Singh](https://github.com/abhinav29102005)**

---

*🐍 Ready to play? Let's get those gestures moving! 🚀*

[![Back to Top](https://img.shields.io/badge/⬆️-Back%20to%20Top-blue?style=for-the-badge)](#-snaky-baby---enhanced-gesture-controlled-snake-game-)

</div>
