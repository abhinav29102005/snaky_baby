# snaky_baby

🐍 Enhanced Gesture-Controlled Snake Game 🎮


This project is an enhanced version of the classic Snake game, controlled entirely by hand gestures using a webcam. It features advanced hand-tracking technology, a polished user interface, and multiplayer support, offering a modern and engaging gameplay experience.

✨ Features
Advanced Gesture Controls: Uses a calibrated gesture-tracking system to translate hand movements into snake direction.

Real-time Hand Calibration: A dedicated calibration phase adapts the controls to each player's unique hand size and movement style for optimal responsiveness.

Intuitive Gameplay: Move your open hand to steer the snake and close your fist to pause or resume the game.

Enhanced Multiplayer Mode: Supports up to four players with a dynamic score-tracking system and smooth transitions between turns.

Modern User Interface: A professional, animated, and scalable UI provides a clear and visually appealing experience.

Adaptive Difficulty: The game's speed increases as you score more points, keeping the challenge fresh.

Fullscreen Support: Easily toggle fullscreen mode with the press of a key for an immersive experience.

Robust Performance: Utilizes multithreading for smooth camera capture and a custom rendering pipeline for consistent frame rates.

🕹️ How to Play
Launch the Game: Run the main.py script. The game will automatically detect and initialize your webcam.

Select Players: From the main menu, press 1, 2, 3, or 4 to choose the number of players.

Enter Names (Multiplayer): If playing with multiple players, enter a name for each player and press ENTER.

Calibrate Your Hand: The calibration screen will prompt you to move your open hand slowly in front of the camera. This process measures your natural movement range for precise controls.

Start Playing: Follow the on-screen instructions. Move your open palm to guide the snake.

Pause the Game: Close your hand into a fist to pause the game. Open your hand to resume.

Watch Out! Avoid hitting the walls or the snake's body.

Check Results: After all players have completed their turns, the final results and a champion will be displayed.

🖥️ System Requirements
Hardware
A computer with a functioning webcam.

A stable internet connection for initial library installation.

Software
Python 3.6 or higher

Required libraries:

opencv-python

mediapipe

numpy

tkinter

📦 Installation
To get started, you'll need to install the necessary Python packages. Open your terminal or command prompt and run the following command:

Bash

pip install opencv-python mediapipe numpy tkinter
🚀 Running the Game
Navigate to the project directory in your terminal and execute the main script:

Bash

python main.py
⌨️ Controls
Key	Function
Q	Quit the game
F	Toggle fullscreen mode
ENTER	Confirm selection / Advance to next screen
BACKSPACE	Delete character (in name input)
ESC	Go back to the main menu
1 to 4	Select number of players (in menu)
Hand Gesture	Function
Open Palm	Move the snake
Closed Fist	Pause / Resume the game

Export to Sheets
Thank you for checking out the project! Enjoy the enhanced experience!
