KNIGHT EXPERIENCE - NATURAL ANIMATION UPDATE

1) Keep your existing assets/music.mp3 in this project's assets folder.
   The supplied knight-exact.png is the exact latest image you selected.
2) Install the requirements in PowerShell:
   py -m pip install -r requirements.txt
3) Run:
   .\run.bat

If you already have run.bat, keep it. It should run:
   py knight_experience.py

Controls:
ESC = quit
SPACE = pause/resume music and animation
F11 = fullscreen

This version does not add fake orange particles or drawn cape lines.
It animates the red cape extracted from the artwork and gently moves the
lower foreground grass. The original composition remains the background.
