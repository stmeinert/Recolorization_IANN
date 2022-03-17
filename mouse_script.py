from pynput.mouse import Button, Controller
import time

mouse = Controller()

while True:
    time.sleep(30)
    mouse.click(Button.left, 1)