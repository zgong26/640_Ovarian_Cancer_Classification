import pyautogui
import time

# Set the duration (20 minutes)
duration = 50 * 60  # 20 minutes * 60 seconds

# Wait for 20 minutes
print(f"Waiting for {duration / 60} minutes")
time.sleep(duration)

# Perform a left click (at the current mouse position)
pyautogui.click()

print("Mouse click performed")
