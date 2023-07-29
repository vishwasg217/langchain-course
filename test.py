
import os

PATH = "./the-algorithm"

for dirpath, dirnames, filenames in os.walk(PATH):
    print(f"dirpath: {dirpath}\n dirnames: {dirnames}\n filenames: {filenames}")
    print("---------------------------------------------------")

    