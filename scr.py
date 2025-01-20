import os

BASE_DIR = "./data"
FINAL_DIR = "./fixed"

if not os.path.exists(FINAL_DIR):
    os.mkdir(FINAL_DIR)

idx = 0

for dirpath, dirname, filenames in os.walk(BASE_DIR):

    for f in filenames:
        source = os.path.join(dirpath, f)
        destination = os.path.join(FINAL_DIR, f"{idx}.png")
        os.rename(source, destination)
        idx+=1

