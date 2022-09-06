import json
from pathlib import Path
import subprocess
import random

data_path = "fmonk/hd-trailers.json"
output_dir = Path("apple-data")
with open(data_path, "rt") as f:
    data = json.load(f)

random.shuffle(data)

for d in data:
    title = d["title"]
    folder_name = title.lower()
    folder_name = ''.join(ch if ch.isalnum() else "_" for ch in folder_name)

    url = d["trailer_link"]

    folder_name = output_dir/Path(folder_name)
    folder_name.mkdir(exist_ok=True)

    movie_path = folder_name/url.split("/")[-1]

    if movie_path.exists():
        print(f"skipping {movie_path}")
        continue

    cmd = ["wget", "-O", str(movie_path), url]
    try:
        result = subprocess.run(cmd)
    except KeyboardInterrupt:
        print(f"removing {movie_path}")
        Path(movie_path).unlink()
        exit()
