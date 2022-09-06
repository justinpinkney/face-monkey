from pathlib import Path
import subprocess

ids = open("trailers.txt").readlines()
ids = [x.strip('\n') for x in ids]

outdir = Path("data")
for vid in ids:
    existing_vids = outdir.glob("*.mp4")
    if any([x.name.startswith(vid) for x in existing_vids]):
        continue

    print(f"downloading {vid}")
    cmd = [
        "youtube-dl", vid,
        "-o", 'data/%(id)s_%(title)s.%(ext)s',
        "--restrict-filenames",
    ]
    result = subprocess.run(cmd)

