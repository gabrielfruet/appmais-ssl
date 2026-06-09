import lightly_studio as ls

# Create an empty dataset and add videos from a folder.
dataset = ls.VideoDataset.load_or_create(name="appmais")
dataset.add_videos_from_path(path="./data/videos_raw")
