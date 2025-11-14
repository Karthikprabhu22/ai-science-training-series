import os
import json
from typing import Dict
from langchain_core.tools import tool
import yt_dlp


@tool
def download_youtube_video(url: str) -> Dict:
    """
    Downloads a YouTube video and metadata using yt_dlp from the given url.

    Parameters
    ----------
    url : str
        The url to the video to be downloaded.

    Returns
    -------
    dict
        A single-line JSON string LLMs can parse, e.g.
        {"ok": True,"video_folder": folder,"metadata": metadata,"video_path": video_path,"title": metadata["title"],"duration": metadata["duration"]}

    Raises
    ------
    Returns a dict describing the artifact.
    """

    folder = "downloads"
    os.makedirs(folder, exist_ok=True)

    video_path = os.path.join(folder, "video.mp4")

    # Download using yt_dlp
    ydl_opts = {
        "format": "mp4",
        "outtmpl": video_path,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    metadata = {
        "title": info.get("title", ""),
        "description": info.get("description", ""),
        "duration": info.get("duration", 0),
        "heatmap": info.get("heatmap",[]),
        "chapters": info.get("chapters", []),
        "video_path": video_path,
    }

    metadata_path = os.path.join(folder, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "ok": True,
        "video_folder": folder,
        "metadata": metadata,
        "video_path": video_path,
        "title": metadata["title"],
        "duration": metadata["duration"],
    }

