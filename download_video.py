import yt_dlp

# Paste your YouTube video URL here
video_url = input("Paste YouTube video URL here: ")

# Download settings
ydl_opts = {
    'format': 'mp4',
    'outtmpl': 'driving_video.mp4',  # Output filename
}

print("Downloading video...")
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])

print("✓ Video downloaded as 'driving_video.mp4'")
print("Now you can run: python run_video.py")
