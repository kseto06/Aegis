from pytube import YouTube
import webbrowser

def launch_video(video_url: str = 'https://youtu.be/FMud2gxl3XE'):
    # Create a YouTube object and play the video on the webbrowser
    try:
        yt = YouTube(video_url)
    except Exception as e:
        print("Exception: ", e)

    webbrowser.open(video_url)

if __name__ == '__main__':
    launch_video()