from moviepy.video.io.VideoFileClip import VideoFileClip

# Load the video

def trim_video(path, start, end):
    
    video = VideoFileClip(path)

    # Cut the video between start_time and end_time
   
    if end > video.duration:
        trimmed_video = video.subclipped(start)
    else:
        trimmed_video = video.subclipped(start, end)
    

    # Save the trimmed video
    output_path = "trimmed_video.mp4"
    trimmed_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

    