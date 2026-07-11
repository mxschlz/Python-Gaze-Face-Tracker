import cv2
import sys
import os
import glob

def play_video(video_path):
    print(f"\nPlaying video: {video_path}")
    print("Controls:")
    print("  'q'       : Next video")
    print("  ESC       : Quit player entirely")
    print("  'p'/SPACE : Pause/Resume")
    print("  'd'       : Skip forward 5 seconds")
    print("  'a'       : Skip backward 5 seconds")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file.")
        return True # Continue to next video

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: 
        fps = 30
    delay = int(1000 / fps)

    paused = False
    quit_all = False

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
            cv2.imshow('Video Player', frame)

        key = cv2.waitKey(delay if not paused else 50) & 0xFF

        if key == 27:  # ESC
            quit_all = True
            break
        elif key == ord('q'):  # q
            break # go to next video
        elif key == ord('p') or key == ord(' '):  # p or SPACE
            paused = not paused
        elif key == ord('d'): # skip forward 5 seconds
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + 5 * fps)
            ret, frame = cap.read()
            if ret: cv2.imshow('Video Player', frame)
        elif key == ord('a'): # skip backward 5 seconds
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 5 * fps))
            ret, frame = cap.read()
            if ret: cv2.imshow('Video Player', frame)
            
    cap.release()
    return quit_all

def main():
    video_list = []
    
    if len(sys.argv) > 1:
        arg_path = sys.argv[1]
        if os.path.isdir(arg_path):
            video_list = glob.glob(os.path.join(arg_path, "*_Video_ML_predictions.mp4"))
        elif os.path.isfile(arg_path):
            video_list = [arg_path]
    else:
        # Search for videos in example_output and example_videos
        for search_dir in ["example_output", "example_videos"]:
            video_list = glob.glob(os.path.join(search_dir, "*_Video_ML_predictions.mp4"))
            if video_list:
                break
                
    if not video_list:
        print("No .mp4 files found.")
        return

    # Sort alphabetically
    video_list.sort()
    
    print(f"Found {len(video_list)} videos. Starting playback...")
    
    for i, video_path in enumerate(video_list):
        print(f"--- Video {i+1} of {len(video_list)} ---")
        if not os.path.exists(video_path):
            print(f"File not found: {video_path}")
            continue
            
        quit_all = play_video(video_path)
        if quit_all:
            print("Exiting player.")
            break
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
