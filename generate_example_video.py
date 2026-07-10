import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from ocapi.ocapi import Ocapi

def process_subject(subject_info):
    subject_id, session = subject_info
    subject_sess = f"{subject_id}_{session}"
    
    video_dir = "/home/maxschulz/IPSY1-Storage/Projects/ac/Experiments/running_studies/OCAPI/all_videos_combined"
    eeg_base_dir = "/home/maxschulz/IPSY1-Storage/Projects/ac/Experiments/running_studies/OCAPI/input"
    output_log_dir = "/home/maxschulz/PycharmProjects/ocapi/example_output"
    os.makedirs(output_log_dir, exist_ok=True)
    
    print(f"[{subject_sess}] Starting processing...")
    
    video_files = glob.glob(os.path.join(video_dir, f"{subject_sess}*Video.mp4"))
    if not video_files:
        return f"[{subject_sess}] FAILED: Video not found."
        
    video_path = video_files[0]
    
    eeg_path = os.path.join(eeg_base_dir, subject_id, session)
    eeg_header_files = glob.glob(os.path.join(eeg_path, f"{subject_sess}*.vhdr"))
    eeg_marker_files = glob.glob(os.path.join(eeg_path, f"{subject_sess}*.vmrk"))
    
    if not eeg_header_files or not eeg_marker_files:
        return f"[{subject_sess}] FAILED: EEG files not found."
        
    eeg_header_file = eeg_header_files[0]
    eeg_marker_file = eeg_marker_files[0]
    
    output_video_path = os.path.join(output_log_dir, f"{subject_sess}_Video_marked_fixed.mp4")
    
    try:
        tracker = Ocapi(
            subject_id=subject_id,
            session=session,
            config_file_path="config.yml",
            WEBCAM=None,
            VIDEO_INPUT=video_path,
            VIDEO_OUTPUT=output_video_path,
            TRACKING_DATA_LOG_FOLDER=output_log_dir
        )
        
        # Force full video decoding and on-screen rendering so it gets saved to the file
        tracker.skip_video_decoding = False
        tracker.SHOW_ON_SCREEN_DATA = False # Disable imshow so it runs headless
        tracker.PRINT_DATA = True
        
        # --- Speed optimization ---
        # Limit the calibration pass to the first 60 seconds instead of the default 1500s.
        # This prevents the slow DeepLabCut detector from having to process 25 minutes of video TWICE.
        tracker.CLUSTERING_CALIB_DURATION_SECONDS = 60
        
        # Sync with EEG
        tracker.sync_with_eeg_and_set_onsets(
            header_file=eeg_header_file,
            marker_file=eeg_marker_file,
            stimulus_description="Stimulus",
            events_of_interest=["S 21", "S 22", "S 23", "S 24"]
        )
        
        tracker.run()
        return f"[{subject_sess}] SUCCESS: Video saved."
    except Exception as e:
        return f"[{subject_sess}] ERROR: {str(e)}"

def main():
    subjects = [
        ("SCS048", "A"),
        ("SCS062", "A"),
        ("SMM049", "A"),
        ("SMM050", "A"),
        ("SMM054", "A")
    ]
    max_workers = min(len(subjects), os.cpu_count() or 4)
    print(f"Starting parallel processing for {len(subjects)} subjects using {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_subject, subj): subj for subj in subjects}
        for future in as_completed(futures):
            print(future.result())

if __name__ == '__main__':
    main()
