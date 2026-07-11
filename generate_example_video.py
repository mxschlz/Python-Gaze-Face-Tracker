import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from ocapi.ocapi import Ocapi

def process_subject(subject_sess, video_path, eeg_base_dir, output_log_dir):
    print(f"[{subject_sess}] Starting processing...")
    
    parts = subject_sess.split('_')
    subject_id = parts[0]
    session = parts[1]
    
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
        
        # We removed the manual 60s cap on CLUSTERING_CALIB_DURATION_SECONDS
        # to ensure it uses the full video for calibration (maximum performance).
        
        # Sync with EEG
        tracker.sync_with_eeg_and_set_onsets(
            header_file=eeg_header_file,
            marker_file=eeg_marker_file,
            stimulus_description="Stimulus",
            events_of_interest=["S 21", "S 22", "S 23", "S 24"]
        )
        
        tracker.run()
        
        # --- POST-PROCESSING: Burn ML Predictions into Video ---
        import pandas as pd
        import cv2 as cv
        
        # Locate the generated logs and summaries
        log_pattern = os.path.join(output_log_dir, f"{subject_sess}_eye_tracking_log*.csv")
        summary_pattern = os.path.join(output_log_dir, f"{subject_sess}_trial_summary*.csv")
        log_files = glob.glob(log_pattern)
        summary_files = glob.glob(summary_pattern)
        
        if log_files and summary_files:
            log_file = max(log_files, key=os.path.getctime)
            summary_file = max(summary_files, key=os.path.getctime)
            
            # Load ML predictions
            summary_df = pd.read_csv(summary_file)
            predictions = {}
            for _, row in summary_df.iterrows():
                trial_id = int(row['trial_id'])
                pred = int(row.get('looked_at_stimulus', 2))
                predictions[trial_id] = "LOOKED" if pred == 1 else "AWAY"
                
            # Load per-frame Trial IDs
            log_df = pd.read_csv(log_file)
            trial_ids_per_frame = log_df['Trial ID'].values
            
            # Post-process the video
            final_video_path = os.path.join(output_log_dir, f"{subject_sess}_Video_ML_predictions.mp4")
            cap = cv.VideoCapture(output_video_path)
            
            if cap.isOpened():
                width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv.CAP_PROP_FPS)
                
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                out = cv.VideoWriter(final_video_path, fourcc, fps, (width, height))
                
                frame_idx = 0
                font = cv.FONT_HERSHEY_SIMPLEX
                
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                        
                    if frame_idx < len(trial_ids_per_frame):
                        t_id = int(trial_ids_per_frame[frame_idx])
                        # Erase old text
                        cv.rectangle(frame, (width - 400, 0), (width, 80), (0, 0, 0), -1)
                        if t_id > 0 and t_id in predictions:
                            pred_text = f"Trial {t_id} ML Prediction: {predictions[t_id]}"
                            color = (0, 255, 0) if predictions[t_id] == "LOOKED" else (0, 0, 255)
                            text_size = cv.getTextSize(pred_text, font, 1.0, 2)[0]
                            text_x = width - text_size[0] - 10
                            cv.putText(frame, pred_text, (text_x, 40), font, 1.0, color, 2)
                    out.write(frame)
                    frame_idx += 1
                    
                cap.release()
                out.release()
                
                # Optionally delete the old video to save space
                try:
                    os.remove(output_video_path)
                except Exception:
                    pass

        return f"[{subject_sess}] SUCCESS: Video saved and labeled with ML predictions."
    except Exception as e:
        return f"[{subject_sess}] ERROR: {str(e)}"

def main():
    video_dir = "/home/maxschulz/IPSY1-Storage/Projects/ac/Experiments/running_studies/OCAPI/all_videos_combined"
    eeg_base_dir = "/home/maxschulz/IPSY1-Storage/Projects/ac/Experiments/running_studies/OCAPI/input"
    rater_1_dir = "/home/maxschulz/IPSY1-Storage/Projects/ac/Experiments/running_studies/OCAPI/output/human_coding/rater_1"
    output_log_dir = "/home/maxschulz/PycharmProjects/ocapi/example_output"
    os.makedirs(output_log_dir, exist_ok=True)

    video_extensions = ['.mkv', '.mp4', '.avi', '.mov']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
    video_files = sorted(video_files)
    
    # Filter files that have human coding (same as validation pipeline)
    tasks_args = []
    for f in video_files:
        base_name = os.path.splitext(os.path.basename(f))[0]
        try:
            parts = base_name.split('_')
            subject_sess = f"{parts[0]}_{parts[1]}"
            rater1_file = os.path.join(rater_1_dir, f"{subject_sess}_VideoCoding.xlsx")
            if not os.path.exists(rater1_file):
                glob_matches = glob.glob(os.path.join(rater_1_dir, f"{subject_sess}*"))
                if not glob_matches:
                    continue
            tasks_args.append((subject_sess, f, eeg_base_dir, output_log_dir))
        except Exception:
            continue
            
    # Process 4 videos concurrently (safe for RAM/CPU)
    max_workers = 25
    print(f"Starting parallel processing for {len(tasks_args)} subjects using {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_subject, *args): args for args in tasks_args}
        for future in as_completed(futures):
            print(future.result())

if __name__ == '__main__':
    main()
