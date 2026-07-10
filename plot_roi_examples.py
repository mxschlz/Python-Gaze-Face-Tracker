import os
import glob
import cv2 as cv
from ocapi.ocapi import Ocapi

def main():
    video_dir = "/home/maxschulz/IPSY1-Storage/Projects/ac/Experiments/running_studies/OCAPI/all_videos_combined"
    
    mp4_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    subjects_dict = {}
    for f in mp4_files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 2:
            subject_id = parts[0]
            session = parts[1]
            subject_sess = f"{subject_id}_{session}"
            if subject_sess not in subjects_dict:
                subjects_dict[subject_sess] = (subject_id, session, f)
                
    subjects = sorted(list(subjects_dict.values()), key=lambda x: (x[0], x[1]))[:5]
    
    for subject_id, session, video_path in subjects:
        subject_sess = f"{subject_id}_{session}"
        print(f"[{subject_sess}] Visualizing ROI...")
        
        try:
            tracker = Ocapi(
                subject_id=subject_id,
                session=session,
                config_file_path="config.yml",
                WEBCAM=None,
                VIDEO_INPUT=video_path,
                VIDEO_OUTPUT=None,
                TRACKING_DATA_LOG_FOLDER="example_output"
            )
            
            # Force visualization on
            tracker.VISUALIZE_ROI_SEARCH = True
            tracker.STIMULUS_ROI_METHOD = "dynamic"
            tracker.PRINT_DATA = True
            
            # We just want to run the dynamic ROI search to visualize it.
            # We don't need to run the full tracker.
            tracker._find_dynamic_roi()
            print(f"[{subject_sess}] ROI search finished.")
            
        except Exception as e:
            print(f"[{subject_sess}] ERROR: {str(e)}")

if __name__ == '__main__':
    main()
