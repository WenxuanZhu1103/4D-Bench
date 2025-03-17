import cv2

def get_video_uniform_frames(video_path, frame_num=6):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < frame_num:
        print(f"the frame number of the video is less than {frame_num}, return empty list")
        return []
    
    interval = total_frames // frame_num
    frames = []
    for i in range(frame_num):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"can not read the {i}th frame")
    
    cap.release()
    return frames


