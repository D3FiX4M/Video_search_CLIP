from datetime import timedelta
import cv2
import numpy as np
import os
import shutil


def format_timedelta(td):
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def get_saving_frames_durations(cap, saving_fps):
    s = []

    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s


def split_frame(video_file, save_path, saving_frames_per_second=1.5):

    cap = cv2.VideoCapture(video_file)

    fps = cap.get(cv2.CAP_PROP_FPS)

    saving_frames_per_second = min(fps, saving_frames_per_second)

    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)

    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            break

        frame_duration = count / fps
        try:

            closest_duration = saving_frames_durations[0]
        except IndexError:

            break

        if frame_duration >= closest_duration:

            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            cv2.imwrite(os.path.join(save_path, f"frame{frame_duration_formatted}.png"), frame)

            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass

        count += 1
