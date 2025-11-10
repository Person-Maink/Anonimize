import cv2
import os
import argparse


def extract_clips(video_path, output_folder, skip_secs=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(output_folder, exist_ok=True)

    print("Controls:")
    print("  space = pause/resume")
    print("  w = mark start")
    print("  s = mark end and save clip")
    print("  a = go backward 2s")
    print("  d = go forward 2s")
    print("  q = quit")

    start_frame = None
    clip_count = 0
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # re-show paused frame
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
            ret, frame = cap.read()

        cv2.imshow("Clip Extractor", frame)
        key = cv2.waitKey(int(1000 / fps)) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):  # pause/resume
            paused = not paused

        elif key == ord("a"):  # go backward
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_frame = max(0, current_frame - int(fps * skip_secs))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            print(f"<< back {skip_secs}s → {new_frame/fps:.2f}s")

        elif key == ord("d"):  # go forward
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_frame = min(total_frames - 1, current_frame + int(fps * skip_secs))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            print(f">> forward {skip_secs}s → {new_frame/fps:.2f}s")

        elif key == ord("w"):  # mark start
            start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f"Marked start at {start_frame / fps:.2f}s")

        elif key == ord("s") and start_frame is not None:  # mark end and save
            end_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if end_frame > start_frame:
                clip_count += 1
                out_path = os.path.join(output_folder, f"clip_{clip_count}.mp4")

                cap2 = cv2.VideoCapture(video_path)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

                for _ in range(start_frame, end_frame):
                    ret2, frame2 = cap2.read()
                    if not ret2:
                        break
                    out.write(frame2)

                out.release()
                cap2.release()
                print(f"Saved {out_path} ({(end_frame - start_frame)/fps:.2f}s)")
            else:
                print("Invalid end before start.")
            start_frame = None

    cap.release()
    cv2.destroyAllWindows()


def batch_extract(input_folder, output_folder, skip_secs=2):
    os.makedirs(output_folder, exist_ok=True)

    video_exts = (".mp4", ".avi", ".mov", ".mkv")
    videos = [f for f in os.listdir(input_folder) if f.lower().endswith(video_exts)]

    if not videos:
        print("No videos found in input folder.")
        return

    print(f"Found {len(videos)} videos in {input_folder}")
    print("Starting interactive extraction...")

    for i, fname in enumerate(videos, 1):
        in_path = os.path.join(input_folder, fname)
        out_subdir = os.path.join(output_folder, os.path.splitext(fname)[0])
        os.makedirs(out_subdir, exist_ok=True)

        print(f"\n[{i}/{len(videos)}] Processing: {fname}")
        print(f"Output folder: {out_subdir}")
        extract_clips(in_path, out_subdir, skip_secs=skip_secs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch interactive clip extraction for all videos in a folder.")
    parser.add_argument("--input_folder", type=str, default="/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/outputs/wilor/videos", help="Path to folder containing input videos")
    parser.add_argument("--output_folder", type=str, default="/home/mayank/Documents/Uni/TUD/Thesis Extra/anonimize/data/clips", help="Path to folder to store clips")
    parser.add_argument("--skip_secs", type=float, default=2, help="Seconds to jump when pressing 'a' or 'd'")
    args = parser.parse_args()

    batch_extract(args.input_folder, args.output_folder, skip_secs=args.skip_secs)
