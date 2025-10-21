import cv2
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def create_rotated_versions(input_path, output_dir):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writers for 3 rotated versions
    out_90  = cv2.VideoWriter(str(Path(output_dir) / "rotated_90.mp4"),  fourcc, fps, (h, w))
    out_180 = cv2.VideoWriter(str(Path(output_dir) / "rotated_180.mp4"), fourcc, fps, (w, h))
    out_270 = cv2.VideoWriter(str(Path(output_dir) / "rotated_270.mp4"), fourcc, fps, (h, w))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_90  = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame_180 = cv2.rotate(frame, cv2.ROTATE_180)
        frame_270 = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        out_90.write(frame_90)
        out_180.write(frame_180)
        out_270.write(frame_270)

    cap.release()
    out_90.release()
    out_180.release()
    out_270.release()
    cv2.destroyAllWindows()


def detect_faces_all_orientations(frame, detector, w, h):
    # orientations = [0, 90, 180, 270]
    orientations = [0]
    boxes = []

    for angle in orientations:
        rotated = frame
        if angle != 0:
            rotated = cv2.rotate(frame, {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE
            }[angle])

        rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        if not results.detections:
            continue

        for det in results.detections:
            box = det.location_data.relative_bounding_box
            x, y, bw, bh = box.xmin, box.ymin, box.width, box.height

            # map back to original orientation
            if angle == 0:
                boxes.append((int(x*w), int(y*h), int(bw*w), int(bh*h)))
            elif angle == 90:
                boxes.append((int(y*h), int(w*(1-x-bw)), int(bh*h), int(bw*w)))
            elif angle == 180:
                boxes.append((int(w*(1-x-bw)), int(h*(1-y-bh)), int(bw*w), int(bh*h)))
            elif angle == 270:
                boxes.append((int(h*(1-y-bh)), int(x*w), int(bh*h), int(bw*w)))
    return boxes


def anonymize_single_video(input_path, output_path, position=0):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4) as detector:
        with tqdm(total=total_frames, desc=f"{input_path.name}", position=position, leave=False) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                boxes = detect_faces_all_orientations(frame, detector, w, h)
                for (x, y, bw, bh) in boxes:
                    x, y = max(0, x), max(0, y)
                    bw = min(bw, w - x)
                    bh = min(bh, h - y)
                    frame[y:y + bh, x:x + bw] = 0

                out.write(frame)
                pbar.update(1)

    cap.release()
    out.release()


def anonymize_all_videos(input_dir, output_dir, num_workers=4):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = [f for f in input_dir.iterdir() if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]]
    if not video_files:
        raise FileNotFoundError(f"No video files found in {input_dir}")

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = []
        for pos, vid in enumerate(video_files):
            out_path = output_dir / f"{vid.stem}_anon{vid.suffix}"
            futures.append(pool.submit(anonymize_single_video, vid, out_path, pos))

        for f in as_completed(futures):
            f.result()  # propagate exceptions


if __name__ == "__main__":
    # create_rotated_versions("data/input_videos", "data/rotated_videos")
    anonymize_all_videos("data/input_videos", "data/anonymized_videos", num_workers=4)
