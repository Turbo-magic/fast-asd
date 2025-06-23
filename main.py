import sieve
from utils import get_video_dimensions, get_video_length, create_video_segments
from custom_types import VideoSegment, Frame, Box
import threading
import queue

SPEAKER_DETECTION_MODEL = "sieve/talknet-asd"
SPEAKER_DETECTION_IN_MEMORY_THRESHOLD = 3000
OBJECT_DETECTION_MODEL = "sieve/yolov8"

def push_video_segments_to_object_detection(video_segment, file, frame_interval=600, models="yolov8l, yolov8l-face", processing_fps=2):
    object_detector = sieve.function.get(OBJECT_DETECTION_MODEL)
    # push the video segments to object detection for every frame_interval frames
    total_num_frames = video_segment.end_frame if video_segment.end_frame else int(video_segment.end * video_segment.fps())
    start = video_segment.start_frame if video_segment.start_frame else int(video_segment.start * video_segment.fps())
    sampled_box_outputs = []
    for i in range(start, total_num_frames, frame_interval):
        start_frame = i
        end_frame = min(i + frame_interval - 1, total_num_frames - 1)
        
        sample_box_output = object_detector.push(
            file,
            confidence_threshold=0.5,
            start_frame=start_frame,
            end_frame=end_frame,
            models=models,
            fps=processing_fps,
            max_num_boxes=3,
        )

        sampled_box_outputs.append({
            "future": sample_box_output,
            "start": start_frame,
            "end": end_frame,
        })

    return sampled_box_outputs

def get_active_speakers(speaker_frames, alpha=0.5, score_threshold=0):
    # smooth the scores of the boxes from the speaker detection model
    active_speakers = {}
    smoothed_scores = {}

    for frame in speaker_frames:
        frame_number = frame['frame_number']
        active_speakers[frame_number] = []

        for box in frame['boxes']:
            box_id = box['track_id']
            raw_score = box['raw_score']

            if box_id not in smoothed_scores:
                smoothed_scores[box_id] = raw_score
            else:
                smoothed_scores[box_id] = alpha * raw_score + (1 - alpha) * smoothed_scores[box_id]

            if smoothed_scores[box_id] > score_threshold:
                active_speakers[frame_number].append(Box(class_id=-1, confidence=1.0, x1=box['x1'], y1=box['y1'], x2=box['x2'], y2=box['y2'], id=box_id, metadata={'raw_score': smoothed_scores[box_id]}))

    return active_speakers

metadata = sieve.Metadata(
    title="Detect Active Speakers",
    description="State-of-the-art active speaker detection based on new, efficent face and speaker detection models.",
    tags=["Video", "Showcase"],
    code_url="https://github.com/sieve-community/fast-asd",
    image=sieve.Image(url="https://storage.googleapis.com/sieve-public-data/asd/speaker-icon.webp"),
    readme=open("README.md", "r").read(),
)

@sieve.function(
    name="active_speaker_detection-staging",
    python_version="3.9",
    metadata=metadata,
    python_packages=[
        "numpy==1.23.5",
        "filterpy==1.4.5",
        "opencv-python==4.7.0.72",
        "scenedetect[opencv]",
    ],
    system_packages=[
        "ffmpeg",
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ],
    run_commands=[
        "pip install lap==0.4.0",
        "pip install sortedcontainers",
        "pip install supervision",
        "pip install 'vidgear[core]'",
        "pip install 'imageio[ffmpeg]'"
    ],
)
def process(
    file: sieve.File,
    speed_boost: bool = False,
    max_num_faces: int = 5,
    return_scene_cuts_only: bool = False,
    return_scene_data: bool = False,
    start_time: float = 0,
    end_time: float = -1,
    processing_fps: float = 2,
    face_size_threshold: float = 0.4,
):
    '''
    :param file: The video file to process
    :param speed_boost: Whether to use the faster but less accurate object detection model when processing the video.
    :param max_num_faces: The maximum number of faces to return per frame. If there are more than this number of faces, only the largest x faces will be returned.
    :param return_scene_cuts_only: Whether to only return the frame data at the start of each scene cut or to return the frame data for every frame in the video.
    :param return_scene_data: Whether to return the scene data along with the frame data. If True, the scene data will be returned in the "related_scene" field of the output.
    :param start_time: The seconds into the video to start processing from. Defaults to 0.
    :param end_time: The seconds into the video to stop processing at. Defaults to -1, which means the end of the video.
    :param processing_fps: The framerate to run object detection at for speaker detection. Defaults to 2.
    :param face_size_threshold: A threshold that determines the minimum size of a face to run speaker detection on. Defaults to 0.5. Lower values allow for smaller faces to be used.
    '''
    # handle webm files by converting them to mp4
    if file.path.endswith(".webm"):
        print("Converting webm file to mp4...")
        import os
        import subprocess
        new_path = file.path.replace(".webm", ".mp4")
        subprocess.run(["ffmpeg", "-y", "-i", file.path, "-c", "copy", new_path])
        file = sieve.File(path=new_path)

    width, height = get_video_dimensions(file.path)
    original_video_width = width
    original_video_height = height
    original_video_length = get_video_length(file.path)

    models = "yolov8l-face"
    if end_time == -1:
        end_time = original_video_length

    if start_time < 0 or start_time > original_video_length:
        raise ValueError(f"start_time must be between 0 and {original_video_length}")
    if end_time < 0 or end_time > original_video_length:
        raise ValueError(f"end_time must be between 0 and {original_video_length}")
    if start_time >= end_time:
        raise ValueError(f"start_time must be less than end_time")
    
    original_video = VideoSegment(
        path=file.path,
        start=start_time,
        end=end_time,
    )

    original_video_fps = original_video.fps()

    scene_detection_result = queue.Queue()
    object_detection_result = queue.Queue()

    # Define a wrapper function to call scene_detection and put the result in the Queue
    def scene_detection_wrapper(file, result_queue, **kwargs):
        print("Cutting video into scene segments...")
        from scene_detection import scene_detection
        result = list(scene_detection(file, **kwargs))
        result_queue.put(result)
        print("Done cutting video into scene segments")

    # Define a wrapper function to call push_video_segments_to_object_detection and put the result in the Queue
    def object_detection_wrapper(original_video, file, result_queue, frame_interval):
        print("Pushing video to object detection...")
        result = push_video_segments_to_object_detection(original_video, file, frame_interval=frame_interval, models=models, processing_fps=processing_fps)
        result_queue.put(result)
        print("Done pushing video to object detection")
    
    scene_detection_thread = threading.Thread(target=scene_detection_wrapper, args=(file, scene_detection_result), kwargs={'threshold': 15.0})
    scene_detection_thread.start()

    scene_detection_thread.join()
    scene_future = scene_detection_result.get()

    segments = create_video_segments(file, scene_future, start_time=start_time, end_time=end_time, fps=original_video_fps, original_video_length=original_video_length)
    total_num_frames = original_video.end_frame if original_video.end_frame else int(original_video.end * original_video_fps)

    # Calculate all detection intervals first, before creating any futures
    all_detection_intervals = []
    for segment_index, segment in enumerate(segments):
        start = segment.start
        end = segment.end
        scene_duration = end - start
        
        segment_intervals = []
        
        if scene_duration <= 5:
            # For short scenes (<=5s), only detect at start (0-1s)
            interval_start = start
            interval_end = min(start + 1, end)
            segment_intervals.append({
                'start_time': interval_start,
                'end_time': interval_end,
                'start_frame': int(interval_start * original_video_fps),
                'end_frame': int(interval_end * original_video_fps),
                'segment_index': segment_index
            })
        else:
            # For longer scenes, detect at start, middle, and end (each 1 second)
            # Start: 0-1s
            start_interval_start = start
            start_interval_end = min(start + 1, end)
            segment_intervals.append({
                'start_time': start_interval_start,
                'end_time': start_interval_end,
                'start_frame': int(start_interval_start * original_video_fps),
                'end_frame': int(start_interval_end * original_video_fps),
                'segment_index': segment_index
            })
            
            # Middle: around the middle 1 second
            # middle_point = start + scene_duration / 2
            # middle_start = max(start, middle_point - 0.5)
            # middle_end = min(end, middle_point + 0.5)
            # segment_intervals.append({
            #     'start_time': middle_start,
            #     'end_time': middle_end,
            #     'start_frame': int(middle_start * original_video_fps),
            #     'end_frame': int(middle_end * original_video_fps),
            #     'segment_index': segment_index
            # })
            
            # End: last 1 second
            # end_start = max(start, end - 1)
            # end_interval_end = end
            # segment_intervals.append({
            #     'start_time': end_start,
            #     'end_time': end_interval_end,
            #     'start_frame': int(end_start * original_video_fps),
            #     'end_frame': int(end_interval_end * original_video_fps),
            #     'segment_index': segment_index
            # })
        
        all_detection_intervals.extend(segment_intervals)

    # Create object detection futures only for our specific intervals
    object_detection_futures = []
    for interval in all_detection_intervals:
        object_detector = sieve.function.get(OBJECT_DETECTION_MODEL)
        future = object_detector.push(
            file,
            confidence_threshold=0.5,
            start_frame=interval['start_frame'],
            end_frame=interval['end_frame'],
            models=models,
            fps=processing_fps,
            max_num_boxes=3,
        )
        object_detection_futures.append({
            "future": future,
            "start": interval['start_frame'],
            "end": interval['end_frame'],
            "start_time": interval['start_time'],
            "end_time": interval['end_time'],
            "segment_index": interval['segment_index']
        })

    # Create speaker detection futures in parallel
    speaker_detection_futures = []
    for interval in all_detection_intervals:
        speaker_detection_futures.append({
            "future": None,
            "start": interval['start_frame'],
            "end": interval['end_frame'],
            "start_time": interval['start_time'],
            "end_time": interval['end_time'],
            "segment_index": interval['segment_index']
        })

    def seconds_to_timecode(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        new_seconds = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{new_seconds:02d}.{milliseconds:03d}"

    def convert_face_detection_outputs_to_string(face_detection_outputs):
        # convert the face detection outputs to a string that can be used as input to the speaker detection model
        out_str = ""
        frame_size = original_video_width * original_video_height
        interpolated_face_detection_outputs = []
        start = face_detection_outputs[0]["frame_number"]
        end = face_detection_outputs[-1]["frame_number"]
        for i in range(start, end + 1):
            frame_segment = None
            for segment in segments:
                segment_start_frame = segment.start_frame if segment.start_frame else int(segment.start * original_video_fps)
                segment_end_frame = segment.end_frame if segment.end_frame else int(segment.end * original_video_fps)
                if i >= segment_start_frame and i <= segment_end_frame:
                    frame_segment = segment
                    break
            if frame_segment is None:
                outputs_to_choose_from = face_detection_outputs
            else:
                outputs_to_choose_from = [output for output in face_detection_outputs if output["frame_number"] >= segment_start_frame and output["frame_number"] <= segment_end_frame]
            if len(outputs_to_choose_from) == 0:
                continue
            closest_frame = min(outputs_to_choose_from, key=lambda x: abs(x["frame_number"] - i))
            # copy the boxes to avoid modifying the original and set the frame number to the current frame
            new_frame = {
                "frame_number": i,
                "boxes": [box.copy() for box in closest_frame["boxes"]],
            }
            interpolated_face_detection_outputs.append(new_frame)
        for frame in interpolated_face_detection_outputs:
            out_str += f"{frame['frame_number']},"
            new_boxes = []
            # onyl keep the face boxes that are greater than 1/50 of the frame size
            for box in frame['boxes']:
                box_area = (box['x2'] - box['x1']) * (box['y2'] - box['y1'])
                if box_area > (face_size_threshold / 100) * frame_size and box['confidence'] > 0.5:
                    new_boxes.append(box)
            for box in new_boxes:
                if box['class_name'] != "face":
                    continue
                out_str += f"{box['x1']:.2f},{box['y1']:.2f},{box['x2']:.2f},{box['y2']:.2f},{box['confidence']:.2f},"
            # remove the last comma
            out_str = out_str[:-1]
            out_str += "\n"
        return out_str
    
    def get_relevant_face_detection_future(i):
        for j, future in enumerate(object_detection_futures):
            if "result" not in future and future["future"].done():
                try:
                    res = future["future"].result()
                    object_detection_futures[j]["result"] = res
                except:
                    print(f"WARNING: Found failed object detection (frame {future['start']}-{future['end']}), retrying...")
                    object_detection_futures[j]["future"] = sieve.function.get(OBJECT_DETECTION_MODEL).push(
                        file,
                        confidence_threshold=0.7,
                        start_frame=future["start"],
                        end_frame=future["end"],
                        models=models,
                        fps=processing_fps,
                        max_num_boxes=30,
                    )
                    continue
        if "result" in object_detection_futures[i]:
            return object_detection_futures[i]["result"]
        else:
            for _ in range(10):  # retry up to 10 times
                try:
                    res = object_detection_futures[i]["future"].result()
                    object_detection_futures[i]["result"] = res
                    return res
                except:
                    print(f"WARNING: Object detection failed, retrying... Attempt {_+1}/10")
                    if "result" in object_detection_futures[i]:
                        del object_detection_futures[i]["result"]
                    object_detection_futures[i]["future"] = sieve.function.get(OBJECT_DETECTION_MODEL).push(
                        file,
                        confidence_threshold=0.7,
                        start_frame=object_detection_futures[i]["start"], # start 10% into the segment to avoid scene boundaries
                        end_frame=object_detection_futures[i]["end"],
                        models=models,
                        # face_detection=False,
                        # speed_boost=speed_boost,
                        fps=processing_fps,
                        # interpolate_frames=True,
                        max_num_boxes=30,
                    )
            raise Exception("Object detection failed 10 times, please try again later.")
    
    def get_speaker_detection_payload(start, end, fps=30):
        # find all indices that contain the start and end
        for i, future in enumerate(speaker_detection_futures):
            if (start >= future["start"] and start <= future["end"]) or (end >= future["start"] and end <= future["end"]) or (start <= future["start"] and end >= future["end"]):
                for _ in range(10):  # retry up to 10 times
                    try:
                        if "result" not in speaker_detection_futures[i]:
                            while not speaker_detection_futures[i]["future"].done():
                                import time
                                time.sleep(0.1)
                            speaker_detection_futures[i]["result"] = list(speaker_detection_futures[i]["future"].result())
                        break  # if successful, break the retry loop
                    except Exception as e:
                        print(f"WARNING: Speaker detection failed, retrying... Attempt {_+1}/10")
                        if "result" in speaker_detection_futures[i]:
                            del speaker_detection_futures[i]["result"]
                        # recreate the future
                        res = list(get_relevant_face_detection_future(i))
                        face_detection_outputs = convert_face_detection_outputs_to_string(res)
                        speaker_detection_futures[i]["future"] = sieve.function.get(SPEAKER_DETECTION_MODEL).push(
                            file,
                            start_time=future["start"] / fps,
                            end_time=future["end"] / fps,
                            return_visualization=False,
                            face_boxes=face_detection_outputs,
                            in_memory_threshold=SPEAKER_DETECTION_IN_MEMORY_THRESHOLD
                        )

                if "result" not in speaker_detection_futures[i]:
                    raise Exception("Speaker detection failed 10 times, please try again later.")
                data = speaker_detection_futures[i]["result"]
                for frame in data:
                    if frame["frame_number"] >= start and frame["frame_number"] <= end:
                        yield frame

    def refresh_futures():
        for i, future in enumerate(object_detection_futures):
            if "result" in future and speaker_detection_futures[i]["future"] is None:
                res = list(get_relevant_face_detection_future(i))
                face_detection_outputs = convert_face_detection_outputs_to_string(res)
                speaker_detection_futures[i]["future"] = sieve.function.get(SPEAKER_DETECTION_MODEL).push(
                    file,
                    start_time=future["start"] / fps,
                    end_time=future["end"] / fps,
                    return_visualization=False,
                    face_boxes=face_detection_outputs,
                    in_memory_threshold=SPEAKER_DETECTION_IN_MEMORY_THRESHOLD
                )
            # print("bro",len(speaker_detection_futures), len(object_detection_futures))
            if speaker_detection_futures[i]["future"] and "result" not in speaker_detection_futures[i] and speaker_detection_futures[i]["future"].done():
                try:
                    res = speaker_detection_futures[i]["future"].result()
                    speaker_detection_futures[i]["result"] = res
                except:
                    print(f"WARNING: Found failed speaker detection (frame {future['start']}-{future['end']}), retrying...")
                    res = list(get_relevant_face_detection_future(i))
                    face_detection_outputs = convert_face_detection_outputs_to_string(res)
                    speaker_detection_futures[i]["future"] = sieve.function.get(SPEAKER_DETECTION_MODEL).push(
                        file,
                        start_time=future["start"] / fps, # start 10% into the segment to avoid scene boundaries
                        end_time=future["end"] / fps,
                        return_visualization=False,
                        face_boxes=face_detection_outputs,
                        in_memory_threshold=SPEAKER_DETECTION_IN_MEMORY_THRESHOLD
                    )
                    continue
        
        for i, future in enumerate(object_detection_futures):
            if "result" not in future and future["future"].done():
                try:
                    res = future["future"].result()
                    object_detection_futures[i]["result"] = res
                except:
                    print(f"WARNING: Found failed object detection (frame {future['start']}-{future['end']}), retrying...")
                    object_detection_futures[i]["future"] = sieve.function.get(OBJECT_DETECTION_MODEL).push(
                        file,
                        confidence_threshold=0.7,
                        start_frame=future["start"], # start 10% into the segment to avoid scene boundaries
                        end_frame=future["end"],
                        models="yolov8l, yolov8l-face",
                        fps=processing_fps,
                        max_num_boxes=30,
                    )
                    continue
        
        # loop over futures to see if any have errored, if so, recreate the future
        for i, future in enumerate(speaker_detection_futures):
            if "result" not in future and future["future"] and future["future"].done():
                try:
                    res = future["future"].result()
                    speaker_detection_futures[i]["result"] = res
                except:
                    print(f"WARNING: Found failed speaker detection (frame {future['start']}-{future['end']}), retrying...")
                    res = list(get_relevant_face_detection_future(i))
                    face_detection_outputs = convert_face_detection_outputs_to_string(res)
                    speaker_detection_futures[i]["future"] = sieve.function.get(SPEAKER_DETECTION_MODEL).push(
                        file,
                        start_time=future["start"] / fps,
                        end_time=future["end"] / fps,
                        return_visualization=False,
                        face_boxes=face_detection_outputs,
                        in_memory_threshold=SPEAKER_DETECTION_IN_MEMORY_THRESHOLD
                    )
                    continue

    print("------------------")
    print("Video Informaton")
    print("Video Length: {:.2f}s".format(original_video_length))
    print("Video FPS: {:.2f}".format(original_video_fps))
    start_frame = original_video.start_frame if original_video.start_frame else int(original_video.start * original_video_fps)
    end_frame = original_video.end_frame if original_video.end_frame else int(original_video.end * original_video_fps)
    print("Start Time: ", start_time, f"(Frame {start_frame})")
    print("End Time: ", end_time, f"(Frame {end_frame})")
    print("Start Frame: ", start_frame)
    print("End Frame: ", end_frame)
    print("Number of Scenes: ", len(segments))
    print(f"Number of Detection Intervals: {len(all_detection_intervals)}")
    print("------------------")
    print("Processing video...")

    # Prepare scene detections output
    shots = []
    for segment_index, segment in enumerate(segments):
        shots.append({
            "startTimestamp": round(segment.start * 1000),
            "endTimestamp": round(segment.end * 1000),
            "startFrame": segment.start_frame if segment.start_frame else int(segment.start * original_video_fps),
            "endFrame": (segment.end_frame if segment.end_frame else int(segment.end * original_video_fps)) - 1,
            "shotSegment": {
                "Confidence": 99.91699981689453,  # This is a placeholder value
                "Index": segment_index
            },
            "type": "SHOT"
        })

    # Process each detection interval
    frame_count = 0
    faces = []
    
    # First, create all speaker detection futures in parallel
    for interval_idx, detection_interval in enumerate(all_detection_intervals):
        # Wait for object detection to complete for this interval
        while not object_detection_futures[interval_idx]["future"].done():
            import time
            time.sleep(0.1)
        
        try:
            face_detection_result = list(object_detection_futures[interval_idx]["future"].result())
        except Exception as e:
            print(f"WARNING: Object detection failed for interval {interval_idx}, skipping...")
            continue
        
        # Convert face detection to string format for speaker detection
        if not face_detection_result:
            print(f"No faces detected in interval {interval_idx}, skipping...")
            continue
            
        face_detection_outputs = convert_face_detection_outputs_to_string(face_detection_result)
        
        # Create speaker detection future for this interval
        speaker_detection_futures[interval_idx]["future"] = sieve.function.get(SPEAKER_DETECTION_MODEL).push(
            file,
            start_time=detection_interval['start_time'],
            end_time=detection_interval['end_time'],
            return_visualization=False,
            face_boxes=face_detection_outputs,
            in_memory_threshold=SPEAKER_DETECTION_IN_MEMORY_THRESHOLD
        )
    
    # Then process the results
    for interval_idx, detection_interval in enumerate(all_detection_intervals):
        segment_index = detection_interval['segment_index']
        segment = segments[segment_index]
        
        print(f"Processing interval {interval_idx + 1}/{len(all_detection_intervals)} for scene {segment_index} [{detection_interval['start_time']:.2f}s - {detection_interval['end_time']:.2f}s]")
        
        # Wait for speaker detection to complete
        while not speaker_detection_futures[interval_idx]["future"].done():
            import time
            time.sleep(0.1)
        
        try:
            speaker_detection_result = list(speaker_detection_futures[interval_idx]["future"].result())
        except Exception as e:
            print(f"WARNING: Speaker detection failed for interval {interval_idx}, skipping...")
            continue
        
        # Process the results for this interval
        for frame in speaker_detection_result:
            frame_number = frame["frame_number"]
            boxes = []
            
            for box in frame["boxes"]:
                # Keep original integer values
                x1 = max(0, box["x1"])
                y1 = max(0, box["y1"])
                x2 = min(original_video_width, box["x2"])
                y2 = min(original_video_height, box["y2"])
                
                # Calculate percentages
                x1_pct = x1 / original_video_width
                y1_pct = y1 / original_video_height
                x2_pct = x2 / original_video_width
                y2_pct = y2 / original_video_height

                boxes.append({
                    "x1": x1_pct,
                    "y1": y1_pct,
                    "x2": x2_pct,
                    "y2": y2_pct,
                    "speaking_score": box['raw_score'],
                    "active": box['raw_score'] > 0,
                })

            # Sort by box size and keep only max_num_faces
            boxes = sorted(boxes, key=lambda x: (x['x2'] - x['x1']) * (x['y2'] - x['y1']), reverse=True)
            if len(boxes) > max_num_faces:
                boxes = boxes[:max_num_faces]
            
            faces.append({
                "frame_number": frame_number,
                "timestamp": round(frame_number / original_video_fps * 1000),
                "faces": boxes,
                "scene_number": segment_index
            })
            
            frame_count += 1
    
    # Return the combined results
    yield {
        "shots": shots,
        "faces": faces
    }

if __name__ == "__main__":
    TEST_URL = "https://storage.googleapis.com/sieve-prod-us-central1-public-file-upload-bucket/d979a930-f2a5-4e0d-84fe-a9b233985c4e/dba9cbf3-8374-44bc-8d9d-cc9833d3f502-input-file.mp4"
    # change "url" to "path" if you want to test with a local file
    file = sieve.File(url=TEST_URL)
    for out in process(file):
        print(out)
