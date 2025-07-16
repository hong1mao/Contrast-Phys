import json
import logging
import os
import glob
import re

import h5py
import numpy as np
import cv2
from torch.utils.data import Dataset
import pandas as pd
from ultralytics import YOLO


class PUREDataset(Dataset):
    """Simplified data loader for the UBFC-rPPG dataset."""

    def __init__(self, data_path, cached_path, file_list_path, split_ratio=(0.0, 1.0), chunk_length=128,
                 preprocess=True, YOLOv8_model_path="D:\\文档\\python项目\\models\\best_face.pt",
                 re_size=36, crop_face=True,
                 larger_box_coef=1.5, backend="HC", use_face_detection=True,
                 label_type="DiffNormalized", data_type=["DiffNormalized"]):
        """
        Args:
            data_path (str): Path to raw dataset folder.
            cached_path (str): Path to save preprocessed .npy files.
            file_list_path (str): Path to save/load file list CSV.
            split_ratio (tuple): (begin, end) ratio of subjects to use.
            chunk_length (int): Length of each video/label chunk.
        """
        self.data_path = data_path
        self.cached_path = cached_path
        self.file_list_path = file_list_path
        self.split_ratio = split_ratio
        self.chunk_length = chunk_length
        self.inputs = []
        self.labels = []
        self.re_size = re_size
        self.larger_box_coef = larger_box_coef
        self.backend = backend
        self.use_face_detection = use_face_detection
        self.label_type = label_type
        self.data_type = data_type
        self.crop_face = crop_face
        # 加载预训练的 YOLOv8 模型
        self.YOLOv8_model_path = YOLOv8_model_path
        self.yolo_model = None  # 不立即加载

        # Ensure directories exist
        os.makedirs(self.cached_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)

        # Preprocess or load from cache
        if preprocess:
            self._preprocess()
        self._load_file_list()

    def _init_YOLO(self):
        """延迟初始化 YOLOv8 模型"""
        if self.yolo_model is None:
            self.yolo_model = YOLO(self.YOLOv8_model_path)
            # 设置日志级别为 WARNING，以抑制 INFO 级别的输出
            logging.getLogger('ultralytics').setLevel(logging.WARNING)

    def __getstate__(self):
        # 在进程拷贝时排除 yolo_model
        state = self.__dict__.copy()
        if 'yolo_model' in state:
            del state['yolo_model']
        return state

    def __setstate__(self, state):
        # 恢复对象状态，并初始化 yolo_model 为 None
        self.__dict__.update(state)
        self.yolo_model = None  # 子进程不会尝试复制原来的模型

    def _get_raw_data(self):
        """Get all subject directories."""
        # 获取所有符合"两位数字-两位数字"格式的文件夹
        data_dirs = [
            d for d in glob.glob(os.path.join("D:\\PURE", "*-*"))
            if re.match(r'^\d{2}-\d{2}$', os.path.basename(d))
        ]
        dirs = [{"index": os.path.basename(d), "path": d} for d in data_dirs]
        return dirs

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        frames = list()
        all_png = sorted(glob.glob(video_file + '\\*.png'))
        for png_path in all_png:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            labels = json.load(f)
            waves = [label["Value"]["waveform"]
                     for label in labels["/FullPackage"]]
        return np.asarray(waves)

    def _save_subject_data(self, frames, bvps, filename):
        """Save subject's full data to single .h5 file."""
        h5_path = os.path.join(self.cached_path, f"{filename}.h5")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('imgs', data=frames)  # shape: (T, H, W, 3)
            f.create_dataset('bvp', data=bvps)  # shape: (T,)
        return h5_path

    def face_detection(self, frame, backend, use_larger_box=False, larger_box_coef=1.0):
        """Face detection on a single frame.

        Args:
            frame(np.array): a single frame.
            backend(str): backend to utilize for face detection.
            use_larger_box(bool): whether to use a larger bounding box on face detection.
            larger_box_coef(float): Coef. of larger box.
        Returns:
            face_box_coor(List[int]): coordinates of face bouding box.
        """
        if backend == "HC":
            # Use OpenCV's Haar Cascade algorithm implementation for face detection
            # This should only utilize the CPU
            detector = cv2.CascadeClassifier(
                './data/haarcascade_frontalface_default.xml')

            # Computed face_zone(s) are in the form [x_coord, y_coord, width, height]
            # (x,y) corresponds to the top-left corner of the zone to define using
            # the computed width and height.
            face_zone = detector.detectMultiScale(frame[:, :, :3].astype(np.uint8))

            if len(face_zone) < 1:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
            elif len(face_zone) >= 2:
                # Find the index of the largest face zone
                # The face zones are boxes, so the width and height are the same
                max_width_index = np.argmax(face_zone[:, 2])  # Index of maximum width
                face_box_coor = face_zone[max_width_index]
                print("Warning: More than one faces are detected. Only cropping the biggest one.")
            else:
                face_box_coor = face_zone[0]
        elif "YOLO" in backend:
            # Use a YOLO trained on WiderFace dataset
            # This utilizes both the CPU and GPU
            self._init_YOLO()
            results = self.yolo_model(frame[:, :, :3].astype(np.uint8))
            best_box = None
            max_area = 0
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2)
            if best_box != None:
                x_min, y_min, x_max, y_max = best_box
                # Convert to this toolbox's expected format
                # Expected format: [x_coord, y_coord, width, height]
                x = x_min
                y = y_min
                width = x_max - x_min
                height = y_max - y_min

                # Find the center of the face zone
                center_x = x + width // 2
                center_y = y + height // 2

                # Determine the size of the square (use the maximum of width and height)
                square_size = max(width, height)

                # Calculate the new coordinates for a square face zone
                new_x = center_x - (square_size // 2)
                new_y = center_y - (square_size // 2)
                face_box_coor = [new_x, new_y, square_size, square_size]

            else:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
        else:
            raise ValueError("Unsupported face detection backend!")

        if use_larger_box:
            face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
            face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
            face_box_coor[2] = larger_box_coef * face_box_coor[2]
            face_box_coor[3] = larger_box_coef * face_box_coor[3]
        return face_box_coor

    def crop_face_resize(self, frames, backend, use_larger_box, larger_box_coef, width, height,
                         use_face_detection=True):
        """Crop face and resize frames.

        Args:
            frames(np.array): Video frames.
            width(int): Target width for resizing.
            height(int): Target height for resizing.
            use_larger_box(bool): Whether enlarge the detected bouding box from face detection.
            larger_box_coef(float): the coefficient of the larger region(height and weight),
                                the middle point of the detected region will stay still during the process of enlarging.
            use_face_detection(bool): Whether to use face detection to crop the face region.
        Returns:
            resized_frames(list[np.array(float)]): Resized and cropped frames
        """
        total_frames, _, _, channels = frames.shape
        if self.crop_face:
            # Face Cropping
            face_region_all = []
            if use_face_detection:
                # 对每个帧进行人脸检测
                for i in range(0, total_frames):
                    frame = frames[i]
                    face_region_all.append(self.face_detection(frame, backend, use_larger_box, larger_box_coef))
            else:
                # 只对第一个帧进行人脸检测
                face_region_all.append(self.face_detection(frames[0], backend, use_larger_box, larger_box_coef))
            face_region_all = np.asarray(face_region_all, dtype='int')

        # Frame Resizing
        resized_frames = np.zeros((total_frames, height, width, channels))
        for i in range(0, total_frames):
            frame = frames[i]
            if self.crop_face:
                if use_face_detection:
                    assert len(face_region_all) == total_frames, "Face region detection failed!"
                    # 获得当前帧的人脸区域
                    reference_index = i
                else:
                    # use the first region obtrained from the first frame.
                    reference_index = 0
                face_region = face_region_all[reference_index]
                frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                        max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]

            resized_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return resized_frames

    def preprocess_data_and_bvps(self, frames, bvps):
        """Preprocesses a pair of data.

        Args:
            frames(np.array): Frames in a video.
            bvps(np.array): Blood volumne pulse (PPG) signal labels for a video.
        Returns:
            frame_clips(np.array): processed video data by frames
            bvps_clips(np.array): processed bvp (ppg) labels by frames
        """
        # resize frames and crop for face region
        frames = self.crop_face_resize(
            frames,
            backend=self.backend,
            use_larger_box=True,
            larger_box_coef=self.larger_box_coef,
            width=self.re_size,
            height=self.re_size,
            use_face_detection=self.use_face_detection)
        # Check data transformation type
        data = list()  # Video data
        for data_type in self.data_type:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(PUREDataset.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(PUREDataset.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)  # concatenate all channels
        if self.label_type == "DiffNormalized":
            # 差分归一化标签
            bvps = PUREDataset.diff_normalize_label(bvps)
        elif self.label_type == "Standardized":
            # 标准化标签
            bvps = PUREDataset.standardized_label(bvps)
        else:
            raise ValueError("Unsupported label type!")

        return data, bvps

    def _preprocess(self):
        """Main preprocessing function."""
        print("Preprocessing dataset...")

        # Get all data without splitting
        all_subjects = self._get_raw_data()
        h5_files = []

        for subject in all_subjects:
            print(f"Processing subject: {subject['index']}")
            subject_id = subject["index"]
            subject_path = subject["path"]

            # 得到图片和标签路径
            images_path = subject_path
            label_path = subject_path + ".json"

            frames = PUREDataset.read_video(images_path)
            bvps = PUREDataset.read_wave(label_path)
            # 重采样标签
            target_length = frames.shape[0]
            bvps = PUREDataset.resample_ppg(bvps, target_length)

            # Preprocess without chunking
            frames, bvps = self.preprocess_data_and_bvps(frames, bvps)
            h5_path = self._save_subject_data(frames, bvps, subject_id)
            h5_files.append(h5_path)

        # Save complete file list
        df = pd.DataFrame({"h5_files": sorted(h5_files)})
        df.to_csv(self.file_list_path, index=False)
        print(f"Preprocessing done. Processed {len(h5_files)} subjects.")

    def _load_file_list(self):
        """Load existing file list from CSV and split according to split_ratio."""
        df = pd.read_csv(self.file_list_path)
        all_files = df['h5_files'].tolist()

        # Split data by subject according to split_ratio
        begin, end = self.split_ratio
        n = len(all_files)
        start_idx = int(begin * n)
        end_idx = int(end * n)
        files_subset = all_files[start_idx:end_idx]

        self.inputs = sorted(files_subset)
        print(f"Loaded {len(self.inputs)} subjects from cache (split ratio: {self.split_ratio}).")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """Load random segment from subject's .h5 file."""
        with h5py.File(self.inputs[idx], 'r') as f:
            img_length = f['imgs'].shape[0]

            # Ensure we have enough frames
            if img_length < self.chunk_length:
                raise ValueError(
                    f"Subject {self.inputs[idx]} has only {img_length} frames, less than required {self.chunk_length}")

            idx_start = np.random.randint(0, img_length - self.chunk_length)
            idx_end = idx_start + self.chunk_length

            img_seq = f['imgs'][idx_start:idx_end]  # shape: (T, H, W, 3)
            bvp_seq = f['bvp'][idx_start:idx_end]  # shape: (T,)

            # Convert to float32 and transpose
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
            bvp_seq = bvp_seq.astype('float32')

        return img_seq, bvp_seq

    def get_test_segments(self, idx):
        """Load all segments from subject's .h5 file for testing.
        Returns:
            list[tuple]: List of (video_segment, label_segment) tuples
            Each segment is 300 frames long
        """
        with h5py.File(self.inputs[idx], 'r') as f:
            img_length = f['imgs'].shape[0]
            segments = []

            # Split the video into segments of 300 frames
            for start_idx in range(0, img_length, 300):
                end_idx = min(start_idx + 300, img_length)

                # Skip segments that are too short
                if end_idx - start_idx < 300:
                    continue

                img_seq = f['imgs'][start_idx:end_idx]  # shape: (300, H, W, 3)
                bvp_seq = f['bvp'][start_idx:end_idx]  # shape: (300,)

                # Convert to float32 and transpose
                img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
                bvp_seq = bvp_seq.astype('float32')

                segments.append((img_seq, bvp_seq))

        return segments

    @staticmethod
    def diff_normalize_data(data):
        """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
        n, h, w, c = data.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
        for j in range(diffnormalized_len):
            diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                    data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        return diffnormalized_data

    @staticmethod
    def diff_normalize_label(label):
        """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
        diff_label = np.diff(label, axis=0)
        diffnormalized_label = diff_label / np.std(diff_label)
        diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
        diffnormalized_label[np.isnan(diffnormalized_label)] = 0
        return diffnormalized_label

    @staticmethod
    def standardized_data(data):
        """Z-score standardization for video data."""
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data

    @staticmethod
    def standardized_label(label):
        """Z-score standardization for label signal."""
        label = label - np.mean(label)
        label = label / np.std(label)
        label[np.isnan(label)] = 0
        return label

    @staticmethod
    def resample_ppg(input_signal, target_length):
        """Samples a PPG sequence into specific length."""
        return np.interp(
            np.linspace(
                1, input_signal.shape[0], target_length), np.linspace(
                1, input_signal.shape[0], input_signal.shape[0]), input_signal)
