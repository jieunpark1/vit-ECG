import os
from pathlib import Path
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import wfdb
from tqdm import tqdm  # ✅ 진행률 표시용
import requests

# ✅ 레코드 리스트 로드
def load_record_list(hea_csv_path):
    df = pd.read_csv(hea_csv_path)
    return df['path'].tolist()  # 예: ['100/100001234', '101/100001235']

# ✅ 라벨 생성 함수
def make_label(measurement_path):
    df = pd.read_csv(measurement_path, low_memory=False)
    report_cols = [col for col in df.columns if col.startswith('report_')]

    def is_abnormal(row):
        keywords = ['abnormal', 'consider', 'infarct', 'ischemia', 'mi']
        for col in report_cols:
            val = str(row[col]).lower()
            if any(keyword in val for keyword in keywords):
                return 1
        return 0

    labels = df.apply(is_abnormal, axis=1)
    return labels.tolist()


# ✅ 유효한 파일인지 검사 (0바이트 파일 제외)
def is_valid_file(file_path):
    return file_path.exists() and file_path.stat().st_size > 0


# ✅ ecg data 다운로드 함수
def download_record_files(rec, base_url, save_dir):
    """
    rec: ex) 'p1027/p10270654/s47328550/47328550'
    base_url: 'https://physionet.org/content/mimic-iv-ecg/1.0/'
    save_dir: local Path
    """
    for ext in ['hea', 'dat']:
        url = f"{base_url}{rec}.{ext}"
        save_path = save_dir / f"{rec}.{ext}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if is_valid_file(save_path):
            continue
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
            return False
    return True


# ✅ 신호 전처리 함수
def preprocess_signal(signal, target_length=5000):
    if signal.shape[0] > target_length:
        signal = signal[:target_length]
    elif signal.shape[0] < target_length:
        pad = target_length - signal.shape[0]
        signal = np.pad(signal, ((0, pad), (0, 0)), 'constant')
    return signal

# ✅ PyTorch Dataset 클래스
class MIMICIVECGDataset(Dataset):
    def __init__(self,
                 hea_csv_path='data/record_list.csv',
                 measurement_path='data/machine_measurements.csv',
                 data_dir='/mnt/hdd14/jieun/mimic-iv-ecg/',
                 database_name='mimic-iv-ecg-1.0',
                 base_url='https://physionet.org/content/mimic-iv-ecg/1.0/',
                 train=True,
                 train_ratio=0.8,
                 shuffle=True,
                 max_records=300000,
                 target_length=5000):

        self.data_dir = Path(data_dir)
        self.database_name = database_name
        self.target_length = target_length
        self.train_ratio = train_ratio
        self.base_url = base_url
        self.signal_paths = load_record_list(hea_csv_path)
        self.labels = make_label(measurement_path)
        self.max_records = max_records
        
        #한정된 개수의 데이터만 가져오기
        if self.max_records is not None:
            self.signal_paths = self.signal_paths[:self.max_records]
            self.labels = self.labels[:self.max_records]

        
        if shuffle:
            paired = list(zip(self.signal_paths, self.labels))
            random.shuffle(paired)
            self.signal_paths, self.labels = zip(*paired)

        split_idx = int(len(self.signal_paths) * self.train_ratio)
        if train:
            self.signal_paths = self.signal_paths[:split_idx]
            self.labels = self.labels[:split_idx]
            print(f"✅ Using {len(self.signal_paths)} training samples.")
        else:
            self.signal_paths = self.signal_paths[split_idx:]
            self.labels = self.labels[split_idx:]
            print(f"✅ Using {len(self.signal_paths)} validation samples.")

        # ✅ 다운로드 진행률 표시 + 실패 기록
        failures = []
        print("🔄 Checking and downloading missing records...")
        for rec in tqdm(self.signal_paths, desc="📥 Downloading records", unit="rec"):
            rec_path = self.data_dir / rec
            hea_file = rec_path.with_suffix('.hea')
            dat_file = rec_path.with_suffix('.dat')

            if not (hea_file.exists() and dat_file.exists()):
                success = download_record_files(rec, self.base_url, self.data_dir)
                if not success:
                    failures.append(rec)
                

        # ✅ 실패 로그 출력
        if failures:
            print(f"\n⚠️ Failed to download {len(failures)} record(s):")
            for f in failures:
                print(" -", f)

    def __len__(self):
        return len(self.signal_paths)

    def __getitem__(self, idx):
        rec = self.signal_paths[idx]
        label = self.labels[idx]
        rec_path = self.data_dir / rec

        try:
            record = wfdb.rdrecord(str(rec_path))
            signal = record.p_signal
        except Exception as e:
            print(f"❌ Error reading {rec_path}: {e}")
            raise e

        signal = preprocess_signal(signal, self.target_length)
        signal = torch.tensor(signal.T, dtype=torch.float32)
        return signal, label

