import os
from pathlib import Path
import numpy as np
import wfdb
import torch
from torch.utils.data import Dataset
import random
import pandas as pd

def load_hea_path(hea_csv_path):
    df = df.load(head_csv_path)
    file_path = df['path']
    data_dir = 'https://physionet.org/files/mimic-iv-ecg/1.0/' + file_path + '.hea'
    return data_dir

def is_abnormal(row):
    keywords = ['abnormal', 'consider', 'infarct', 'ischemia', 'mi']
    for col in report_cols:
        val = str(row[col]).lower()
        if any(keyword in val for keyword in keywords):
            return 1
    return 0


def make_label(measurement_path):
    df = df.load(measurement_path)
    report_cols = [col for col in meas_df.columns if col.startswith('report_')]
    label = report_cols.apply(is_abnormal, axis=1)
    return label
    

def remove_nan(signal):
    return np.nan_to_num(signal)

def truncate_or_pad(signal, target_length=5000):
    current_length = signal.shape[0]
    if current_length > target_length:
        return signal[:target_length]
    elif current_length < target_length:
        pad_width = target_length - current_length
        pad = np.zeros((pad_width, signal.shape[1]))
        return np.concatenate([signal, pad], axis=0)
    return signal

def zscore_normalize(signal):
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (signal - mean) / std

def preprocess_signal(signal, target_length=5000):
    signal = remove_nan(signal)
    signal = truncate_or_pad(signal, target_length)
    signal = zscore_normalize(signal)
    return signal.astype(np.float32)

class MIMICIVECGDataset(Dataset):
    def __init__(self,
                 data_dir='data/mimic-iv-ecg/files',
                 max_subjects=100,
                 max_records_per_subject=10,
                 target_length=5000,
                 train=True,
                 train_ratio=0.8,
                 shuffle=True):

        self.data_dir = Path(data_dir)
        self.signal_paths=[]
        self.database_name = 'mimic-iv-ecg'
        self.target_length = target_length
        self.records = []
        self.rec_path = []
        self.data_dir.mkdir(parents=True, exist_ok=True)


        #for hea in self.data_dir.rglob('*.hea'):
        #    rel = hea.relative_to(self.data_dir).with_suffix('')
        #    self.signal_paths.append(str(rel))
        print("📥 Fetching subject list...")
        subjects = wfdb.get_record_list(self.database_name)
        print(f"Found {len(subjects)} subjects")

        # 실제 레코드 경로 리스트로 확장
        for subj in subjects[:max_subjects]:
            pth = f"/files/subj"
            subj_spec1 = wfdb.rdrecord(pth)
            print(subj_spec1)
            for subj_spec2 in wfdb.get_record_list(f"{self.databse_name/subj/subj_spec1}"):
                print(subj_spec2[:10])
            #print("222222", len(subj), subj)
        #    try:
        #        recs = wfdb.get_record_list(f"{self.database_name}/{subj_clean}")
        #        print("recs", len(recs), recs[:5])
        #        if not recs:
        #            print(f"⚠️ No records found for subject {subj_clean}")
        #            continue
        #    except Exception as e:
        #        print(f"⚠️ Failed to get records for subject {subj_clean}: {e}")
        #        continue

        #for rec in recs[:max_records_per_subject]:
        #    full_rec = f"{subj_clean}/{rec}"
        #    self.records.append(full_rec)

        #if len(self.records) == 0:
        #    raise ValueError("❌ No valid records found in PhysioNet database.")

        if shuffle:
            random.shuffle(self.records)

        total = len(self.records)
        split_idx = int(total * train_ratio)
        if train:
            self.records = self.records[:split_idx]
            print(f"✅ Using {len(self.records)} training records.")
        else:
            self.records = self.records[split_idx:]
            print(f"✅ Using {len(self.records)} validation records.")

        # 다운로드 (없으면)
        for rec in self.records:
            local_path = self.data_dir / rec
            if not (local_path.with_suffix('.dat').exists() and local_path.with_suffix('.hea').exists()):
                print(f"📦 Downloading {rec}")
                wfdb.dl_database(
                    db_dir=self.database_name,
                    records=[rec],
                    dl_dir=str(self.data_dir)
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec_name = self.signal_paths[idx]
        full = self.data_dir / rec
        try:
            record = wfdb.rdrecord(str(full))
            signal = record.p_signal
        except Exception as e:
            print(f"❌ Failed to read record {rec_name}: {e}")
            raise e

        signal = record.p_signal
        signal = preprocess_signal(signal, target_length=self.target_length)
        # shape (time, channel) → (channel, time) for PyTorch
        signal = torch.tensor(signal.T, dtype=torch.float32)

        label = idx % 2  # 임시 dummy label (실제 레이블 없으면 변경 필요)
        return signal, label

