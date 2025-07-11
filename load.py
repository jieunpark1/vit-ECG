import wfdb
import numpy as np
import sys
from pathlib import Path

database_name = 'mimic-iv-ecg/1.0'

subjects = wfdb.get_record_list(database_name)
print(f"The '{database_name}' database contains data from {len(subjects)} subjects")


subject = subjects[0]
print("예시 subject:", subject)

# 그 subject 아래에 있는 record 경로들
records = wfdb.get_record_list(f'{database_name}/{subject}')
print(f"{subject}에는 {len(records)}개의 record가 있습니다.")
print("예시 record:", records[:5])

print("""
Note the formatting of these records:
 - intermediate directory ('p100' in this case)
 - subject identifier (e.g. 'p10014354')
 - record identifier (e.g. '81739927'
 """)
