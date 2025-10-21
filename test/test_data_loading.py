import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_kmmlu_subjects, load_kmmlu_dataset

if __name__ == "__main__":
    # 1. subject 목록 확인
    subjects = get_kmmlu_subjects()
    print(f"Total subjects: {len(subjects)}")
    print(f"First 5: {subjects[:5]}")

    # 2. 샘플 데이터 구조 확인
    dataset = load_kmmlu_dataset("Accounting", split="test")
    print(f"\nDataset size: {len(dataset)}")
    print(f"\nSample structure:")
    print(dataset[5])

    # 3. 데이터 필드 확인
    print(f"\nColumns: {dataset.column_names}")
