from datasets import get_dataset_config_names, load_dataset
from typing import List, Dict, Optional


def get_kmmlu_subjects() -> List[str]:
    """KMMLU의 모든 subject 목록 반환"""
    subjects = get_dataset_config_names("HAERAE-HUB/KMMLU")
    return subjects


def load_kmmlu_dataset(subject: str, split: str = "test"):
    """특정 subject의 KMMLU 데이터셋 로드"""
    dataset = load_dataset("HAERAE-HUB/KMMLU", subject, split=split)
    return dataset


def load_multiple_subjects(subjects: List[str], split: str = "test") -> Dict:
    """여러 subject를 한번에 로드"""
    datasets = {}
    for subject in subjects:
        datasets[subject] = load_kmmlu_dataset(subject, split)
    return datasets
