"""
All-Angles-Bench Dataset for VLMEvalKit.
Multi-View Understanding Benchmark for MLLMs.

Source: https://github.com/Chenyu-Wang567/All-Angles-Bench
HuggingFace: ch-chenyu/All-Angles-Bench

Supports:
- AllAnglesBench_EgoHumans: EgoHumans subset (170 samples, all images available)
- AllAnglesBench_Full: Full dataset (2132 samples, requires Ego-Exo4D images)

Categories (6 types):
- counting
- attribute_identification
- relative_distance
- relative_direction
- manipulation
- camera_pose_estimation
"""

import base64
import io
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from .image_mcq import ImageMCQDataset


def pil_to_base64(pil_image, format='JPEG'):
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


class AllAnglesBenchDataset(ImageMCQDataset):
    """
    All-Angles-Bench Multi-View Understanding Dataset.

    Categories (6 types):
    - counting: Object counting across views
    - attribute_identification: Identifying object attributes
    - relative_distance: Estimating relative distances
    - relative_direction: Determining directional relationships
    - manipulation: Understanding object manipulation
    - camera_pose_estimation: Estimating camera positions

    Metrics:
    - Per-category accuracy
    - Overall accuracy (unweighted average of all categories)
    """

    TYPE = 'MCQ'

    # Data paths
    DATA_DIR = '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/VLMEvalKit_Thinkmorph/external_benchmark_codebase/All-Angles-Bench/data'
    DATA_FILE = 'egohumans_only.json'  # Default to EgoHumans subset

    def __init__(self, dataset='AllAnglesBench_EgoHumans', **kwargs):
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AllAnglesBench_EgoHumans']

    def load_data(self, dataset):
        """Load All-Angles-Bench data."""

        data_path = os.path.join(self.DATA_DIR, self.DATA_FILE)
        with open(data_path, 'r') as f:
            data = json.load(f)

        records = []
        for idx, sample in enumerate(data):
            # Load images as base64
            img_b64_list = []
            for img_path in sample['image_path']:
                full_path = os.path.join(self.DATA_DIR, img_path)
                if os.path.exists(full_path):
                    img = Image.open(full_path)
                    img_b64_list.append(pil_to_base64(img))
                else:
                    # Skip sample if any image missing
                    break
            else:
                # All images loaded successfully
                records.append({
                    'index': idx,
                    'image': img_b64_list,  # Multi-image as list
                    'question': sample['question'],
                    'A': str(sample.get('A', '')),
                    'B': str(sample.get('B', '')),
                    'C': str(sample.get('C', '')),
                    'answer': sample['answer'],
                    'category': sample['category'],
                    'folder': sample.get('folder', ''),
                    'sourced_dataset': sample.get('sourced_dataset', ''),
                })

        return pd.DataFrame(records)

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Custom evaluation with per-category accuracy.
        Overall is unweighted average of all category accuracies.
        """
        from ..smp import load, dump

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', f'_result.{suffix}')

        data = load(eval_file)

        # Score each sample
        if 'hit' not in data.columns:
            for i in range(len(data)):
                item = data.iloc[i]
                pred = str(item.get('prediction', '')).strip().upper()
                gt = str(item.get('answer', '')).strip().upper()

                # Check exact match first
                hit = 1 if pred == gt else 0

                # Check for <answer>X</answer> format (ThinkMorph style)
                if hit == 0:
                    answer_match = re.search(r'<answer>([A-C])</answer>', pred)
                    if answer_match and answer_match.group(1) == gt:
                        hit = 1

                # Check for "answer is X" or "answer: X" patterns
                if hit == 0:
                    answer_pattern = re.search(r'(?:answer|choice|option)\s*(?:is|:)?\s*([A-C])\b', pred, re.IGNORECASE)
                    if answer_pattern and answer_pattern.group(1).upper() == gt:
                        hit = 1

                # Check for standalone letter at the end (e.g., "...so the answer is A" -> last letter)
                if hit == 0:
                    # Find last occurrence of A, B, or C as a word boundary
                    last_letter = re.findall(r'\b([A-C])\b', pred)
                    if last_letter and last_letter[-1] == gt:
                        hit = 1

                data.loc[data.index[i], 'hit'] = hit
            dump(data, result_file)

        # Compute per-category accuracy
        category_acc = {}
        categories = data['category'].unique()

        for cat in categories:
            cat_data = data[data['category'] == cat]
            acc = cat_data['hit'].mean() * 100  # Percentage
            category_acc[cat] = acc

        # Compute Overall as unweighted average of all category accuracies
        overall_acc = np.mean(list(category_acc.values())) if category_acc else 0.0

        # Build result DataFrame
        res = {'Category': ['Overall'], 'Accuracy': [overall_acc], 'Count': [len(data)]}
        for cat in sorted(categories):
            cat_count = len(data[data['category'] == cat])
            res['Category'].append(cat)
            res['Accuracy'].append(category_acc[cat])
            res['Count'].append(cat_count)

        res_df = pd.DataFrame(res)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(res_df, score_file)

        return res_df


class AllAnglesBench_EgoHumans(AllAnglesBenchDataset):
    """
    EgoHumans subset of All-Angles-Bench (170 samples).
    All images available without needing Ego-Exo4D license.
    """

    DATA_FILE = 'egohumans_only.json'

    def __init__(self, dataset='AllAnglesBench_EgoHumans', **kwargs):
        super(AllAnglesBenchDataset, self).__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AllAnglesBench_EgoHumans']


class AllAnglesBench_Full(AllAnglesBenchDataset):
    """
    Full All-Angles-Bench dataset (2132 samples).
    Requires Ego-Exo4D images to be downloaded separately.
    """

    DATA_FILE = 'data.json'

    def __init__(self, dataset='AllAnglesBench_Full', **kwargs):
        super(AllAnglesBenchDataset, self).__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AllAnglesBench_Full']
