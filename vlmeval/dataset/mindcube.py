"""
MindCube Dataset for VLMEvalKit.
Spatial Mental Modeling from Limited Views benchmark.

Source: https://github.com/mll-lab-nu/MindCube
HuggingFace: MLL-Lab/MindCube

Supports:
- MindCube_Tiny: Full tinybench (1050 samples)
- MindCube_Tiny_200: Balanced 200-sample subset
"""

import base64
import io
import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from .image_mcq import ImageMCQDataset


def pil_to_base64(pil_image, format='PNG'):
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def bytes_to_base64(img_bytes):
    """Convert image bytes to base64 string."""
    return base64.b64encode(img_bytes).decode('utf-8')


def extract_answer_from_question(question, answer_key):
    """Extract answer text from question options.

    Args:
        question: Question string with options like "A. xxx B. yyy C. zzz"
        answer_key: Answer key like "A", "B", "C", etc.

    Returns:
        Full answer text like "A. xxx"
    """
    # Pattern to match options like "A. Above" or "A) Above"
    pattern = r'([A-E])[.\)]\s*([^A-E]+?)(?=\s*[A-E][.\)]|$)'
    matches = re.findall(pattern, question)
    for letter, text in matches:
        if letter == answer_key:
            return f"{letter}. {text.strip()}"
    return answer_key


class MindCubeDataset(ImageMCQDataset):
    """
    MindCube Spatial Reasoning Dataset.

    Settings (from ID prefix):
    - among: Multi-object relative positioning
    - around: Circular/omnidirectional spatial relationships
    - rotation: Orientation/rotation relationships
    - translation: Movement/displacement relationships (excluded from overall metrics)

    Metrics:
    - Per-setting accuracy
    - Overall accuracy (unweighted average, excluding translation)
    """

    TYPE = 'MCQ'

    # Data path
    DATA_PATH = '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/VLMEvalKit_Thinkmorph/external_benchmark_codebase/MindCube/data/data/raw/MindCube_tinybench.jsonl'
    # Image base directory
    IMAGE_BASE_DIR = '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/VLMEvalKit_Thinkmorph/external_benchmark_codebase/MindCube/data/data'

    # Settings to include in overall metrics (translation excluded per original codebase)
    SETTINGS_IN_OVERALL = {'among', 'around', 'rotation'}

    def __init__(self, dataset='MindCube_Tiny', nsamples=None, seed=42, **kwargs):
        self.nsamples = nsamples
        self.seed = seed
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['MindCube_Tiny']

    def load_data(self, dataset):
        """Load MindCube tinybench data."""

        with open(self.DATA_PATH, 'r') as f:
            lines = f.readlines()

        # Parse all samples
        all_samples = [json.loads(line) for line in lines]

        # Sample if nsamples specified
        if self.nsamples is not None and self.nsamples < len(all_samples):
            random.seed(self.seed)
            all_samples = random.sample(all_samples, self.nsamples)

        records = []
        for idx, sample in enumerate(all_samples):
            # Extract setting from id (e.g., "among_group693_q1_5_2" -> "among")
            setting = sample['id'].split('_')[0]

            # Load images from file paths and convert to base64
            img_b64_list = []
            for img_path in sample['images']:
                full_path = os.path.join(self.IMAGE_BASE_DIR, img_path)
                if os.path.exists(full_path):
                    img = Image.open(full_path)
                    img_b64_list.append(pil_to_base64(img))
                else:
                    print(f"Warning: Image not found: {full_path}")
                    continue

            # Parse question and extract choices
            question = sample['question']
            gt_answer = sample['gt_answer']

            # Extract choices from question text
            # Pattern: A. xxx B. yyy C. zzz D. www E. vvv
            choices = {'A': '', 'B': '', 'C': '', 'D': '', 'E': ''}
            choice_pattern = r'([A-E])[.\)]\s*([^A-E]+?)(?=\s*[A-E][.\)]|$)'
            matches = re.findall(choice_pattern, question)
            for letter, text in matches:
                choices[letter] = text.strip()

            records.append({
                'index': idx,
                'image': img_b64_list,  # Multi-image as list
                'question': question,
                'A': choices['A'],
                'B': choices['B'],
                'C': choices['C'],
                'D': choices['D'],
                'E': choices['E'],
                'answer': gt_answer,
                'category': setting,  # Use setting as category for metrics
                'sample_id': sample['id'],
                'type': sample.get('type', ''),
            })

        return pd.DataFrame(records)

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Custom evaluation with per-category accuracy.
        Overall is unweighted average of category accuracies (excluding translation).
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

                # Check if prediction contains the answer letter
                if hit == 0:
                    # Extract first letter from prediction
                    pred_match = re.search(r'\b([A-E])\b', pred)
                    if pred_match and pred_match.group(1) == gt:
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

        # Compute Overall as unweighted average of included categories
        included_accs = [acc for cat, acc in category_acc.items()
                        if cat in self.SETTINGS_IN_OVERALL]
        overall_acc = np.mean(included_accs) if included_accs else 0.0

        # Build result DataFrame
        res = {'Category': ['Overall'], 'Accuracy': [overall_acc], 'Count': [len(data)]}
        for cat in sorted(categories):
            cat_count = len(data[data['category'] == cat])
            in_overall = '✓' if cat in self.SETTINGS_IN_OVERALL else '✗'
            res['Category'].append(f"{cat} ({in_overall})")
            res['Accuracy'].append(category_acc[cat])
            res['Count'].append(cat_count)

        res_df = pd.DataFrame(res)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(res_df, score_file)

        return res_df


class MindCube_Tiny_200(MindCubeDataset):
    """
    Balanced 200-sample subset of MindCube Tinybench.
    Stratified sampling by setting (among, around, rotation).
    """

    def __init__(self, dataset='MindCube_Tiny_200', seed=42, **kwargs):
        self.seed = seed
        # Don't pass nsamples to parent, we do stratified sampling
        super(MindCubeDataset, self).__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['MindCube_Tiny_200']

    def load_data(self, dataset):
        """Load balanced 200-sample subset."""

        with open(self.DATA_PATH, 'r') as f:
            lines = f.readlines()

        # Parse all samples and group by setting
        samples_by_setting = defaultdict(list)
        for line in lines:
            sample = json.loads(line)
            setting = sample['id'].split('_')[0]
            samples_by_setting[setting].append(sample)

        # Calculate proportional sampling
        total = sum(len(v) for v in samples_by_setting.values())
        target_total = 200

        # Stratified sampling
        random.seed(self.seed)
        selected_samples = []

        for setting, samples in samples_by_setting.items():
            # Proportional count
            n_samples = max(1, int(len(samples) / total * target_total))
            # Don't exceed available samples
            n_samples = min(n_samples, len(samples))
            selected = random.sample(samples, n_samples)
            selected_samples.extend(selected)

        # If we need more samples to reach 200, add randomly
        if len(selected_samples) < target_total:
            remaining = []
            for setting, samples in samples_by_setting.items():
                for s in samples:
                    if s not in selected_samples:
                        remaining.append(s)
            extra_needed = target_total - len(selected_samples)
            if remaining:
                selected_samples.extend(random.sample(remaining, min(extra_needed, len(remaining))))

        # If we have too many, trim
        if len(selected_samples) > target_total:
            selected_samples = selected_samples[:target_total]

        # Convert to records
        records = []
        for idx, sample in enumerate(selected_samples):
            setting = sample['id'].split('_')[0]

            # Load images from file paths and convert to base64
            img_b64_list = []
            for img_path in sample['images']:
                full_path = os.path.join(self.IMAGE_BASE_DIR, img_path)
                if os.path.exists(full_path):
                    img = Image.open(full_path)
                    img_b64_list.append(pil_to_base64(img))
                else:
                    print(f"Warning: Image not found: {full_path}")
                    continue

            question = sample['question']
            gt_answer = sample['gt_answer']

            # Extract choices
            choices = {'A': '', 'B': '', 'C': '', 'D': '', 'E': ''}
            choice_pattern = r'([A-E])[.\)]\s*([^A-E]+?)(?=\s*[A-E][.\)]|$)'
            matches = re.findall(choice_pattern, question)
            for letter, text in matches:
                choices[letter] = text.strip()

            records.append({
                'index': idx,
                'image': img_b64_list,
                'question': question,
                'A': choices['A'],
                'B': choices['B'],
                'C': choices['C'],
                'D': choices['D'],
                'E': choices['E'],
                'answer': gt_answer,
                'category': setting,
                'sample_id': sample['id'],
                'type': sample.get('type', ''),
            })

        return pd.DataFrame(records)
