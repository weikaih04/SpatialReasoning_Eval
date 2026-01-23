import base64
import io
import json
import os
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from .image_mcq import ImageMCQDataset


def pil_to_base64(pil_image, format='PNG'):
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


class AI2ThorPathTracing(ImageMCQDataset):
    """
    AI2Thor Path Tracing QA Dataset.
    Source: linjieli222/ai2thor-path-tracing-qa-v7
    Uses only topdown_image (single image per sample).
    4-choice MCQ (A/B/C/D).
    """

    TYPE = 'MCQ'

    def __init__(self, dataset='AI2ThorPathTracing', nsamples=None, **kwargs):
        self.nsamples = nsamples
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorPathTracing']

    def load_data(self, dataset):
        from datasets import load_dataset

        # Load from HuggingFace
        hf_ds = load_dataset('linjieli222/ai2thor-path-tracing-qa-v7', split='train')

        records = []
        for idx, ex in enumerate(hf_ds):
            # Early exit if we have enough samples
            if self.nsamples is not None and len(records) >= self.nsamples:
                break

            # Convert topdown_image to base64
            img_b64 = pil_to_base64(ex['topdown_image'])

            # Map choices list to A/B/C/D
            choices = ex['choices']

            records.append({
                'index': len(records),
                'image': img_b64,
                'question': ex['question'],
                'A': choices[0] if len(choices) > 0 else '',
                'B': choices[1] if len(choices) > 1 else '',
                'C': choices[2] if len(choices) > 2 else '',
                'D': choices[3] if len(choices) > 3 else '',
                'answer': ex['answer'],
            })

        return pd.DataFrame(records)


class AI2ThorPerspective_NoArrow(ImageMCQDataset):
    """
    AI2Thor Perspective QA Dataset (No Arrow version).
    Source: weikaih/ai2thor-perspective-qa-800-balanced-val-v3
    Uses marked_image_no_arrow and question_no_arrow.
    2-choice MCQ (A/B).

    Categories (6 splits):
    - distance_change_closer
    - distance_change_further
    - relative_position_left_left
    - relative_position_left_right
    - relative_position_right_left
    - relative_position_right_right

    Overall is computed as unweighted average of 6 category accuracies.
    """

    TYPE = 'MCQ'

    def __init__(self, dataset='AI2ThorPerspective_NoArrow', nsamples=None, **kwargs):
        self.nsamples = nsamples
        super().__init__(dataset=dataset, **kwargs)

    def evaluate(self, eval_file, **judge_kwargs):
        """Custom evaluate that computes Overall as unweighted average of category accuracies."""
        from collections import defaultdict
        import numpy as np
        from ..smp import load, dump
        from .utils import build_judge

        # First run the parent evaluation to get per-sample results
        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', f'_result.{suffix}')

        data = load(eval_file)

        # Build judge for answer matching
        judge_kwargs['model'] = judge_kwargs.get('model', 'exact_matching')

        # Score each sample
        if 'hit' not in data.columns:
            for i in range(len(data)):
                item = data.iloc[i]
                pred = str(item.get('prediction', ''))
                gt = str(item.get('answer', ''))
                # Simple exact matching for A/B answers
                hit = 1 if pred.strip().upper() == gt.strip().upper() else 0
                # Also check if prediction contains the answer
                if hit == 0 and gt.strip().upper() in pred.strip().upper():
                    hit = 1
                data.loc[data.index[i], 'hit'] = hit
            dump(data, result_file)

        # Compute per-category accuracy
        category_acc = {}
        categories = data['category'].unique()

        for cat in categories:
            cat_data = data[data['category'] == cat]
            acc = cat_data['hit'].mean() * 100  # Convert to percentage
            category_acc[cat] = acc

        # Compute Overall as unweighted average of category accuracies
        overall_acc = np.mean(list(category_acc.values()))

        # Build result DataFrame
        res = {'Category': ['Overall'], 'Accuracy': [overall_acc], 'Count': [len(data)]}
        for cat in sorted(categories):
            cat_count = len(data[data['category'] == cat])
            res['Category'].append(cat)
            res['Accuracy'].append(category_acc[cat])
            res['Count'].append(cat_count)

        res_df = pd.DataFrame(res)

        # Save results
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(res_df, score_file)

        return res_df

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorPerspective_NoArrow']

    def load_data(self, dataset):
        from datasets import load_dataset

        # Load all splits from HuggingFace
        hf_ds = load_dataset('weikaih/ai2thor-perspective-qa-800-balanced-val-v3')

        records = []
        done = False
        for split_name in hf_ds.keys():
            if done:
                break
            for ex in hf_ds[split_name]:
                # Early exit if we have enough samples
                if self.nsamples is not None and len(records) >= self.nsamples:
                    done = True
                    break

                # Convert marked_image_no_arrow to base64
                img_b64 = pil_to_base64(ex['marked_image_no_arrow'])

                # Map answer_choices to A/B (answer_choices is a JSON string)
                choices_raw = ex['answer_choices']
                if isinstance(choices_raw, str):
                    choices = json.loads(choices_raw)
                else:
                    choices = choices_raw

                records.append({
                    'index': len(records),
                    'image': img_b64,
                    'question': ex['question_no_arrow'],
                    'A': choices[0] if len(choices) > 0 else '',
                    'B': choices[1] if len(choices) > 1 else '',
                    'answer': ex['answer'],
                    'category': split_name,  # Use split name for 6-category breakdown
                })

        return pd.DataFrame(records)


class AI2ThorPerspective_Arrow(ImageMCQDataset):
    """
    AI2Thor Perspective QA Dataset (With Arrow version).
    Source: weikaih/ai2thor-perspective-qa-800-balanced-val-v3
    Uses marked_image_with_arrow and question_with_arrow.
    2-choice MCQ (A/B).

    Categories (6 splits):
    - distance_change_closer
    - distance_change_further
    - relative_position_left_left
    - relative_position_left_right
    - relative_position_right_left
    - relative_position_right_right

    Overall is computed as unweighted average of 6 category accuracies.
    """

    TYPE = 'MCQ'

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorPerspective_Arrow']

    def evaluate(self, eval_file, **judge_kwargs):
        """Custom evaluate that computes Overall as unweighted average of category accuracies."""
        import numpy as np
        from ..smp import load, dump

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', f'_result.{suffix}')

        data = load(eval_file)

        # Score each sample
        if 'hit' not in data.columns:
            for i in range(len(data)):
                item = data.iloc[i]
                pred = str(item.get('prediction', ''))
                gt = str(item.get('answer', ''))
                hit = 1 if pred.strip().upper() == gt.strip().upper() else 0
                if hit == 0 and gt.strip().upper() in pred.strip().upper():
                    hit = 1
                data.loc[data.index[i], 'hit'] = hit
            dump(data, result_file)

        # Compute per-category accuracy
        category_acc = {}
        categories = data['category'].unique()

        for cat in categories:
            cat_data = data[data['category'] == cat]
            acc = cat_data['hit'].mean() * 100
            category_acc[cat] = acc

        # Compute Overall as unweighted average of category accuracies
        overall_acc = np.mean(list(category_acc.values()))

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

    def load_data(self, dataset):
        from datasets import load_dataset

        # Load all splits from HuggingFace
        hf_ds = load_dataset('weikaih/ai2thor-perspective-qa-800-balanced-val-v3')

        records = []
        global_idx = 0
        for split_name in hf_ds.keys():
            for ex in hf_ds[split_name]:
                # Convert marked_image_with_arrow to base64
                img_b64 = pil_to_base64(ex['marked_image_with_arrow'])

                # Map answer_choices to A/B (answer_choices is a JSON string)
                choices_raw = ex['answer_choices']
                if isinstance(choices_raw, str):
                    choices = json.loads(choices_raw)
                else:
                    choices = choices_raw

                records.append({
                    'index': global_idx,
                    'image': img_b64,
                    'question': ex['question_with_arrow'],
                    'A': choices[0] if len(choices) > 0 else '',
                    'B': choices[1] if len(choices) > 1 else '',
                    'answer': ex['answer'],
                    'category': split_name,  # Use split name for 6-category breakdown
                })
                global_idx += 1

        return pd.DataFrame(records)


class SideviewOverfit(ImageMCQDataset):
    """
    Sideview Overfit Test Dataset.
    Loads first N samples from sideview training parquet data to test if model can reproduce training samples.
    Uses original training instruction directly (no MCQ format).
    ThinkMorph.py has special handling to use the question field as-is.
    """

    TYPE = 'VQA'  # Not MCQ, but we still inherit from ImageMCQDataset for convenience

    # Path to sideview training parquet data
    PARQUET_DIR = '/weka/oe-training-default/jieyuz2/improve_segments/visual_cot/ThinkMorph_training/bagel_example/editing/ai2thor-path-tracing-train-sideview-only'

    def __init__(self, dataset='SideviewOverfit', nsamples=10, **kwargs):
        self.nsamples = nsamples
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['SideviewOverfit']

    def load_data(self, dataset):
        # Find parquet files
        parquet_files = sorted([f for f in os.listdir(self.PARQUET_DIR) if f.endswith('.parquet')])
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.PARQUET_DIR}")

        records = []
        samples_loaded = 0

        for pq_file in parquet_files:
            if samples_loaded >= self.nsamples:
                break

            pq_path = os.path.join(self.PARQUET_DIR, pq_file)
            table = pq.read_table(pq_path)
            df_pq = table.to_pandas()

            for idx, row in df_pq.iterrows():
                if samples_loaded >= self.nsamples:
                    break

                # Extract input image (first image in image_list)
                image_list = row['image_list']
                if len(image_list) == 0:
                    continue

                # Convert bytes to PIL Image then to base64
                input_img_bytes = image_list[0]
                pil_img = Image.open(io.BytesIO(input_img_bytes))
                img_b64 = pil_to_base64(pil_img)

                # Use full training instruction as question (includes system prompt)
                instruction = row['instruction_list'][0] if len(row['instruction_list']) > 0 else ""

                # Extract expected output text for reference
                output_texts = row['output_text_list']
                expected_text = ' '.join(output_texts) if len(output_texts) > 0 else ""

                # No MCQ options - ThinkMorph.py will use instruction directly
                records.append({
                    'index': samples_loaded,
                    'image': img_b64,
                    'question': instruction,  # Full training instruction
                    'answer': expected_text,  # Expected output for reference
                })
                samples_loaded += 1

        return pd.DataFrame(records)


class AI2ThorMultiViewCounting(ImageMCQDataset):
    """
    AI2Thor Multi-View Counting Dataset.
    Source: weikaih/ai2thor-multiview-counting-val-800-v2-400
    Multi-image input (3-5 frames per sample).
    4-choice MCQ (A/B/C/D).

    Splits:
    - square: 200 samples with 4 fixed camera positions
    - rotation: 200 samples with 3-5 rotation frames
    """

    TYPE = 'MCQ'

    # Filter by trajectory type: None = all, 'square' or 'rotation'
    TRAJECTORY_FILTER = None

    def __init__(self, dataset='AI2ThorMultiViewCounting', nsamples=None, **kwargs):
        self.nsamples = nsamples
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorMultiViewCounting']

    def load_data(self, dataset):
        from datasets import load_dataset
        import re

        # Load from HuggingFace
        hf_ds = load_dataset('weikaih/ai2thor-multiview-counting-val-800-v2-400', split='train')

        records = []
        for idx, ex in enumerate(hf_ds):
            # Early exit if we have enough samples
            if self.nsamples is not None and len(records) >= self.nsamples:
                break

            # Filter by trajectory type if specified
            traj_id = ex.get('trajectory_id', '')
            if self.TRAJECTORY_FILTER is not None:
                if self.TRAJECTORY_FILTER == 'square' and 'square' not in traj_id.lower():
                    continue
                if self.TRAJECTORY_FILTER == 'rotation' and 'rotation' not in traj_id.lower():
                    continue

            # Collect all non-None frames
            frames = []
            for i in range(8):
                frame = ex.get(f'frame_{i}')
                if frame is not None:
                    frames.append(frame)

            if len(frames) == 0:
                continue

            # Convert frames to base64 (multi-image input)
            img_b64_list = [pil_to_base64(f) for f in frames]

            # Question already has choices formatted
            question = ex['question']

            # Parse choices from question if not separate
            # Format: "How many X? A) 1 B) 2 C) 3 D) 4"
            choices = ['', '', '', '']
            choice_pattern = r'([A-D])\)\s*(\d+)'
            matches = re.findall(choice_pattern, question)
            for letter, value in matches:
                idx_choice = ord(letter) - ord('A')
                if 0 <= idx_choice < 4:
                    choices[idx_choice] = value

            records.append({
                'index': len(records),
                'image': img_b64_list,  # List of base64 for multi-image
                'question': question,
                'A': choices[0],
                'B': choices[1],
                'C': choices[2],
                'D': choices[3],
                'answer': ex['answer'],
                'category': ex.get('movement_type', ''),
                'query_object': ex.get('query_object', ''),
            })

        return pd.DataFrame(records)


class AI2ThorMultiViewCounting_Square(AI2ThorMultiViewCounting):
    """Multi-View Counting - Square trajectory only (4 fixed camera positions)."""

    TRAJECTORY_FILTER = 'square'

    def __init__(self, dataset='AI2ThorMultiViewCounting_Square', nsamples=None, **kwargs):
        self.nsamples = nsamples
        ImageMCQDataset.__init__(self, dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorMultiViewCounting_Square']


class AI2ThorMultiViewCounting_Rotation(AI2ThorMultiViewCounting):
    """Multi-View Counting - Rotation trajectory only (3-5 rotation frames)."""

    TRAJECTORY_FILTER = 'rotation'

    def __init__(self, dataset='AI2ThorMultiViewCounting_Rotation', nsamples=None, **kwargs):
        self.nsamples = nsamples
        ImageMCQDataset.__init__(self, dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorMultiViewCounting_Rotation']


class AI2ThorMultiViewCounting_10(AI2ThorMultiViewCounting):
    """Quick test version with only 10 samples."""

    def __init__(self, dataset='AI2ThorMultiViewCounting_10', **kwargs):
        super().__init__(dataset=dataset, nsamples=10, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorMultiViewCounting_10']


class AI2ThorMultiViewCounting_Square_10(AI2ThorMultiViewCounting_Square):
    """Quick test version - Square only, 10 samples."""

    def __init__(self, dataset='AI2ThorMultiViewCounting_Square_10', **kwargs):
        super().__init__(dataset=dataset, nsamples=10, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorMultiViewCounting_Square_10']


class AI2ThorMultiViewCounting_Rotation_10(AI2ThorMultiViewCounting_Rotation):
    """Quick test version - Rotation only, 10 samples."""

    def __init__(self, dataset='AI2ThorMultiViewCounting_Rotation_10', **kwargs):
        super().__init__(dataset=dataset, nsamples=10, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorMultiViewCounting_Rotation_10']
