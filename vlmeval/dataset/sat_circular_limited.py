"""
SAT Circular dataset with sample limit support.
"""
import re
import pandas as pd
import numpy as np
from .image_mcq import ImageMCQDataset


class SATCircularLimited(ImageMCQDataset):
    """
    SAT Circular dataset with optional sample limit.
    Inherits from ImageMCQDataset and adds nsamples parameter.
    """

    TYPE = 'MCQ'

    DATASET_URL = {
        "SAT_circular": "https://huggingface.co/datasets/luckychao/vlmevalkit_tsv/resolve/main/SAT_circular.tsv",
    }

    DATASET_MD5 = {
        "SAT_circular": "3853dc9e3d3c5af96e6d7ed7d058859e",
    }

    def __init__(self, dataset='SAT_circular', nsamples=None, **kwargs):
        self.nsamples = nsamples
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['SAT_circular', 'SAT_circular_10']

    def load_data(self, dataset):
        # Remove the suffix if it's SAT_circular_10
        base_dataset = dataset.replace('_10', '')

        # Load data using parent class method
        df = super().load_data(base_dataset)

        # Limit samples if nsamples is set
        if self.nsamples is not None:
            df = df.head(self.nsamples)

        return df


def _sat_rule_based_evaluate(eval_file):
    """Rule-based evaluation: extract answer from <answer> tags, no GPT judge needed."""
    from ..smp import load, dump

    suffix = eval_file.split('.')[-1]
    data = load(eval_file)

    for i in range(len(data)):
        item = data.iloc[i]
        pred = str(item.get('prediction', ''))
        gt = str(item.get('answer', '')).strip().upper()

        m = re.search(r'<answer>\s*(.*?)\s*</answer>', pred, re.IGNORECASE)
        if m:
            letter = m.group(1).strip().upper().replace('.', '').replace(')', '')
        else:
            letter = ''

        data.loc[data.index[i], 'hit'] = 1 if letter == gt else 0

    result_file = eval_file.replace(f'.{suffix}', f'_result.{suffix}')
    dump(data, result_file)

    # Per-category accuracy
    categories = data['category'].unique() if 'category' in data.columns else ['all']
    res = {}
    for cat in categories:
        cat_data = data[data['category'] == cat] if 'category' in data.columns else data
        res[cat] = cat_data['hit'].mean()

    overall = np.mean(list(res.values()))
    res['Overall'] = overall

    result = {'split': ['test'], 'Overall': [overall]}
    for cat in sorted(c for c in res if c != 'Overall'):
        result[cat] = [res[cat]]

    res_df = pd.DataFrame(result)
    score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
    dump(res_df, score_file)
    return res_df


class SATPerspectiveTaking(ImageMCQDataset):
    """
    SAT Circular dataset - Only Perspective Taking category.
    Filters for samples where category == 'perspective'.
    """

    TYPE = 'MCQ'

    DATASET_URL = {
        "SAT_circular": "https://huggingface.co/datasets/luckychao/vlmevalkit_tsv/resolve/main/SAT_circular.tsv",
    }

    DATASET_MD5 = {
        "SAT_circular": "3853dc9e3d3c5af96e6d7ed7d058859e",
    }

    def __init__(self, dataset='SAT_perspective', nsamples=None, **kwargs):
        self.nsamples = nsamples
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['SAT_perspective', 'SAT_perspective_10']

    def evaluate(self, eval_file, **judge_kwargs):
        return _sat_rule_based_evaluate(eval_file)

    def load_data(self, dataset):
        # Load the base SAT_circular dataset
        df = super().load_data('SAT_circular')

        # Filter for perspective taking category
        df = df[df['category'] == 'perspective'].reset_index(drop=True)

        # Limit samples if nsamples is set
        if self.nsamples is not None:
            df = df.head(self.nsamples)

        return df

