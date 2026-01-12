"""
SAT Circular dataset with sample limit support.
"""
import pandas as pd
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

    def load_data(self, dataset):
        # Load the base SAT_circular dataset
        df = super().load_data('SAT_circular')

        # Filter for perspective taking category
        df = df[df['category'] == 'perspective'].reset_index(drop=True)

        # Limit samples if nsamples is set
        if self.nsamples is not None:
            df = df.head(self.nsamples)

        return df

