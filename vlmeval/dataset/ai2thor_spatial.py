import base64
import io
import json
import pandas as pd
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
            # Convert topdown_image to base64
            img_b64 = pil_to_base64(ex['topdown_image'])

            # Map choices list to A/B/C/D
            choices = ex['choices']

            records.append({
                'index': idx,
                'image': img_b64,
                'question': ex['question'],
                'A': choices[0] if len(choices) > 0 else '',
                'B': choices[1] if len(choices) > 1 else '',
                'C': choices[2] if len(choices) > 2 else '',
                'D': choices[3] if len(choices) > 3 else '',
                'answer': ex['answer'],
            })

        df = pd.DataFrame(records)
        if self.nsamples is not None:
            df = df.head(self.nsamples)
        return df


class AI2ThorPerspective_NoArrow(ImageMCQDataset):
    """
    AI2Thor Perspective QA Dataset (No Arrow version).
    Source: weikaih/ai2thor-perspective-qa-800-balanced-val-v3
    Uses marked_image_no_arrow and question_no_arrow.
    2-choice MCQ (A/B).
    """

    TYPE = 'MCQ'

    def __init__(self, dataset='AI2ThorPerspective_NoArrow', nsamples=None, **kwargs):
        self.nsamples = nsamples
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorPerspective_NoArrow']

    def load_data(self, dataset):
        from datasets import load_dataset

        # Load all splits from HuggingFace
        hf_ds = load_dataset('weikaih/ai2thor-perspective-qa-800-balanced-val-v3')

        records = []
        global_idx = 0
        for split_name in hf_ds.keys():
            for ex in hf_ds[split_name]:
                # Convert marked_image_no_arrow to base64
                img_b64 = pil_to_base64(ex['marked_image_no_arrow'])

                # Map answer_choices to A/B (answer_choices is a JSON string)
                choices_raw = ex['answer_choices']
                if isinstance(choices_raw, str):
                    choices = json.loads(choices_raw)
                else:
                    choices = choices_raw

                records.append({
                    'index': global_idx,
                    'image': img_b64,
                    'question': ex['question_no_arrow'],
                    'A': choices[0] if len(choices) > 0 else '',
                    'B': choices[1] if len(choices) > 1 else '',
                    'answer': ex['answer'],
                    'category': ex['question_type'],
                })
                global_idx += 1

        df = pd.DataFrame(records)
        if self.nsamples is not None:
            df = df.head(self.nsamples)
        return df


class AI2ThorPerspective_Arrow(ImageMCQDataset):
    """
    AI2Thor Perspective QA Dataset (With Arrow version).
    Source: weikaih/ai2thor-perspective-qa-800-balanced-val-v3
    Uses marked_image_with_arrow and question_with_arrow.
    2-choice MCQ (A/B).
    """

    TYPE = 'MCQ'

    @classmethod
    def supported_datasets(cls):
        return ['AI2ThorPerspective_Arrow']

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
                    'category': ex['question_type'],
                })
                global_idx += 1

        return pd.DataFrame(records)

