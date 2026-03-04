import os.path as osp
import warnings
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
from PIL import Image
import torch


class Chameleon(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='facebook/chameleon-7b', **kwargs):
        try:
            from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
        except Exception as e:
            logging.critical('Please install the latest transformers.')
            raise e

        processor = ChameleonProcessor.from_pretrained(model_path)
        model = ChameleonForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)

        self.model = model.cuda().eval()
        self.processor = processor

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = ''
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        # Completion-style prompt for base model (no instructions)
        prompt = f'{question}\n{options_prompt}Answer:'

        message = [dict(type='image', value=s) for s in tgt_path]
        message.extend([dict(type='text', value=prompt)])
        return message

    def generate_inner(self, message, dataset=None):
        content, images = '', []
        for x in message:
            if x['type'] == 'text':
                content += x['value']
            elif x['type'] == 'image':
                content += '<image>\n'
                images.append(Image.open(x['value']))

        inputs = self.processor(
            text=[content],
            images=images,
            padding=True,
            return_tensors='pt'
        ).to(device='cuda', dtype=torch.bfloat16)
        generate_ids = self.model.generate(**inputs, max_new_tokens=256)
        input_token_len = inputs.input_ids.shape[1]
        text = self.processor.batch_decode(
            generate_ids[:, input_token_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return text
