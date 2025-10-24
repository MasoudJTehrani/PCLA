# Description: This file contains utility functions for the InternVL2 model.
# Partially taken from: https://huggingface.co/OpenGVLab/InternVL2-1B

import importlib.util
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from hydra.utils import to_absolute_path
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_num_image_tokens_per_patch(encoder_variant: str) -> int:
    # we want to know how many image tokens we use so that we can adjust the batch padding
    tmp_config = AutoConfig.from_pretrained(encoder_variant, trust_remote_code=True)
    image_size = tmp_config.force_image_size or tmp_config.vision_config.image_size
    patch_size = tmp_config.vision_config.patch_size
    num_image_tokens = int((image_size // patch_size) ** 2 * (tmp_config.downsample_ratio ** 2))
    return num_image_tokens

def get_assistant_loss_mask(user_starts, assistant_starts, prompt_tokenized_ids):
    # assistant_start_end = []
    seq_length = prompt_tokenized_ids.shape[1]
    loss_mask = torch.zeros(prompt_tokenized_ids.shape, dtype=torch.bool) # loss is calculated where this mask is True
    
    for batch_id, (user_list, assistant_list) in enumerate(zip(user_starts, assistant_starts)):
        # batch_pairs = []
        # assume we start with user always:
        assert user_list[0] < assistant_list[0], "First user start should be before first assistant start"
        assert len(user_list) == len(assistant_list), "Number of user and assistant starts should be the same"
        
        for i, start in enumerate(assistant_list):
            # End is the start of the next user sequence OR the last index if it's the final sequence
            end = user_list[i + 1] - 1 if i < len(user_list) - 1 else seq_length - 1  # updated variable name to seq_length
            # batch_pairs.append((start, end))
            loss_mask[batch_id, start:end+1] = True

        # assistant_start_end.append(batch_pairs)
    return loss_mask


def get_chat_tokens(tokenizer, prompts: List[str], user_start_token_str: str, assistant_start_token_str: str) -> Dict:
    # this handles also multi-round conversations, which is not used by simlingo -> code would simplify by a lot if only condering one round
    # but since i have already implemented it, i will keep it
    prompt_tokenized = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
    prompt_tokenized_ids = prompt_tokenized["input_ids"]
    prompt_tokenized_valid = prompt_tokenized["input_ids"] != tokenizer.pad_token_id
    prompt_tokenized_mask = prompt_tokenized_valid

    # mask user prompt (question) to calculate loss only on assistant tokens (answer)

    user_start_token_ids = torch.tensor(tokenizer(user_start_token_str)["input_ids"])
    assistant_start_token_ids = torch.tensor(tokenizer(assistant_start_token_str)["input_ids"])

    seq_len_to_find = user_start_token_ids.shape[0]
    seq_len_to_find_assistant = assistant_start_token_ids.shape[0]
    # Create a mask by sliding the sequence across the original tensor
    matches_user = (prompt_tokenized_ids.unfold(1, seq_len_to_find, 1) == user_start_token_ids).all(dim=2)
    matches_assistant = (prompt_tokenized_ids.unfold(1, seq_len_to_find_assistant, 1) == assistant_start_token_ids).all(dim=2)
    # Get all matches
    match_indices_user = torch.nonzero(matches_user, as_tuple=True)
    match_indices_assistant = torch.nonzero(matches_assistant, as_tuple=True)
    
    # tuple(batch id), tuple(start index) -> dict key: batch id, value: list of start indices
    position_user_start_indices = [[] for _ in range(len(match_indices_user[0]))]
    position_assistant_start_indices = [[] for _ in range(len(match_indices_assistant[0]))]
    for i in range(len(match_indices_user[0])):
        batch_id = match_indices_user[0][i].item()
        position_user_start_indices[batch_id].append(match_indices_user[1][i].item())
    for i in range(len(match_indices_assistant[0])):
        batch_id = match_indices_assistant[0][i].item()
        position_assistant_start_indices[batch_id].append(match_indices_assistant[1][i].item())

    # get the start and end of the assistant
    loss_mask = get_assistant_loss_mask(position_user_start_indices, position_assistant_start_indices, prompt_tokenized_ids)

    return {
        'phrase_ids': prompt_tokenized_ids,
        'phrase_valid': prompt_tokenized_valid,
        'phrase_mask': prompt_tokenized_mask,
        'language_string': prompts,
        'loss_masking': loss_mask
    }


def get_custom_chat_template(conversations: List[Dict], tokenizer, encoder_variant: str, num_image_tokens_total: int, cache_root_dir: str = 'pretrained') -> Optional[Dict]:
    # get the custom chat template
    # for full conversation, question only
    # https://huggingface.co/docs/transformers/main/en/chat_templating#can-i-use-chat-templates-in-training
    # this adds special tokens and bring it in the right format for the pretrained LLM
        
    # taken from:
    # https://github.com/OpenGVLab/InternVL/blob/9d3a709b16874e73ffdd38b9cf53296fae4589b9/internvl_chat/internvl/train/constants.py#L7
    # https://github.com/OpenGVLab/InternVL/blob/9d3a709b16874e73ffdd38b9cf53296fae4589b9/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py#L294
    IMG_START_TOKEN='<img>'
    IMG_END_TOKEN='</img>'
    IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
    IMG_TOKEN = '<image>'

    cache_dir = f"{cache_root_dir}/{(encoder_variant.split('/')[1])}"
    # get absolute path from workspace dir not wokring dir
    cache_dir = to_absolute_path(cache_dir)
    model_path = f"{cache_dir}/conversation.py"
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=encoder_variant, local_dir=cache_dir)
        
    #import from file from model_path
    spec = importlib.util.spec_from_file_location('get_conv_template', model_path)
    conv_module = importlib.util.module_from_spec(spec)
    sys.modules['get_conv_template'] = conv_module
    spec.loader.exec_module(conv_module)

    image_tokens_templates = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_tokens_total + IMG_END_TOKEN

    prompts_conv = []
    prompts_question = []
    # get the custom chat template
    for idx, conv in enumerate(conversations):
        assert len(conv) == 2, "For question and answer templates only two turn conversation (user + assistant) is supported. During training is should work but is not checked!!"
        template_conv = conv_module.get_conv_template('internlm2-chat')
        template_question = conv_module.get_conv_template('internlm2-chat')

        # add full conversation
        for conv_part_idx, conv_part in enumerate(conv):
            content_str = conv_part['content'][0]['text']
            if conv_part['role'] == 'assistant':
                template_conv.append_message(template_conv.roles[1], content_str)
            elif conv_part['role'] == 'user':
                if conv_part_idx == 0 and IMG_TOKEN not in content_str:
                    content_str = f"{IMG_TOKEN}\n" + content_str
                template_conv.append_message(template_conv.roles[0], content_str)
            else:
                raise ValueError(f"Role {conv_part['role']} not supported")
        
        assert conv[0]['role'] == 'user', "First turn should be user as this should be the question."
        content_str_user = conv[0]['content'][0]['text']
        if IMG_TOKEN not in content_str_user:
            content_str_user = f"{IMG_TOKEN}\n" + content_str_user
        template_question.append_message(template_question.roles[0], content_str_user)
        template_question.append_message(template_question.roles[1], None)

        # get the prompt
        prompt_conv = template_conv.get_prompt()
        prompt_question = template_question.get_prompt()

        # replace system prompt to reduce tokens and save memory
        # template_conv.system_template -> '<|im_start|>system\n{system_message}'
        system_prompt = template_conv.system_template.replace('{system_message}', template_conv.system_message) + template_conv.sep
        prompt_conv = prompt_conv.replace(system_prompt, '')
        prompt_question = prompt_question.replace(system_prompt, '')

        # replace <image> with image token placeholders
        prompt_conv = prompt_conv.replace(IMG_TOKEN, image_tokens_templates, 1)
        prompt_question = prompt_question.replace(IMG_TOKEN, image_tokens_templates, 1)

        prompts_conv.append(prompt_conv)
        prompts_question.append(prompt_question)

    # on list of prompts to get the padding right
    user_start_token_str = template_conv.roles[0]
    assistant_start_token_str = template_conv.roles[1]

    conv_dict = get_chat_tokens(tokenizer, prompts_conv, user_start_token_str, assistant_start_token_str)
    question_dict = get_chat_tokens(tokenizer, prompts_question, user_start_token_str, assistant_start_token_str)

    return conv_dict, question_dict



def preprocess_image_batch(
        images_batch_list, 
        input_size=448, 
        use_global_img=False, 
        max_num_grid=2
    ):
    
    transform = build_transform(input_size=input_size)
    images_processed_tmp = []
    images_sizes_tmp = []
    for idx, img in enumerate(images_batch_list):
        image_np = img.numpy().astype(np.uint8)
        image_np = np.transpose(image_np, (1, 2, 0))
        image = Image.fromarray(image_np)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=use_global_img, max_num=max_num_grid)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        images_processed_tmp.append(pixel_values)
        images_sizes_tmp.append([image.size[1], image.size[0]])
    
    images_processed = {
        'pixel_values': torch.stack(images_processed_tmp), 
        'image_sizes': torch.tensor(images_sizes_tmp)
        }
    return images_processed


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values