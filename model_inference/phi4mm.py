import os
import requests
import torch
from PIL import Image
# import soundfile
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
###
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input
from utils.video_process import read_video, download_video, prepare_base64frames, prepare_base64_video
import json
from vllm.multimodal.utils import fetch_image
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
from tqdm import tqdm

def generate_by_phi4mm(model_name, 
                            queries, 
                            prompt, 
                            total_frames, 
                            temperature, 
                            max_tokens):

    model_path = model_name #"microsoft/Phi-4-multimodal-instruct"

    kwargs = {}
    kwargs['torch_dtype'] = torch.bfloat16

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # print(processor.tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype='auto',
        _attn_implementation='flash_attention_2',
    ).cuda()
    
    # print("model.config._attn_implementation:", model.config._attn_implementation)

    generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

    responses = []
    with torch.no_grad():
        for query in tqdm(queries):
            vision_input_base64 = prepare_base64frames(model_name=model_path, video_url=query['video'], total_frames=total_frames, video_tmp_dir="video_cache")
            vision_input = [fetch_image(f"data:image/jpeg;base64,{base64_image}") for base64_image in vision_input_base64]
            qa_text_message, qa_text_prompt = prepare_qa_text_input(model_path, query, prompt)
            placeholders = "".join(f"<|image_{i}|>"
                                for i, _ in enumerate(vision_input, start=1))

            messages = [
                {'role': 'user', 'content': placeholders + qa_text_prompt},
            ]


            final_prompt = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # print(f'>>> Prompt\n{final_prompt}')

            inputs = processor(final_prompt, vision_input, return_tensors='pt').to('cuda:0')


            generation_args = {
                'max_new_tokens': 1000,
                'temperature': 1.0, #0.0,
                'do_sample': False,
            }

    
            generate_ids = model.generate(
                **inputs, **generation_args, generation_config=generation_config,
            )

            # remove input tokens
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
            response = processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # print(response)
            responses.append(response)

            # Clean up to free memory
            del inputs
            torch.cuda.empty_cache()

    return responses


def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):

    assert model_name in ["microsoft/Phi-4-multimodal-instruct"], "Invalid model name"

    responses = generate_by_phi4mm(model_name, 
                                      queries, 
                                      prompt=prompt, 
                                      total_frames=total_frames, 
                                      temperature=GENERATION_TEMPERATURE, 
                                      max_tokens=MAX_TOKENS)
    for query, response in zip(queries, responses):
        query["response"] = response
    
    json.dump(queries, open(output_path, "w"), indent=4, ensure_ascii=False)