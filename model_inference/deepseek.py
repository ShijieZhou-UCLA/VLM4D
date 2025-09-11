import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from DeepSeek_VL2.deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from DeepSeek_VL2.deepseek_vl2.utils.io import load_pil_images

from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
from utils.video_process import read_video, download_video
import os
from utils.prepare_input import prepare_qa_text_input, prepare_multi_image_input, resize_deepseek
import json
import base64
from tqdm import tqdm


def generate_by_deepseek(model_name, 
                            queries, 
                            prompt, 
                            total_frames, 
                            temperature, 
                            max_tokens):

    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_name)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    responses = []

    # query = queries[1] 
    for query in tqdm(queries):
        # read frames
        video_path, video_id = download_video(video_url=query['video'])
        image_subdir = os.path.join("video_cache", video_id, f"{total_frames}_frames")
        tmp_file = os.path.join(image_subdir, "base64frames.json")
        images_path = []

        if not os.path.exists(image_subdir):
            os.makedirs(image_subdir, exist_ok=True)
            base64frames, _ = read_video(video_path, total_frames)
            for i, frame in enumerate(base64frames):
                with open(os.path.join(image_subdir, f"frame_{i}.jpg"), "wb") as f:
                    f.write(base64.b64decode(frame))
                images_path.append(os.path.join(image_subdir, f"frame_{i}.jpg"))
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(base64frames, f)
        else:
            for i in range(total_frames):
                images_path.append(os.path.join(image_subdir, f"frame_{i}.jpg"))
        
        # prepare conversation
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
        placeholders = "".join(f"This is frame_{i} <image> of the input video\n"
                            for i, _ in enumerate(images_path, start=1))
        
        conversation = [
            {
            "role": "<|User|>",
            "content": placeholders + qa_text_prompt,
            "images": images_path,
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        # print(conversation[0]["content"])

        pil_images = load_pil_images(conversation)
        # pil_images = [resize_deepseek(image, 32) for image in pil_images]
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=False #True
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        # print('answer??????',answer)
        # print(f"{prepare_inputs['sft_format'][0]}", answer)
        # responses = [answer]

        responses.append(answer)

        del outputs

    return responses


def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):

    responses = generate_by_deepseek(model_name, 
                                      queries, 
                                      prompt=prompt, 
                                      total_frames=total_frames, 
                                      temperature=GENERATION_TEMPERATURE, 
                                      max_tokens=MAX_TOKENS)
    for query, response in zip(queries, responses):
        query["response"] = response
    
    json.dump(queries, open(output_path, "w"), indent=4, ensure_ascii=False)