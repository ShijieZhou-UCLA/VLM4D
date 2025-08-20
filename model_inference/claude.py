from utils.prepare_input import prepare_qa_inputs
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT
import anthropic
import os
import time
import json 
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()


def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=10)
    messages = prepare_qa_inputs(model_name, queries, total_frames, prompt=prompt)

    for i, query in tqdm(enumerate(queries), total=len(queries)):
        for attempt in range(10):
            try:
                output = client.messages.create(
                    model=model_name, #claude-sonnet-4-20250514
                    max_tokens=1024,
                    messages=messages[i],
                )
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == 9:
                    output = 'error'
                else:
                    time.sleep(100)
        response = output.content[0].text
        query["response"] = response
    
    json.dump(queries, open(output_path, "w"), indent=4, ensure_ascii=False)
