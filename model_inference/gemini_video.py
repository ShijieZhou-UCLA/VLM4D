from utils.prepare_input import prepare_qa_inputs, prepare_qa_text_input
from google import genai
import os
import time
from utils.video_process import download_video
from tqdm import tqdm
import json 

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):
    
    # DAILY_LIMIT = 1000 # Set based on your actual quota, leave some buffer
    # queries = queries[:DAILY_LIMIT] #queries[DAILY_LIMIT:2*DAILY_LIMIT]

    rpm_limit = 100
    request_interval = 60 / rpm_limit
    last_request_time = time.time()
    
    for query in tqdm(queries):
        elapsed = time.time() - last_request_time
        if elapsed < request_interval:
            time.sleep(request_interval - elapsed)
        last_request_time = time.time()

        video_path, _ = download_video(query['video'])
        video_file = client.files.upload(file=video_path)
        while video_file.state != "ACTIVE":
            time.sleep(0.5)
            video_file = client.files.get(name=video_file.name)
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
        # print(qa_text_prompt)

        output = client.models.generate_content(
            model=model_name, contents=[video_file, qa_text_prompt],
        )

        response = output.text
        # print(response)
        query["response"] = response

        client.files.delete(name=video_file.name)

    json.dump(queries, open(output_path, "w"), indent=4, ensure_ascii=False)
    