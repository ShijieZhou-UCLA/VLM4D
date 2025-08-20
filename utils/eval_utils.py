from openai import AsyncAzureOpenAI
import asyncio
import os, json
from pydantic import BaseModel
from utils.api_utils import generate_from_openai_chat_completion_structured_format


class EvaluationOutput(BaseModel):
    extracted_answer: str
    correct: bool


INSTRUCTION="""Your task is to evaluate whether the model's final answer is correct by comparing it to the ground-truth answer provided for the given question.

You should first extract the final answer from the model's response, and then compare the extracted answer with the choice that matches the ground-truth answer to determine its correctness.
"""

MULTI_CHOICE_INSTRUCTION=INSTRUCTION + "Output your response in the following structured format:\n" + """{
    "extracted_answer": // str value "A" "B" "C" "D", followed by a colon and the corresponding answer text, e.g., "A: Answer A text". If the model's response does not contain a valid choice and reasoning, then "No Valid Answer".
    "correct": // boolean value, True if the extracted answer matches the ground-truth answer (correct choice), False otherwise ("No Valid Answer" is also considered False).
}
"""



def prepare_evaluation_message(example, response):
    user_prompt = ""
    question_type = example["question_type"]
    if question_type == "multiple-choice":
        optionized_list = [f"{key}: {value}" for i, (key, value) in enumerate(example['choices'].items())]
        optionized_str = "\n".join(optionized_list)
        question_context = f"Question: {example['question']}\n\nOptions:\n{optionized_str}"
    else:
        print("Unsupported question type:", question_type)
        return None
    
    gt_answer = f"Ground Truth Answer: {example['answer']}"
    model_response = f"Model Response to the Question: {response}"
    
    user_prompt = f"{question_context}\n\n{gt_answer}\n\n{model_response}"
    
    message = [
        {"role": "system", "content": MULTI_CHOICE_INSTRUCTION},
        {"role": "user", "content": user_prompt},
    ]
    return message

async def get_acc_async(examples, client):
    evaluation_messages = [
        prepare_evaluation_message(example, example['response'])
        for example in examples
    ]
    
    os.makedirs("cache", exist_ok=True)
    json.dump(evaluation_messages, open("cache/evaluation_messages.json", "w"), indent=4, ensure_ascii=False)
    
    # Just await directly, no asyncio.run():
    outputs = await generate_from_openai_chat_completion_structured_format(     
        client=client, 
        messages=evaluation_messages,
        engine_name="o4-mini", 
        max_tokens=1024, 
        requests_per_minute=1000,
        structured_format=EvaluationOutput
    )
    
    count = 0
    results = []
    for example, output in zip(examples, outputs):
        result = {
            "id": example["id"],
            "question": example["question"],
            "choices": example["choices"],
            "response": example["response"],
            "ground_truth_answer": example["answer"],  
        }
        try:
            result["extracted_answer"] = output.extracted_answer
            result["correct"] = output.correct
        except Exception as e:
            result["extracted_answer"] = ""
            result["correct"] = False
            print(f"Error: {e}")
        
        results.append(result)
        count += result["correct"]
            
    return count / len(examples), results
