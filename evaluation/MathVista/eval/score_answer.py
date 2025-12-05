import os
import copy
import argparse
import json
import openai
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from utils import *
from prompts import *

# This function will be executed by each thread.
# It processes a single sample, calls the API, and handles retries.
def process_sample(inst_tuple):
    """
    Processes a single sample to get its judgment score.
    """
    # Unpack arguments for clarity
    inst, api_key, quick_match, trunk_length = inst_tuple
    
    # It's good practice to work on a copy in multithreaded contexts
    save_inst = copy.deepcopy(inst)

    # Apply response trunking if specified
    if 'pred_answer' in save_inst and trunk_length > 0:
        save_inst['pred_answer'] = ' '.join(save_inst['pred_answer'].split(' ')[-trunk_length:])

    try:
        judgement = match_answer(save_inst, api_key, quick_match)
        count = 0
        # Retry logic in case of invalid API response
        while True:
            count += 1
            if judgement.strip() in ['0', '1']:
                save_inst['judgement'] = int(judgement)
                break
            if count > 10:  # Set a reasonable retry limit
                print(f"Error: Max retries reached for sample {save_inst.get('sample_index', 'N/A')}. Last response: {judgement}")
                save_inst['judgement'] = -1  # Mark as error
                break
            
            print(f"Warning: Wrong return format '{judgement}'. Retrying for sample {save_inst.get('sample_index', 'N/A')}.")
            judgement = match_answer(save_inst, api_key, quick_match)

    except Exception as e:
        print(f"An exception occurred while processing sample {save_inst.get('sample_index', 'N/A')}: {e}")
        save_inst['judgement'] = -1 # Mark as error

    return save_inst

def create_test_prompt(demo_prompt, inst):
    demo_prompt = demo_prompt.strip()
    answer2 = inst['answer2']
    pred_answer = inst.get('pred_answer', '') # Use .get for safety
    full_prompt = demo_prompt.format(question="", gt=answer2, extraction=pred_answer)
    return full_prompt

def match_answer(inst, api_key, quick_match=False):
    # quick match
    if quick_match:
        return '1' if inst['answer2'] == inst.get('pred_answer') else '0'
    
    # general extraction using OpenAI API
    try:
        # Note: get_chat_response should be thread-safe.
        # The official openai library client is thread-safe.
        full_prompt = create_test_prompt(demo_prompt_score, inst)
        extraction = get_chat_response(full_prompt, api_key)
        return extraction.replace("Judgement:", "").strip()
    except Exception as e:
        print(f"Error in calling API for sample {inst.get('sample_index', 'N/A')}: {e}")
        return "" # Return empty string on failure to trigger retry

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_extraction_file', type=str, required=True, help='Input JSON file with samples.')
    parser.add_argument('--save_file', type=str, required=True, help='Output JSONL file to save results.')
    parser.add_argument('--api_key', type=str, required=True, help='API key for OpenAI.')
    parser.add_argument('--num_threads', type=int, default=16, help='Number of parallel threads to use.')
    parser.add_argument('--quick_match', action='store_true', help='Use simple string matching instead of GPT.')
    parser.add_argument('--cache', action='store_true', help='Skip samples that are already in the save_file.')
    parser.add_argument('--trunk_response', type=int, default=-1, help='Truncate response to the last N words before sending to GPT.')
    args = parser.parse_args()

    # Set API key globally for convenience, as the library uses it.
    openai.api_key = args.api_key

    # --- Caching Setup ---
    # Read already processed samples from the .jsonl file to avoid re-running them.
    processed_indices = set()
    if args.cache and os.path.exists(args.save_file):
        print(f"Cache enabled. Loading processed samples from {args.save_file}...")
        with open(args.save_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'sample_index' in data:
                        processed_indices.add(data['sample_index'])
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode line in cache file: {line.strip()}")
        print(f"Found {len(processed_indices)} already processed samples.")

    # --- Data Loading and Filtering ---
    print(f"Reading input file: {args.answer_extraction_file}...")
    results = read_json(args.answer_extraction_file)
    
    # Filter out samples that have already been processed
    tasks_to_run = [inst for inst in results if inst.get('sample_index') not in processed_indices]
    print(f"Total samples: {len(results)}. Samples to process: {len(tasks_to_run)}.")
    
    if not tasks_to_run:
        print("No new samples to process. Exiting.")
        exit()

    # Prepare arguments for each thread
    tasks_with_args = [(inst, args.api_key, args.quick_match, args.trunk_response) for inst in tasks_to_run]
    
    # --- Ensure Output Directory Exists ---
    # Create parent directories for the output file if they don't exist
    os.makedirs(os.path.dirname(args.save_file) or '.', exist_ok=True)
    
    
    # --- Multithreaded Execution ---
    # Open the output file in append mode.
    with open(args.save_file, 'a', encoding='utf-8') as f:
        # Create a thread pool.
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Use executor.map to apply process_sample to all tasks and get an iterator for results.
            # Wrap with tqdm for a live progress bar.
            results_iterator = executor.map(process_sample, tasks_with_args)
            
            print(f"Starting processing with {args.num_threads} threads...")
            for result in tqdm(results_iterator, total=len(tasks_to_run)):
                # As each result is completed, write it to the JSONL file immediately.
                if result:
                    f.write(json.dumps(result) + '\n')
                    f.flush() # Ensure it's written to disk right away.
    
    print(f"Processing complete. All results saved to {args.save_file}.")