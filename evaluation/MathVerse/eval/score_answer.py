import os
import copy
import argparse
from tqdm import tqdm
from collections import defaultdict
from utils import *

# OpenAI
import openai

from prompts import *


# load demo prompt
def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, inst):
    demo_prompt = demo_prompt.strip()
    answer = inst['answer']
    pred_answer = inst['pred_answer'][-100:]
    full_prompt = demo_prompt.format(question = "", gt=answer, extraction=pred_answer)
    return full_prompt

def match_answer(inst, api_key, quick_match=False):
    # quick match
    if quick_match:
        return '1' if inst['answer'] == inst['pred_answer'] else '0'
    # general extraction
    try:
        full_prompt = create_test_prompt(demo_prompt_score, inst)
        extraction = get_chat_response(full_prompt, api_key)
        return extraction.replace("Judgement:", "").strip()
    except Exception as e:
        print(e)
        print(f"Error in matching answer")

    return ""


def trunk_response(response, trunk_length):
    if trunk_length <= 0:
        return response
    else:
        return_res = ' '.join(response.split(' ')[-trunk_length:])
        return return_res

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--answer_extraction_file', type=str, default='answer.json')
    parser.add_argument('--save_file', type=str, default='answer.json')
    # match
    parser.add_argument('--quick_match', action='store_true', help='use rules to match answer for some problems')
    # output
    parser.add_argument('--save_every', type=int, default=10, help='save every n problems')
    parser.add_argument('--cache', action='store_true', help='cache results')
    parser.add_argument('--trunk_response', type=int, default=-1, help='trunk response to the last n words')
    parser.add_argument('--api_key', type=str, help='api key for openai')
    # args
    args = parser.parse_args()

    # set api key
    openai.api_key = args.api_key

    # read results
    result_file = args.answer_extraction_file
    print(f"Reading {result_file}...")
    results = read_json(result_file)

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    if os.path.exists(args.save_file):
        save_results = json.load(open(args.save_file))
    else:
        save_results = []

    score_dict = defaultdict(lambda: defaultdict(list))
    score_version_dict = defaultdict(list)
    # tqdm, enumerate results
    for i, inst in enumerate(tqdm(results)):
        save_inst = save_results[i] if i < len(save_results) else copy.deepcopy(inst)
        if args.cache and 'judgement' in save_inst:
            pass
        else:
            judgement = match_answer(save_inst, args.api_key, args.quick_match)
            count = 0
            while True:
                count += 1
                if count > 20:
                    print(f"Error in matching answer for {save_inst['sample_index']} , {judgement}")
                    save_inst['judgement'] = -1
                    break
                if judgement.strip() not in ['0', '1']:
                    print('Wrong return format: ', judgement)
                    judgement = match_answer(save_inst, args.api_key, args.quick_match)
                else:
                    save_inst['judgement'] = int(judgement)
                    break

            save_results.append(save_inst)

        if i % args.save_every == 0 or i == len(results)-1:
            print(f"Saving results to {args.save_file}...")
            save_json(save_results, args.save_file)
            print(f"Results saved.")
      