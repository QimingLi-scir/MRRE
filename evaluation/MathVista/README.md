# Benchmark Evaluation

## 1. Generate Answer

### Evaluation Command

Navigate to the evaluation directory and run the answer generation script:
```bash
cd eval

python generate_answer.py \
    --input_json "$INPUT_JSON" \
    --output_json "$OUTPUT_JSON" \
    --image_root "$IMAGE_ROOT" \
    --lan "$LAN"
```

Images can be downloaded from the MathVista dataset: https://huggingface.co/datasets/AI4Math/MathVista

## 2. Score Answer

### Input File Format

The evaluation code accepts JSON format input files, with each sample containing the following fields:
```json
[
  {
    "pid": "1",
    "question": "text",
    "image": "images/1.jpg",
    "choices": null,
    "unit": null,
    "precision": 1.0,
    "answer": "D",
    "question_type": "free_form",
    "answer_type": "float",
    "metadata": {
      "category": "math-targeted-vqa",
      "context": "scientific figure",
      "grade": "college",
      "img_height": 720,
      "img_width": 1514,
      "language": "english",
      "skills": "['scientific reasoning']",
      "source": "SciBench",
      "split": "testmini",
      "task": "textbook question answering"
    },
    "query": "Question text for evaluation",
    "zh_question": "Chinese question (optional)",
    "pred_answer": "D",
    "response": "Complete model response"
  }
]
```

### Evaluation Command

Navigate to the evaluation directory and run the scoring script:
```bash
cd eval

python score_answer.py \
    --answer_extraction_file input_file_path \
    --save_file output_file_path \
    --api_key API_KEY \
    --num_threads num_threads \
    --cache \
    --trunk_response 30
```

### Parameter Description

- `--answer_extraction_file`: Path to the input JSON file
- `--save_file`: Path to save evaluation results
- `--api_key`: API key
- `--num_threads`: Number of concurrent threads
- `--cache`: Enable caching
- `--trunk_response`: Response truncation length (default: 30)