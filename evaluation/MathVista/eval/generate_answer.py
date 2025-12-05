import os
import json
import re
import argparse
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from prompts import lingual_prompt

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# ========== 模型加载 ==========
model_path = "../../../model/llava-onevision-qwen2-7b-ov-chat-hf"  # 实际本地路径
model_path = os.path.abspath(model_path)      # 确保是绝对路径
device = "cuda" if torch.cuda.is_available() else "cpu"

# 设置环境变量强制本地加载
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

try:
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map="auto",
        trust_remote_code=True  # 添加这个参数
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
    # 备用方案：不使用local_files_only
    print("Trying without local_files_only...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
from accelerate import dispatch_model
import torch
# 检查模型是否已经被分发
if hasattr(model, 'hf_device_map'):
    print("Model is already dispatched, no need to move to device")
else:
    model = model.to(device)

processor = AutoProcessor.from_pretrained(
    model_path,
    local_files_only=True  # 强制从本地加载
)

# ========== 回答提取函数（使用正则提取选项） ==========
def extract_formula_answer(formula_string):
    """
    提取 \\boxed{} 中的所有内容，支持嵌套大括号
    """
    is_formula = 1
    # 查找 \boxed{ 的位置
    boxed_pattern = r'\\boxed\s*\{'
    match = re.search(boxed_pattern, formula_string)
    
    if not match:
        return formula_string, 0
    
    # 找到开始位置
    start_pos = match.end() - 1  # 指向第一个 '{'
    
    # 使用括号计数来找到匹配的结束括号
    brace_count = 0
    pos = start_pos
    
    while pos < len(formula_string):
        if formula_string[pos] == '{':
            brace_count += 1
        elif formula_string[pos] == '}':
            brace_count -= 1
            if brace_count == 0:
                # 找到匹配的结束括号
                content = formula_string[start_pos + 1:pos]
                return content.strip(), is_formula
        pos += 1
    
    # 如果没有找到匹配的结束括号，返回原字符串
    is_formula = 0
    return formula_string, is_formula

def extract_answer(text):
    """
    从模型输出中提取选择题答案
    优先匹配 \\boxed{X} 格式，然后是其他格式
    """
    # 1. 匹配 \boxed{} 大括号中所有内容
    boxed_match = re.search(r'\\boxed\{([^}]*)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # 2. 匹配 "Answer: xx" 等格式，支持任意内容作为答案
    answer_match = re.search(r'(?:answer is |Answer\s*[:]\s*)(.+?)(?:\n|$)', text)
    if answer_match:
        return answer_match.group(1).strip()
    
    # 3. 匹配最后一个引号内的内容
    quote_matches = re.findall(r'[\'"]([^\'"]*?)[\'"]', text)
    if quote_matches:
        return quote_matches[-1].strip()  # 返回最后一个匹配项
    
    return text[-20:]

def extract_bracket(text):
    """
    使用正则表达式提取小括号内容
    
    Args:
        text (str): 输入的文本字符串
    
    Returns:
        str: 小括号内的内容，若无小括号则返回原text
    """
    # 使用正则表达式匹配第一个小括号内的内容
    match = re.search(r'\(([^)]*)\)', text)
    
    if match:
        return match.group(1)  # 返回第一个捕获组的内容
    else:
        return text  # 没有匹配到括号，返回原文本


def predict_answer(prompt, image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"图像加载失败: {image_path} — {e}")
        return "Invalid", "Invalid"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    inputs = processor(text=text, images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=8192,  
            temperature=0.1, 
            top_p=0.001, 
            # do_sample=False,      # 不启用采样
            repetition_penalty=1.1,  # 重复惩罚
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # 只获取新生成的token，排除输入部分
    input_length = inputs['input_ids'].shape[1]
    if len(output[0]) > input_length:
        generated_tokens = output[0][input_length:]
        response = processor.decode(generated_tokens, skip_special_tokens=True)
    else:
        print("警告: 模型没有生成新的token")
        response = ""
    
    return response, extract_answer(response)


# ========== 主处理逻辑 ==========

def calculate_category_accuracy(processed_data):
    """
    根据problem_version字段计算各类别的准确率
    """
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for item in processed_data:
        problem_version = item.get("problem_version", "unknown")
        category_stats[problem_version]['total'] += 1
        
        if item.get("pred_answer") == item.get("answer"):
            category_stats[problem_version]['correct'] += 1
    
    return category_stats


def print_accuracy_report(category_stats, total_correct, total_count):
    """
    打印详细的准确率报告
    """
    print("\n" + "="*60)
    print("准确率统计报告")
    print("="*60)
    
    # 总体准确率
    overall_acc = total_correct / total_count if total_count > 0 else 0
    print(f"总体准确率: {total_correct}/{total_count} = {overall_acc:.2%}")
    print("-"*60)
    
    # 各类别准确率
    print("各类别准确率:")
    for category, stats in sorted(category_stats.items()):
        correct = stats['correct']
        total = stats['total']
        acc = correct / total if total > 0 else 0
        print(f"  {category:<20}: {correct:>3}/{total:<3} = {acc:>6.2%}")
    
    print("="*60)


def append_single_result(output_path, entry):
    """
    追加单个结果到JSON文件
    实现真正的实时保存
    """
    try:
        # 读取现有数据
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
        
        # 添加新结果
        data.append(entry)
        
        # 写回文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"保存单个结果时出错: {e}")


def save_progress_periodically(output_path, processed_data, save_interval=10):
    """
    定期保存所有数据（备份策略）
    每处理save_interval个数据后完整保存一次
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        print(f"已保存进度，当前处理了 {len(processed_data)} 个数据")
    except Exception as e:
        print(f"定期保存时出错: {e}")


def process(input_path, output_path, image_root, lan):
    # 加载输入数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 如果输出文件已存在，则加载，跳过已完成的
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        with open(output_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        processed_ids = {item["pid"] for item in processed_data}
        print(f"发现已处理的数据: {len(processed_data)} 条")
    else:
        processed_data = []
        processed_ids = set()
        # 创建空的JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print("创建新的输出文件")

    # 从已处理数据中计算当前统计
    category_stats = calculate_category_accuracy(processed_data)
    correct = sum(1 for item in processed_data if item.get("pred_answer") == item.get("answer"))
    total = len(processed_data)

    # 正式处理
    pbar = tqdm(data, desc="Evaluating", unit="item")
    processed_count = 0  # 新处理的数据计数
    
    for entry in pbar:
        # 跳过已处理过的数据
        key = entry["pid"]
        if key in processed_ids:
            continue
        
        # # 跳过"according"开头的数据
        # query_cot = entry.get("query_cot", "")
        # if query_cot.startswith("According"):
        #     # print(f"跳过数据 (sample_index={entry.get('sample_index')}, problem_index={entry.get('problem_index')}): query_cot以'According'开头")
        #     continue
        
        # 跳过非数学类问题
        question_type = entry.get("question_type","")
        # metadata = entry.get("metadata", "")
        # if metadata.get("category", "") != "math-targeted-vqa":
        #     continue
        
        if lan == 'en':
            question_for_eval = entry.get("query", "")
        else:
            question_for_eval = entry.get(lan+"_question", "")
        if question_type == "multi_choice":
            # QUESTION, CHOICES = split_question_and_choices(question_for_eval)
            prompt = lingual_prompt[lan+"_mc"].format(question_for_eval=question_for_eval)
        else:
            # QUESTION = question_for_eval
            prompt = lingual_prompt[lan+"_ff"].format(question_for_eval=question_for_eval)
                
        image_rel_path = entry["image"]
        image_path = os.path.join(image_root, image_rel_path)
        print(f"\n处理: {image_path}")

        response, pred = predict_answer(prompt, image_path)
        entry["pred_answer"] = pred
        entry["response"] = response
        print(f"预测答案: {pred}")

         # 更新统计信息
        problem_version = entry.get("problem_version", "unknown")
        category_stats[problem_version]['total'] += 1
        
        
        question_type = entry.get("question_type","")
        if question_type == "multi_choice":
            choice_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
            answer_list = entry.get("choices", "")
            for i, choice in enumerate(choice_list):
                if extract_bracket(pred) == choice and i < len(answer_list):
                    pred == answer_list[i] 
           
        if pred == entry["answer"]:
            correct += 1
            category_stats[problem_version]['correct'] += 1
        total += 1

        processed_data.append(entry)
        processed_count += 1

        # # 方案1: 每个结果都立即保存（真正的实时保存）
        # append_single_result(output_path, entry)
        
        # 方案2: 每10个结果保存一次（平衡性能和安全性）
        if processed_count % 10 == 0:
            save_progress_periodically(output_path, processed_data, save_interval=10)

        # 更新 tqdm 描述，显示总体准确率
        overall_acc = correct / total if total > 0 else 0
        pbar.set_postfix(correct=correct, total=total, acc=f"{overall_acc:.2%}")

    # 处理完成后，打印详细的准确率报告
    final_category_stats = calculate_category_accuracy(processed_data)
    print_accuracy_report(final_category_stats, correct, total)
    
    # 最终完整保存一次（确保数据完整性）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    print(f"处理完成！最终保存了 {len(processed_data)} 条数据")


# ========== 参数解析 ==========
def parse_arguments():
    parser = argparse.ArgumentParser(description='运行模型评估脚本')
    parser.add_argument('--input_json', type=str, 
                       default="dataset/MathVerse/testmini.json",
                       help='输入JSON文件路径')
    parser.add_argument('--output_json', type=str, 
                       default="output/MathVerse/MathVerse_en.json",
                       help='输出JSON文件路径')
    parser.add_argument('--image_root', type=str, 
                       default="dataset/MathVerse/images",
                       help='图像根目录路径')
    parser.add_argument('--lan', type=str, 
                       default="de",
                       help='语言')
    return parser.parse_args()


# ========== 运行入口 ==========
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    # 打印使用的参数
    print("="*50)
    print("运行参数:")
    print(f"  输入文件: {args.input_json}")
    print(f"  输出文件: {args.output_json}")
    print(f"  图像目录: {args.image_root}")
    print("="*50)
    
    # 运行处理
    process(args.input_json, args.output_json, args.image_root, args.lan)