import json
# lan_list = ["en","zh", 'ja', 'es', 'ru', 'de', 'fr', 'th', 'sw']
lan_list = ["zh", 'ja', 'es', 'ru', 'de', 'fr', 'th', 'sw']
choice_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
for lan in lan_list:

    with open(f'path_to_folder/{lan}.json', 'r') as f:
        data = json.load(f)
        # print(lan, len(data))
    for entry in data:
        answer = entry["answer"]
        choices = entry["choices"]
        if entry["question_type"] == "multi_choice":
            for i, choice in enumerate(choices):
                if answer == choice:
                    entry["answer2"] = choice_list[i]
        else:
            entry["answer2"] = answer
    
    with open(f'intern/mathvista_m2/{lan}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    