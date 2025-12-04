demo_prompt_extract = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.

1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n\\boxed{(-2, 1)}'
Extracted Answer: (-2, 1)

2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D

3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)

4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: not a answer

5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3

6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1

7.
"""


demo_prompt_score = """
Below are two answers to a math question. [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent by comparing their numerical values or options, ignoring units, irrelevant text, or formatting differences. Extract only the numerical value, mathematical expression, or option (e.g., A, B, C) from both answers for comparison.
If the numerical values, expressions, or options are identical or equivalent (e.g., 0.5m and 50cm), Judgement is 1; otherwise, Judgement is 0. Answer the judgment directly without any explanation.

[Standard Answer]: B
[Model_answer] : B
Judgement: 1

[Standard Answer]: C
[Model_answer] : C.
Judgement: 1

[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \((-2, 1)\)
Judgement: 0

[Standard Answer]: \((-2, 1)\)
[Model_answer] : Extracted Answer: (-2, 1)
Judgement: 1

[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \((-4, 1]\)
Judgement: 0

[Standard Answer]: y=-2 t+180
[Model_answer] : y = -2t + 180
Judgement: 1

[Standard Answer]:  100
[Model_answer] :  1000
Judgement: 0

[Standard Answer]: Volume $=33.51 \mathrm{{~cm}}^{{3}}$
[Model_answer] : 33.51
Judgement: 1

[Standard Answer]:  1000
[Model_answer] :  Jibu:Kikombe cha upima inategemea 1000 ml.
Judgement: 1

[Standard Answer]: {gt}
[Model_answer] : {extraction}
Judgement: """



lingual_prompt = {
    "en_mc":"Please first conduct reasoning, then answer the question and put the correct option letter into \\boxed{{}}, e.g., \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}}, at the end.\nQuestion: {question_for_eval}",
    "zh_mc":"请先进行推理，然后回答问题，并在最后将正确的选项字母填入\\boxed{{}}中，例如\\boxed{{A}}、\\boxed{{B}}、\\boxed{{C}}、\\boxed{{D}}。\n问题：{question_for_eval}",
    "ja_mc":"まず推論を行い、次に質問に答えて、正しいオプション文字を \\boxed{{}} に入れます。例: \\boxed{{A}}、\\boxed{{B}}、\\boxed{{C}}、\\boxed{{D}}、最後に。\n質問: {question_for_eval}",
    "es_mc":"Primero realice el razonamiento, luego responda la pregunta y coloque la letra de la opción correcta en \\boxed{{}}, por ejemplo, \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}}, al final.\nPregunta: {question_for_eval}",
    "ru_mc":"Сначала проведите рассуждение, затем ответьте на вопрос и вставьте букву правильного варианта ответа в \\boxed{{}}, например, \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}}, в конце.\nВопрос: {question_for_eval}",
    "fr_mc":"Veuillez d'abord effectuer un raisonnement, puis répondre à la question et mettre la lettre d'option correcte dans \\boxed{{}}, par exemple, \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}}, à la fin.\nQuestion : {question_for_eval}",
    "de_mc":"Bitte führen Sie zunächst eine Argumentation durch, beantworten Sie dann die Frage und setzen Sie am Ende den richtigen Optionsbuchstaben in \\boxed{{}}, z. B. \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}}.\nFrage: {question_for_eval}",
    "th_mc":"โปรดใช้เหตุผลก่อน จากนั้นตอบคำถามและใส่ตัวอักษรตัวเลือกที่ถูกต้องลงใน \\boxed{{}} เช่น \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}} ที่ส่วนท้าย ที่ท้ายคำถาม: {question_for_eval}",
    "sw_mc":"Tafadhali kwanza elekeza hoja, kisha ujibu swali na uweke herufi ya chaguo sahihi kwenye \\ boxed{{}}, k.m., \\ boxed{{A}}, \\ boxed{{B}}, \\ boxed{{C}}, \\boxed{{D}}, mwishoni.\nSwali:{question_for_eval}",
    "en_ff":"Please first conduct reasoning, then answer the question and put the final answer into \\boxed{{}} at the end.\nQuestion: {question_for_eval}",
    "zh_ff":"请先进行推理，然后回答问题，并将最终答案放入最后的\\boxed{{}}中。\n问题：{question_for_eval}",
    "ja_ff":"まず推論を行い、次に質問に答え、最終的な回答を最後に \\boxed{{}} に入力してください。\n質問: {question_for_eval}",
    "es_ff":"Por favor, primero realice el razonamiento, luego responda la pregunta y coloque la respuesta final en \\boxed{{}} al final.\nPregunta: {question_for_eval}",
    "ru_ff":"Сначала проведите рассуждение, затем ответьте на вопрос и поместите окончательный ответ в \\boxed{{}} в конце.\nВопрос: {question_for_eval}",
    "fr_ff":"Veuillez d'abord effectuer un raisonnement, puis répondre à la question et mettre la réponse finale dans \\boxed{{}} à la fin.\nQuestion : {question_for_eval}",
    "de_ff":"Bitte führen Sie zunächst eine Argumentation durch, beantworten Sie dann die Frage und tragen Sie die endgültige Antwort am Ende in \\boxed{{}} ein.\nFrage: {question_for_eval}",
    "th_ff":"กรุณาใช้เหตุผลก่อน จากนั้นตอบคำถาม และใส่คำตอบสุดท้ายลงในช่อง \\boxed{{}} ที่ท้ายคำถาม: {question_for_eval}",
    "sw_ff":"Tafadhali kwanza elekeza hoja, kisha ujibu swali na uweke jibu la mwisho kwenye \\ boxed{{}} mwishoni.\nSwali:{question_for_eval}"
}

according_prompt = {
    "en_mc":"According to the question shown in the image, please directly answer the question and put the correct option letter into \\boxed{{}}, e.g., \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}}, at the end.",
    "en_ff":"According to the question shown in the image, please directly answer the question and put the final value into \\boxed{{}} at the end.",
    "zh_mc":"根据图片中的问题，请直接回答问题，并在末尾将正确的选项字母填入\\boxed{{}}中，例如\\boxed{{A}}、\\boxed{{B}}、\\boxed{{C}}、\\boxed{{D}}。",
    "ja_mc":"画像に表示されている質問に従って、質問に直接答え、最後に \\boxed{{}} に正しいオプション文字を入力してください。例: \\boxed{{A}}、\\boxed{{B}}、\\boxed{{C}}、\\boxed{{D}}。",
    "es_mc":"De acuerdo con la pregunta que se muestra en la imagen, responda directamente la pregunta y coloque la letra de la opción correcta en \\boxed{{}}, por ejemplo, \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}}, al final.",
    "ru_mc":"В соответствии с вопросом, показанным на изображении, пожалуйста, ответьте на него напрямую и вставьте правильный вариант ответа в \\boxed{{}}, например, \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}}, в конце.",
    "fr_mc":"Conformément à la question affichée dans l'image, veuillez répondre directement à la question et mettre la lettre d'option correcte dans \\boxed{{}}, par exemple, \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}}, à la fin.",
    "de_mc":"Beantworten Sie die Frage entsprechend der im Bild gezeigten Frage direkt und setzen Sie am Ende den richtigen Optionsbuchstaben in \\boxed{{}}, z. B. \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}}.",
    "th_mc":"ตามคำถามที่แสดงในภาพ โปรดตอบคำถามโดยตรงและใส่ตัวอักษรตัวเลือกที่ถูกต้องลงในช่อง \\boxed{{}} เช่น \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}} ",
    "sw_mc":"Kulingana na swali lililoonyeshwa kwenye picha, tafadhali jibu swali moja kwa moja na uweke herufi ya chaguo sahihi kwenye \\ boxed{{}}, k.m., \\ boxed{{A}}, \\ boxed{{B}}, \\ boxed{{C}}, \\boxed{{D}}, mwishoni.",
    "zh_ff":"根据图片中的问题，请直接回答问题，并在最后的 \\boxed{{}} 中填入正确的值。",
    "ja_ff":"画像に表示されている質問に従って、質問に直接答え、最後に \\boxed{{}} に正しい値を入力してください。",
    "es_ff":"De acuerdo con la pregunta que se muestra en la imagen, responda directamente la pregunta y coloque el valor correcto en \\boxed{{}} al final.",
    "ru_ff":"В соответствии с вопросом, показанным на изображении, пожалуйста, ответьте на него напрямую и вставьте правильное значение в \\boxed{{}} в конце.",
    "fr_ff":"Conformément à la question affichée dans l'image, veuillez répondre directement à la question et mettre la valeur correcte dans \\boxed{{}} à la fin.",
    "de_ff":"Beantworten Sie die Frage entsprechend der im Bild angezeigten Frage direkt und geben Sie am Ende den richtigen Wert in \\boxed{{}} ein.",
    "th_ff":"ตามคำถามที่แสดงในภาพ โปรดตอบคำถามโดยตรงและใส่ค่าที่ถูกต้องลงใน \\boxed{{}} ที่ท้ายคำถาม",
    "sw_ff":"Kulingana na swali lililoonyeshwa kwenye picha, tafadhali jibu swali moja kwa moja na uweke jibu la mwisho kwenye \\boxed{{}} mwishoni."
}

forcing_raw = {
    "en": "Use English to think and answer.",
    "zh": "使用中文进行思考和回答。",
    "ar": "استخدم العربية للتفكير والإجابة.",
    "bn": "বাংলা ব্যবহার করে চিন্তা এবং উত্তর দিন।",
    "de": "Verwende Deutsch, um zu denken und zu antworten.",
    "es": "Usa español para pensar y responder.",
    "fr": "Utilisez le français pour penser et répondre.",
    "id": "Gunakan bahasa Indonesia untuk berpikir dan menjawab.",
    "it": "Usa italiano per pensare e rispondere.",
    "ja": "日本語を使って考え、回答してください。",
    "ko": "한국어로 생각하고 답변하세요.",
    "ms": "Gunakan bahasa Melayu untuk berfikir dan menjawab.",
    "pt": "Use português para pensar e responder.",
    "ru": "Используйте русский язык для размышлений и ответов.",
    "sw": "Tumia Kiswahili kufikiri na kujibu.",
    "te": "తెలుగును ఉపయోగించి ఆలోచించి సమాధానం ఇవ్వండి.",
    "th": "ใช้ภาษาไทยในการคิดและตอบคำถาม.",
    "vi": "Sử dụng tiếng Việt để suy nghĩ và trả lời.",
}