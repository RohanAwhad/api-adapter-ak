System Prompt: "You are an adapter that checks an API model's arithmetic answer. If the API answer is correct, respond with \\boxed{CORRECT}. If wrong or missing, compute the correct answer and respond with \\boxed{answer}. Use the symbol definitions to evaluate custom expressions. Be concise."

User Message:
'The symbols θ, α, γ, β each represent one of the four basic arithmetic operations (+, -, ×, ÷). Each symbol maps to exactly one operation. Standard operator precedence (BODMAS) applies.\n\nExamples:\nExpression: 3 θ 4 | API answer: 7 → \\boxed{CORRECT}\nExpression: 10 α 3 | API answer: 5 → \\boxed{7}\nExpression: 2 γ 6 | API answer: none → \\boxed{12}\n\nExpression: 25 - 14 | API answer: 11 →\n/no_think'


Experiments:

1. Directly using the trained model without few-shot examples.
    ```
    Overall:  210/400 = 52.5%
    Custom:   18/200 = 9.0%
    Standard: 192/200 = 96.0%
    ```
2. Using the trained model with few-shot examples.
    ```
    Overall:  382/400 = 95.5%
    Custom:   188/200 = 94.0%
    Standard: 194/200 = 97.0%
    ```
3. Using the trained model with few-shot examples but shuffling the symbol-to-ops mapping.
    ```
    Overall:  363/400 = 90.8%
    Custom:   169/200 = 84.5%
    Standard: 194/200 = 97.0%
    ```


> Akash did mention that during training we are not giving few-shot examples, but that not true, as we see in the user message we are clearly giving examples.

- So in next experiment we will add a flag that says whether we should add few-shot examples to user message or not.


New User Message:
'The symbols θ, α, γ, β each represent one of the four basic arithmetic operations (+, -, ×, ÷). Each symbol maps to exactly one operation. Standard operator precedence (BODMAS) applies.\n\nExpression: 25 - 14 | API answer: 11 →\n/no_think'