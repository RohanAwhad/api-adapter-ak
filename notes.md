System Prompt: "You are an adapter that checks an API model's arithmetic answer. If the API answer is correct, respond with \\boxed{CORRECT}. If wrong or missing, compute the correct answer and respond with \\boxed{answer}. Use the symbol definitions to evaluate custom expressions. Be concise."

User Message:
'The symbols θ, α, γ, β each represent one of the four basic arithmetic operations (+, -, ×, ÷). Each symbol maps to exactly one operation. Standard operator precedence (BODMAS) applies.\n\nExamples:\nExpression: 3 θ 4 | API answer: 7 → \\boxed{CORRECT}\nExpression: 10 α 3 | API answer: 5 → \\boxed{7}\nExpression: 2 γ 6 | API answer: none → \\boxed{12}\n\nExpression: 25 - 14 | API answer: 11 →\n/no_think'


Experiments:

1. Directly using the trained model without few-shot examples.
2. Using the trained model with few-shot examples.
3. Using the trained model with few-shot examples but shuffling the symbol-to-ops mapping.
4. Using the untrained model with few-shot examples.
5. Using the untrained model with few-shot examples but shuffling the symbol-to-ops mapping.