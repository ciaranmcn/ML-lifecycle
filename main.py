from generator import generate_response
from logger import log_result

prompt = input("Enter your prompt:")

result = generate_response(prompt)
log_result(result)

print("\n--- Response ---\n")
print(result["response"])
print("\n----------------\n")

feedback = input("Was this response good? (y/n): ").strip()
result["feedback"] = "good" if feedback == "y" else "bad"
