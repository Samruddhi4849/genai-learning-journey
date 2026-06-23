from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI

# Initialize the model
model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.7
)

# ─────────────────────────────────────────
# The SAME question for all 3 styles
# ─────────────────────────────────────────
question = "Explain what machine learning is"

# ─────────────────────────────────────────
# STYLE 1 — Zero Shot
# Just ask directly. No examples. No hints.
# Model uses only its own training knowledge.
# ─────────────────────────────────────────
zero_shot_prompt = question

# ─────────────────────────────────────────
# STYLE 2 — Few Shot
# Give 2 examples of how you want answers
# formatted BEFORE asking your real question.
# Model learns the pattern from your examples.
# ─────────────────────────────────────────
few_shot_prompt = """
Here are some examples of how concepts are explained simply:

Question: Explain what the internet is
Answer: The internet is like a giant postal system for information. 
Instead of sending physical letters, computers send digital 
messages to each other through cables and wireless signals, 
allowing anyone to share information instantly worldwide.

Question: Explain what a database is
Answer: A database is like a super organized filing cabinet. 
Instead of paper files, it stores digital information in a 
structured way so you can quickly find, update, or delete 
any piece of information you need.

Now explain in the same simple style:
Question: Explain what machine learning is
Answer:
"""

# ─────────────────────────────────────────
# STYLE 3 — Chain of Thought
# Tell the model to think step by step
# before giving the final answer.
# Forces the model to reason, not just recall.
# ─────────────────────────────────────────
chain_of_thought_prompt = """
Explain what machine learning is.

Think through this step by step:
Step 1: Start with what traditional programming is
Step 2: Explain the problem with traditional programming
Step 3: Explain how machine learning solves that problem
Step 4: Give a real world example
Step 5: Give a simple one line summary at the end
"""

# ─────────────────────────────────────────
# Send all 3 prompts to the model
# ─────────────────────────────────────────
print("Sending prompts to model...\n")

response1 = model.invoke(zero_shot_prompt)
response2 = model.invoke(few_shot_prompt)
response3 = model.invoke(chain_of_thought_prompt)

# ─────────────────────────────────────────
# Print all 3 responses side by side
# ─────────────────────────────────────────

divider = "=" * 60

print(divider)
print("STYLE 1 — ZERO SHOT (Just asked directly)")
print(divider)
print(response1.content)

print("\n")
print(divider)
print("STYLE 2 — FEW SHOT (Given examples first)")
print(divider)
print(response2.content)

print("\n")
print(divider)
print("STYLE 3 — CHAIN OF THOUGHT (Step by step reasoning)")
print(divider)
print(response3.content)

print("\n")
print("=" * 60)
print("OBSERVATION: Notice how each style produces a different")
print("quality and format of response!")
print("=" * 60)