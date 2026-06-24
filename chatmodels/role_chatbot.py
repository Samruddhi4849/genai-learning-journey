from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ─────────────────────────────────────────
# Initialize the model
# ─────────────────────────────────────────
model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.7
)

# ─────────────────────────────────────────
# PERSONA 1 — Strict Coding Mentor
# Never gives direct answers, only hints
# ─────────────────────────────────────────
strict_mentor_system = """
You are a strict coding mentor with 20 years of experience.
Your teaching philosophy:
- NEVER give direct answers or complete code solutions
- Always respond with hints, questions, and guided thinking
- Ask "What do you think happens when...?" type questions
- If student is stuck, give only the next small hint
- Use phrases like "Think about...", "What if you tried...", 
  "Have you considered..."
- Be firm but not rude
- Your goal is to make students think, not copy answers
"""

# ─────────────────────────────────────────
# PERSONA 2 — Friendly Tutor
# Warm, encouraging, gives full explanations
# ─────────────────────────────────────────
friendly_tutor_system = """
You are a friendly and encouraging tutor who loves teaching.
Your teaching philosophy:
- Always give clear, complete, and detailed explanations
- Use simple language and fun real-world analogies
- Be warm, positive and encouraging
- Celebrate when students ask good questions
- Use phrases like "Great question!", "Let me explain this simply"
- Give working code examples whenever helpful
- End responses with an encouraging note
"""

# ─────────────────────────────────────────
# PERSONA 3 — Sarcastic Senior Developer
# (Bonus fun persona to try!)
# ─────────────────────────────────────────
sarcastic_dev_system = """
You are a sarcastic senior developer with 15 years of experience
who has seen it all and is mildly tired of explaining basics.
Your style:
- Answer correctly but with light sarcasm and humor
- Make jokes about common beginner mistakes
- Reference how things were harder "back in the day"
- Still helpful underneath the sarcasm
- Use phrases like "Oh, this classic mistake...", 
  "Let me guess, you didn't read the docs?"
- Never be genuinely mean, just playfully sarcastic
"""

# ─────────────────────────────────────────
# Choose your persona here — change this
# to switch between personas!
# ─────────────────────────────────────────
personas = {
    "1": ("Strict Coding Mentor", strict_mentor_system),
    "2": ("Friendly Tutor", friendly_tutor_system),
    "3": ("Sarcastic Senior Developer", sarcastic_dev_system)
}

# ─────────────────────────────────────────
# Let user pick a persona
# ─────────────────────────────────────────
print("=" * 60)
print("       ROLE-BASED CHATBOT")
print("=" * 60)
print("\nChoose a persona:")
print("  1 — Strict Coding Mentor")
print("  2 — Friendly Tutor")
print("  3 — Sarcastic Senior Developer")
print()

choice = input("Enter 1, 2, or 3: ").strip()

if choice not in personas:
    print("Invalid choice, defaulting to Friendly Tutor")
    choice = "2"

persona_name, system_prompt = personas[choice]

print(f"\n✅ Persona selected: {persona_name}")
print("=" * 60)
print("Start chatting! Type 'quit' to exit, 'switch' to change persona")
print("=" * 60)
print()

# ─────────────────────────────────────────
# Conversation history — this is what makes
# it remember previous messages!
# ─────────────────────────────────────────
chat_history = [SystemMessage(content=system_prompt)]

# ─────────────────────────────────────────
# Main chat loop
# ─────────────────────────────────────────
while True:
    # Get user input
    user_input = input("You: ").strip()

    # Exit condition
    if user_input.lower() == "quit":
        print("\nBot: Goodbye! Keep coding! 👋")
        break

    # Switch persona mid conversation
    if user_input.lower() == "switch":
        print("\nChoose new persona:")
        print("  1 — Strict Coding Mentor")
        print("  2 — Friendly Tutor")
        print("  3 — Sarcastic Senior Developer")
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in personas:
            persona_name, system_prompt = personas[choice]
            # Reset history with new system prompt
            # but keep conversation context
            chat_history[0] = SystemMessage(content=system_prompt)
            print(f"✅ Switched to: {persona_name}\n")
        continue

    # Skip empty input
    if not user_input:
        continue

    # Add user message to history
    chat_history.append(HumanMessage(content=user_input))

    # Send full history to model
    response = model.invoke(chat_history)

    # Add bot response to history
    chat_history.append(AIMessage(content=response.content))

    # Print response
    print(f"\n{persona_name}: {response.content}\n")
    print("-" * 40)