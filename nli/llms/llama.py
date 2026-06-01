import ollama

MODEL = "llama3.1:8b"

OPEN_SYSTEM = "You are a knowledgeable assistant. Give detailed, thorough answers."

LOGIC_SYSTEM = """You are a logical reasoning assistant.
Rules:
  1. If all A are B, and all B are C, then all A are C.
  2. If all A are B, then all B are not necessarily A.
  3. If all C are A and all C are B then some A are B.
  4. Do not use outside knowledge.

Examples:
  Q: All cats are mammals. All mammals are animals. Are all cats animals?
  A: Yes. By Rule 1, since all cats are mammals and all mammals are animals,
     all cats are animals.
"""

SINGLE_WORD_SYSTEM = (
    "Answer with a SINGLE WORD only. "
    "Do not add punctuation, explanation, or any other text."
)


def ask(system_prompt, user_prompt):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    results = {}

    results["open_1"] = ask(
        OPEN_SYSTEM,
        "Who is the most influencial mathematician and why?",
    )

    results["open_2"] = ask(
        OPEN_SYSTEM,
        "Summarize the background information regarding Homer's Odessy so a reader can know all the revelant information before reading.",
    )

    results["open_3"] = ask(
        OPEN_SYSTEM,
        "What is the next big breakthrough in quantum computing currently being worked on?",
    )

    results["reason_1"] = ask(
        LOGIC_SYSTEM,
        "All roses are flowers. All flowers are plants. Are all roses plants? Explain.",
    )

    results["reason_2"] = ask(
        LOGIC_SYSTEM,
        "If all squares are rectangles, are all rectangles squares? Explain.",
    )

    results["reason_3"] = ask(
        LOGIC_SYSTEM,
        "If all dogs are mammals and all dogs are 4-legge, then are any bears mammals?",
    )

    results["single_1"] = ask(
        SINGLE_WORD_SYSTEM,
        "True or False: 2+2 = 4.",
    )

    results["single_2"] = ask(
        SINGLE_WORD_SYSTEM,
        "True or False: The moon landing was fake.",
    )

    results["single_3"] = ask(
        SINGLE_WORD_SYSTEM,
        "Answer in one number only, How many letters are in this prompt?",
    )

    for key, answer in results.items():
        print(f"\n{'=' * 60}")
        print(f"[{key.upper()}]")
        print(answer)
