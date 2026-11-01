import getpass
import os
from pathlib import Path

from nyxvault import KnowledgeBase

DATA = Path("data/knowledge.jsonl")


def teacher_mode(kb: KnowledgeBase) -> None:
    expected = os.environ.get("NYXVAULT_TEACHER_PASSWORD")
    if not expected:
        print("Teacher mode is locked: set NYXVAULT_TEACHER_PASSWORD in your local environment first.")
        return
    supplied = getpass.getpass("Teacher password: ")
    import hmac
    if not hmac.compare_digest(supplied, expected):
        print("Wrong password.")
        return

    print("\nTEACHER MODE — type /exit to leave.")
    print("Add knowledge as: question => answer")
    while True:
        line = input("Teach> ").strip()
        if line == "/exit":
            break
        if "=>" not in line:
            print("Format: question => answer")
            continue
        question, answer = (part.strip() for part in line.split("=>", 1))
        if not question or not answer:
            print("Both question and answer are required.")
            continue
        kb.add(question, answer)
        print("Saved locally. Chat mode can use this example now.")


def chat_mode(kb: KnowledgeBase) -> None:
    print("\nCHAT MODE — read-only; type /exit to leave.")
    while True:
        question = input("You> ").strip()
        if question == "/exit":
            break
        print("NYXVAULT>", kb.answer(question))


def main() -> None:
    kb = KnowledgeBase(DATA)
    print("NYXVAULT — experimental from-scratch assistant")
    print("1. Chat mode (read-only)")
    print("2. Teacher mode (password required)")
    choice = input("Choose mode: ").strip()
    if choice == "1":
        chat_mode(kb)
    elif choice == "2":
        teacher_mode(kb)
    else:
        print("Choose 1 or 2.")


if __name__ == "__main__":
    main()
