import getpass
import hmac
import os
from pathlib import Path

from nyxvault import Nyxvault, train
from neural_brain import NeuralBrain, train_neural

DATA = Path("data/knowledge.jsonl")
INDEX = Path("data/model_index.json")


def teacher_mode() -> None:
    expected = os.environ.get("NYXVAULT_TEACHER_PASSWORD")
    if not expected:
        print("Teacher Mode заблокирован. Сначала задай NYXVAULT_TEACHER_PASSWORD в локальном окружении.")
        return
    supplied = getpass.getpass("Пароль Teacher Mode: ")
    if not hmac.compare_digest(supplied, expected):
        print("Неверный пароль.")
        return

    print("\nTEACHER MODE — обучение NYXVAULT. Команды: /train, /count, /exit")
    print("Добавляй материал в формате: вопрос => правильный ответ")
    while True:
        line = input("Учитель> ").strip()
        if line == "/exit":
            break
        if line == "/count":
            from nyxvault import load_examples
            print("Примеров в учебной базе:", len(load_examples(DATA)))
            continue
        if line == "/train":
            stats = train(DATA, INDEX)
            neural = train_neural(DATA)
            print(f"Индекс обновлён: {stats['examples']} примеров.")
            print(f"Нейросеть обучена: {neural['characters']} символов; loss={neural['loss']}.")
            continue
        if "=>" not in line:
            print("Формат: вопрос => ответ")
            continue
        question, answer = (part.strip() for part in line.split("=>", 1))
        if not question or not answer:
            print("Нужны и вопрос, и ответ.")
            continue
        bot = Nyxvault(DATA, INDEX)
        bot.teach(question, answer)
        print("Пример сохранён. Выполни /train, чтобы обновить модель.")


def chat_mode() -> None:
    bot = Nyxvault(DATA, INDEX)
    neural = NeuralBrain() if Path("data/neural_model.npz").exists() else None
    print("\nCHAT MODE — только чтение, обучение отключено. /exit — выход.")
    while True:
        question = input("Ты> ").strip()
        if question == "/exit":
            break
        answer = bot.answer(question)
        if answer.startswith("Я пока не знаю") and neural is not None:
            generated = neural.generate("Вопрос: " + question + "\nОтвет:", max_new=160)
            if generated:
                answer = "[Экспериментальная генерация] " + generated
        print("NYXVAULT>", answer)


def main() -> None:
    print("NYXVAULT")
    print("Создатель: Nyxvault Studio")
    print("Владелец: Акрамзод Анас")
    print("\n1. Chat Mode — обычное общение, без обучения")
    print("2. Teacher Mode — обучение под паролем")
    choice = input("Выбери режим: ").strip()
    if choice == "1":
        chat_mode()
    elif choice == "2":
        teacher_mode()
    else:
        print("Выбери 1 или 2.")


if __name__ == "__main__":
    main()
