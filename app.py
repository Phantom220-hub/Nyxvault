import getpass
import hmac
import os
from pathlib import Path

from nyxvault import Nyxvault, train, load_examples
from neural_brain import NeuralBrain, train_neural

DATA = Path("data/knowledge.jsonl")
INDEX = Path("data/model_index.json")
MODEL = Path("data/neural_model.npz")


def teacher_mode() -> None:
    expected = os.environ.get("NYXVAULT_TEACHER_PASSWORD")
    if not expected:
        print("Teacher Mode заблокирован. Задай NYXVAULT_TEACHER_PASSWORD в локальном окружении.")
        return
    supplied = getpass.getpass("Пароль Teacher Mode: ")
    if not hmac.compare_digest(supplied, expected):
        print("Неверный пароль.")
        return

    print("\nTEACHER MODE — обучение NYXVAULT.")
    print("Команды: /train, /count, /help, /exit")
    print("Материал добавляется в формате: вопрос => правильный ответ")
    while True:
        try:
            line = input("Учитель> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход из Teacher Mode.")
            break
        if line == "/exit":
            break
        if line == "/help":
            print("/train — переобучить модели; /count — число примеров; /exit — выход.")
            print("Добавление: вопрос => ответ. Добавляй только проверенные факты.")
            continue
        if line == "/count":
            print("Примеров в учебной базе:", len(load_examples(DATA)))
            continue
        if line == "/train":
            try:
                stats = train(DATA, INDEX)
                neural = train_neural(DATA, MODEL)
                print(f"Индекс: {stats['examples']} примеров, словарь {stats['vocabulary']}.")
                print(f"RNN обучена: {neural['characters']} символов, {neural['epochs']} эпох, loss={neural['loss']}.")
            except (ValueError, OSError) as error:
                print("Не удалось обучить модель:", error)
            continue
        if "=>" not in line:
            print("Формат: вопрос => ответ. Для помощи введи /help.")
            continue
        question, answer = (part.strip() for part in line.split("=>", 1))
        if not question or not answer:
            print("Нужны и вопрос, и ответ.")
            continue
        if len(question) > 1000 or len(answer) > 4000:
            print("Слишком длинный пример: максимум 1000 символов в вопросе и 4000 в ответе.")
            continue
        bot = Nyxvault(DATA, INDEX)
        bot.teach(question, answer)
        print("Пример сохранён локально. Выполни /train, чтобы обновить модели.")


def chat_mode() -> None:
    bot = Nyxvault(DATA, INDEX)
    neural = NeuralBrain(MODEL) if MODEL.exists() else None
    history = []
    print("\nCHAT MODE — знания не изменяются. /clear — очистить контекст, /exit — выход.")
    while True:
        try:
            question = input("Ты> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход из Chat Mode.")
            break
        if question == "/exit":
            break
        if question == "/clear":
            history.clear()
            print("Контекст текущего разговора очищен.")
            continue
        if not question:
            continue

        answer = bot.answer(question)
        if answer.startswith("Я пока не знаю") and neural is not None:
            recent = history[-4:]
            context = "".join(f"Вопрос: {q}\nОтвет: {a}\n" for q, a in recent)
            prompt = context + f"Вопрос: {question}\nОтвет:"
            generated = neural.generate(prompt, max_new=160, temperature=0.7, top_k=6)
            if generated and neural.looks_plausible(generated, DATA):
                answer = "[Экспериментальная RNN-генерация] " + generated
        print("NYXVAULT>", answer)
        history.append((question, answer))
        history = history[-4:]


def main() -> None:
    print("NYXVAULT")
    print("Создатель: Nyxvault Studio")
    print("Владелец: Акрамзод Анас")
    print("\n1. Chat Mode — общение с временным контекстом")
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
