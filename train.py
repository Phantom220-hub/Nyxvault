"""Train and report NYXVAULT's local learning systems."""
import json
from pathlib import Path

from nyxvault import train
from neural_brain import NeuralBrain, train_neural

if __name__ == "__main__":
    retrieval = train()
    neural = train_neural()
    brain = NeuralBrain()
    sample = brain.generate("Вопрос: Что такое Python?\nОтвет:", max_new=100, temperature=0.35, top_k=1)
    report = {
        "retrieval_examples": retrieval["examples"],
        "retrieval_vocabulary": retrieval["vocabulary"],
        "neural_architecture": neural["architecture"],
        "neural_characters": neural["characters"],
        "neural_vocabulary": neural["vocabulary"],
        "neural_epochs": neural["epochs"],
        "neural_loss": neural["loss"],
        "sample_generation": sample,
    }
    Path("data/training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("NYXVAULT training complete")
    for key, value in report.items():
        print(f"{key}: {value}")
