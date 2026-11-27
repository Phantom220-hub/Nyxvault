"""Train both NYXVAULT learning systems."""
from nyxvault import train
from neural_brain import train_neural

if __name__ == "__main__":
    retrieval = train()
    neural = train_neural()
    print("NYXVAULT training complete")
    print("Retrieval examples:", retrieval["examples"])
    print("Neural characters:", neural["characters"])
    print("Neural vocabulary:", neural["vocabulary"])
    print("Neural epochs:", neural["epochs"])
    print("Neural loss:", neural["loss"])
