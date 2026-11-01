"""Train/rebuild NYXVAULT's local retrieval model from its approved corpus."""
from nyxvault import train

if __name__ == "__main__":
    result = train()
    print("NYXVAULT training complete")
    print("Examples:", result["examples"])
    print("Vocabulary:", result["vocabulary"])
    print("Index:", result["index"])
