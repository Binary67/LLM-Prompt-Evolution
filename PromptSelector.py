import random

class PromptPool:
    """Simple container for prompts and their accuracy."""

    def __init__(self):
        self.Prompts = []

    def AddPrompt(self, Prompt: str, Accuracy: float) -> None:
        self.Prompts.append({"Prompt": Prompt, "Accuracy": Accuracy})

    def GetBestPrompt(self) -> str:
        if not self.Prompts:
            raise ValueError("Prompt pool is empty")
        Best = max(self.Prompts, key=lambda Item: Item["Accuracy"])
        return Best["Prompt"]

    def GetRandomPrompt(self) -> str:
        if not self.Prompts:
            raise ValueError("Prompt pool is empty")
        return random.choice(self.Prompts)["Prompt"]


def SelectPrompt(PromptPool: PromptPool, Epsilon: float) -> str:
    """Select a prompt using epsilon-greedy strategy."""
    if random.random() < Epsilon:
        return PromptPool.GetRandomPrompt()
    return PromptPool.GetBestPrompt()
