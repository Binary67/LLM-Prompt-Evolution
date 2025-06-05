import asyncio
from DataGeneration import GenerateData
from RunEvolution import RunEvolution


async def Main():
    TrainingData, ValidationData = GenerateData()
    await RunEvolution(TrainingData, ValidationData)


if __name__ == "__main__":
    asyncio.run(Main())
