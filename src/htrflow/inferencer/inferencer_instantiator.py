#!/usr/bin/env python3
from datasets import load_dataset, Image, DatasetDict
from typing import Type, TypeVar
from base_inferencer import BaseInferencer

inferencer = TypeVar("inferencer", bound=BaseInferencer)


# The inferencer container, which takes hte instantiated inferencer of your choice, a huggingface dataset, and a dict of pipeline toggles.
class InferencerContainer:
    def __init__(
        self, dataset_input: str, inferencer: Type[inferencer], pipeline: dict = {"region": False, "line": False}
    ):
        self.dataset = load_dataset(dataset_input)
        self.inferencer = inferencer
        self.pipeline: dict = pipeline
