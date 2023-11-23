from collections import OrderedDict
from typing import Type

import torch
import yaml

from htrflow.models.openmmlab_models import OpenmmlabModel
from inferencer.inferencer_instantiator import InferencerContainer


class MultiModelManager:
    def __init__(self, yaml_file_path):
        self.yaml_file_path = yaml_file_path
        self.models = OrderedDict()  # Holds the loaded models

    def load_models(self):
        with open(self.yaml_file_path, "r") as file:
            model_definitions = yaml.safe_load(file)
            for model_def in model_definitions:
                model_id = model_def.get("model_id")
                cache_dir = model_def.get("cache_dir")
                device = model_def.get("device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                model_manager = OpenmmlabModel.from_pretrained(model_id, cache_dir, device)
                self.models[model_id] = model_manager


# This is the new pipeline class, currently in skeleton/outline form.
class ProcessingPipeline:
    def __init__(self, inferencer):
        self.inferencer = inferencer
        self.pipeline = inferencer.pipeline

    def dispatch_processor(self):
        """Dispatcher for all the processing functions, using tail recursion. First, check if any pipeline options are set. Then moves through the pipeline in order, dispatching to approopriate functions and then switching that value to False. This repeats until every step in the dict is false and processing is complete."""
        if True in self.pipeline.items():
            if self.pipeline["region"] is True:
                self.pipeline["region"] = False
                self.region_processor()
            elif self.pipeline["line"] is True:
                self.pipeline["line"] = False
                self.line_processor()

            self.dispatch_processor()

    def region_processor(self):
        pass

    def line_processor(self):
        pass
