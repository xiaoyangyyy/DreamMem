from __future__ import annotations

from .config import ExperimentConfig
from .dcoder_text import TextConceptEncoder
from .tasks import TaskGenerator


def make_task_generator(config: ExperimentConfig):
    if config.task_type == "concept":
        return TaskGenerator(config.data)
    if config.task_type == "relational":
        from .relational import RelationalTaskGenerator

        return RelationalTaskGenerator(config.data)
    if config.task_type == "operators":
        from .operators import OperatorTaskGenerator

        return OperatorTaskGenerator(config.data)
    if config.task_type == "large":
        from .large_tasks import LargeTaskGenerator

        return LargeTaskGenerator(config.data)
    if config.task_type == "natural":
        from .natural import NaturalTaskGenerator

        return NaturalTaskGenerator(config.data)
    if config.task_type == "advanced":
        from .advanced_operators import AdvancedTaskGenerator

        return AdvancedTaskGenerator(config.data)
    raise ValueError(f"Unknown task type: {config.task_type}")


def make_task_components(config: ExperimentConfig, tokenizer, mask_token_id: int):
    if config.task_type == "concept":
        return make_task_generator(config), TextConceptEncoder(tokenizer, config.data, mask_token_id)
    if config.task_type == "relational":
        from .relational import RelationalTextEncoder

        return make_task_generator(config), RelationalTextEncoder(tokenizer, config.data, mask_token_id)
    if config.task_type == "operators":
        from .operators import OperatorTextEncoder

        return make_task_generator(config), OperatorTextEncoder(tokenizer, config.data, mask_token_id)
    if config.task_type == "large":
        from .large_tasks import LargeTextEncoder

        return make_task_generator(config), LargeTextEncoder(tokenizer, config.data, mask_token_id)
    if config.task_type == "natural":
        from .natural import NaturalTextEncoder

        return make_task_generator(config), NaturalTextEncoder(tokenizer, config.data, mask_token_id)
    if config.task_type == "advanced":
        from .advanced_operators import AdvancedTextEncoder

        return make_task_generator(config), AdvancedTextEncoder(tokenizer, config.data, mask_token_id)
    raise ValueError(f"Unknown task type: {config.task_type}")
