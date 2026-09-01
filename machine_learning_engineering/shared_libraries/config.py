"""Configuration for Machine Learning Engineering Agent."""

import dataclasses
import os


@dataclasses.dataclass
class DefaultConfig:
    """Default configuration (Test Profile)."""

    data_dir: str = os.path.normpath("./machine_learning_engineering/tasks")  # the directory path where the machine learning tasks and their data are stored.
    task_name: str = os.environ.get("MLE_TASK_NAME", "california-housing-prices")  # The name of the specific task to be loaded and processed.
    task_type: str = os.environ.get(
        "MLE_TASK_TYPE", "Tabular Regression"
    )  # The type of machine learning problem.
    lower: bool = os.environ.get("MLE_LOWER", "True").lower() in (
        "true",
        "1",
        "yes",
    )  # True if a lower value of the metric is better.
    workspace_dir: str = os.path.normpath(
        os.environ.get(
            "MLE_WORKSPACE_DIR", "./machine_learning_engineering/workspace"
        )
    )  # Directory for intermediate outputs, results, logs (override per A/B variant via MLE_WORKSPACE_DIR; pure output, input is copied fresh from data_dir each run).
    agent_model: str = os.environ.get(
        "ROOT_AGENT_MODEL", "openai-gpt-oss-120b"
    )  # Name the LLM model to be used by the agent (GWDG Academic Cloud).
    api_base: str = os.environ.get(
        "OPENAI_API_BASE", "https://chat-ai.academiccloud.de/v1"
    )  # OpenAI-compatible API base URL.
    api_key: str = os.environ.get("OPENAI_API_KEY", "")  # API key for the backend.
    task_description: str = ""  # The detailed description of the task.
    task_summary: str = ""  # The concise summary of the task.
    start_time: float = 0.0  # Timestamp indicating the start time of the task. Typically represented in seconds since the epoch.
    seed: int = 42  # The random seed value used to ensure reproducibility of experiments.
    exec_timeout: int = (
        300  # The maximum time in seconds allowed to complete the task.
    )
    llm_timeout: int = (
        300  # The maximum time in seconds for a single LLM API request.
    )
    num_solutions: int = 1  # The number of different solutions to generate or attempt for the given task.
    num_model_candidates: int = 1  # The number of different model architectures or hyperparameter sets to consider as candidates.
    max_retry: int = (
        5  # The maximum number of times to retry a failed operation.
    )
    max_debug_round: int = 2  # The maximum number of iterations or rounds allowed for the debugging step.
    max_rollback_round: int = 2  # The maximum number of times the system can rollback to a previous state, in case of errors or poor performance.
    inner_loop_round: int = 1  # The number of iterations or rounds to be executed within an inner loop of the system.
    outer_loop_round: int = 1  # The number of iterations or rounds to be executed within the outer loop, which might encompass multiple inner loops.
    ensemble_loop_round: int = 1  # The number of rounds or iterations dedicated to ensembling, combining multiple models or solutions.
    num_top_plans: int = 2  # The number of highest-scoring plans or strategies to select or retain.
    use_data_leakage_checker: bool = False  # Enable (`True`) or disable (`False`) a check for data leakage in the machine learning pipeline.
    use_data_usage_checker: bool = False  # Enable (`True`) or disable (`False`) a check for how data is being used, potentially for compliance or best practices.
    use_rag: bool = False  # Set to False as RAG is deprecated. When False, all references to query_skrub_documentation are stripped from prompts.
    use_xai_correction: bool = os.environ.get(
        "USE_XAI_CORRECTION", "True"
    ).lower() in ("true", "1", "yes")  # Enable or disable the in-between XAI Correction agent
    use_xai_refinement: bool = os.environ.get(
        "USE_XAI_REFINEMENT", "False"
    ).lower() in ("true", "1", "yes")  # Enable or disable the inner refinement loop XAI Refinement agent
    xai_max_concentration: float = 0.80  # Max concentration of absolute feature attribution allowed for a single feature.
    xai_allow_static_fallback: bool = os.environ.get(
        "XAI_ALLOW_STATIC_FALLBACK", ""
    ).lower() in (
        "true",
        "1",
        "yes",
    )  # If False, dynamic XAI audit failure yields FAIL instead of LLM code review.


def get_config() -> DefaultConfig:
    """Returns the configuration based on the MLE_PROFILE environment variable."""
    config = DefaultConfig()
    profile = os.environ.get("MLE_PROFILE", "google").lower()
        
    if profile == "google":
        config.exec_timeout = 600
        config.num_solutions = 2
        config.num_model_candidates = 2
        config.max_retry = 5
        config.max_debug_round = 3
        
    return config


CONFIG = get_config()
