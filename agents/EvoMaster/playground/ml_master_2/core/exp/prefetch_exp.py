import logging
from typing import Any
from evomaster.core.exp import BaseExp
from evomaster.utils.types import TaskInstance
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
from ..utils.code import read_code,save_code_to_file
import uuid
import os
from evomaster.agent import BaseAgent
import json

class PrefetchExp(BaseExp):
    """Experiment for retrieving relevant knowledge from the wisdom database via RAG.

    Uses embedding-based retrieval to find data knowledge and model knowledge
    from past experiments that are relevant to the current task.
    """

    def __init__(self, prefetch_agent, config, exp_name):
        super().__init__(prefetch_agent, config)
        self.prefetch_agent = prefetch_agent
        self.uid = uuid.uuid4()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workspace_path = self.prefetch_agent.session.config.workspace_path
        self._exp_name = exp_name

    @property
    def exp_name(self) -> str:
        """Return the experiment stage name."""
        return self._exp_name

    def _resolve_embedding_credentials(self) -> tuple[str | None, str | None, str | None]:
        api_key = os.environ.get("OPENAI_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL")

        if not api_key:
            return None, None, "No OpenAI embedding API key configured; skipping RAG retrieval."

        if api_key.startswith("sk-or-v1") and not base_url:
            return (
                None,
                None,
                "OPENAI_API_KEY appears to be an OpenRouter key without an embedding base URL; skipping RAG retrieval.",
            )

        return api_key, base_url, None

    def run(self, task_description: str, vec_dir: str, nodes_data: str, model: str, task_id: str = "detect-insults-in-social-commentary") -> dict:
        """Execute the prefetch experiment to retrieve relevant knowledge via RAG.

        Uses the prefetch agent to generate a task descriptor, then performs
        embedding-based retrieval to find relevant data and model knowledge.

        Args:
            task_description: Natural language description of the ML task.
            vec_dir: Directory containing embedding vectors for RAG.
            nodes_data: Path to the nodes data file (JSON).
            model: Embedding model name to use.
            task_id: Unique task identifier.

        Returns:
            Tuple of (data_knowledge, model_knowledge, prefetch_descriptor).
        """
        self.logger.info("Starting prefetch task execution")
        self.logger.info(f"Task: {task_description}")

        data_knowledge = "NO DATA KNOWLEDGE this time"
        model_knowledge = "NO MODEL KNOWLEDGE this time"
        prefetch_descriptor = task_description

        # Early return for debugging
        # return data_knowledge, model_knowledge, prefetch_descriptor

        if self.prefetch_agent:
            self.logger.info("=" * 60)
            self.logger.info("Step 1: Prefetch Agent analyzing task...")
            self.logger.info("=" * 60)
            BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=1)
            
            prefetch_original_format_kwargs = self.prefetch_agent._prompt_format_kwargs.copy()
            self.prefetch_agent._prompt_format_kwargs.update({
                'task_description': task_description,
            })
            prefetch_task = TaskInstance(
                task_id=f"{task_id}_prefetch",
                task_type="prefetch",
                task_description=task_description,
                input_data={},
            )
            prefetch_trajectory = self.prefetch_agent.run(prefetch_task)
            prefetch_descriptor = self._extract_agent_response(prefetch_trajectory)
            self.prefetch_agent._prompt_format_kwargs = prefetch_original_format_kwargs
            self.logger.info(f"Prefetch descriptor: {prefetch_descriptor}")
            ### Execute RAG retrieval: use_skill run_script to call search.py
            #### For debugging, uncomment the hardcoded prefetch_descriptor below
            # prefetch_descriptor = """This competition involves single class classification to detect insulting comments in social commentary datasets with timestamps and unicode escaped text. The goal is creating a generalizable classifier for near real time identification of insults directed at conversation participants excluding non participants or standalone profanity. Input includes a label column denoting neutral or insulting status followed by time attributes formatted as YYYYMMDDhhmmssZ and text fields. Output requires probability scores from 0 to 1 indicating insult likelihood submitted in the first column of the submission file. Evaluation uses the Area under the Receiver Operating Curve metric penalizing high probability incorrect predictions. Final standings depend on performance against an unpublished verification set released late in the timeline. Competitors must lock self contained code for up to five final models prior to verification set release and generate solutions using that locked code. Dataset characteristics include less than one percent label noise with strong tendencies toward overfitting. Individual participation is mandatory for recruiting opportunities involving code review for top entries."""
            embedding_api_key, embedding_base_url, skip_reason = self._resolve_embedding_credentials()
            if skip_reason:
                self.logger.info(skip_reason)
                self.logger.info(f"Data knowledge: {data_knowledge}")
                self.logger.info(f"Model knowledge: {model_knowledge}")
                return data_knowledge, model_knowledge, prefetch_descriptor
            query_escaped = json.dumps(prefetch_descriptor)
            script_args = (
                f"--vec_dir {vec_dir} --query {query_escaped} --nodes_data {nodes_data} --output json --embedding_type openai --embedding_dimensions 3072 "
                f"--top_k 1 --threshold 0.7 --output json --model {model}"
            )
            if embedding_api_key:
                script_args += f" --embedding_api_key {json.dumps(embedding_api_key)}"
            if embedding_base_url:
                script_args += f" --embedding_base_url {json.dumps(embedding_base_url)}"
            tool_call_obj = ChatCompletionMessageToolCall(
                id="call_123",
                type="function",
                function=Function(
                    name="use_skill",
                    arguments=json.dumps({
                        "skill_name": "rag",
                        "action": "run_script",
                        "script_name": "search.py",
                        "script_args": script_args,
                    }),
                )
            )
            observation, info = self.prefetch_agent._execute_tool(tool_call_obj)

            # Parse the output JSON
            try:
                output_str = observation
                if "Script output:\n" in output_str:
                    output_str = output_str.split("Script output:\n", 1)[1]
                if "\n\nStderr:" in output_str:
                    output_str = output_str.split("\n\nStderr:")[0]
                rag_output = json.loads(output_str.strip())
                results = rag_output.get("results", [])
                if results:
                    first_result = results[0]
                    content = first_result.get("content", {})
                    data_knowledge = content.get("data_knowledge", "NO DATA KNOWLEDGE this time")
                    model_knowledge = content.get("model_knowledge", "NO MODEL KNOWLEDGE this time")
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Failed to parse RAG output: {e}")
                data_knowledge = "NO DATA KNOWLEDGE this time"
                model_knowledge = "NO MODEL KNOWLEDGE this time"
        self.logger.info(f"Data knowledge: {data_knowledge}")
        self.logger.info(f"Model knowledge: {model_knowledge}")

        return data_knowledge, model_knowledge, prefetch_descriptor
