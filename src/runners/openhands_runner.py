import asyncio
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional
from src.core.file_io import generate_files_from_answer
from src.runners.base import BaseRunner

from openhands.sdk import Agent, Conversation, LLM, Tool
from openhands.sdk.workspace import LocalWorkspace
from openhands.sdk.event.llm_convertible import MessageEvent
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

logger = logging.getLogger(__name__)

class MockAnswer:
    def __init__(self, requirements: str, train: str, inference: str):
        self.requirements = requirements
        self.train = train
        self.inference = inference


def extract_code_from_text(content: str, filename: str) -> Optional[str]:
    """Simple extraction of code blocks from markdown that might contain the filename."""
    # Normalize content in case it's escaped from a repr() or similar
    content_norm = content.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
    
    # Look for code blocks
    blocks = re.findall(r'```(?:python|text|bash)?\n(.*?)\n```', content_norm, re.DOTALL)
    
    # Strategy 1: Block contains the filename as a comment near the top
    for block in blocks:
        if any(line.strip().startswith(('#', '//')) and filename in line for line in block.splitlines()[:5]):
            return block
            
    # Strategy 2: Filename appears right before a code block
    pattern = rf'(?:^|\n)[#* \t]*{re.escape(filename)}.*?\n```(?:python|text|bash)?\n(.*?)\n```'
    match = re.search(pattern, content_norm, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
        
    return None


def run_openhands_sync(
    model: str,
    provider: str,
    api_key: str,
    temperature: float,
    notebook_content: str,
    prompt: str,
    api_base: Optional[str] = None
) -> tuple[MockAnswer, dict, float]:
    """Run OpenHands programmatically for a single notebook in a temporary workspace."""
    
    # We create a temporary directory for the workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Write the notebook content to a file inside the workspace
        nb_path = temp_path / "notebook.ipynb"
        nb_path.write_text(notebook_content, encoding="utf-8")

        (temp_path / "notebook.md").write_text(notebook_content, encoding="utf-8")
        
        # Configure the LLM
        if api_base:
            llm = LLM(
                model=provider + "/" + model, 
                api_key=api_key,
                base_url=api_base,
                temperature=temperature,
            )
        else:
            llm = LLM(model=provider + "/" + model, api_key=api_key, temperature=temperature, native_tool_calling=False)

        # Initialize the OpenHands agent
        agent = Agent(llm=llm, tools=[Tool(name=TerminalTool.name),Tool(name=FileEditorTool.name),Tool(name=TaskTrackerTool.name)])

        # Initialize conversation
        workspace = LocalWorkspace(working_dir=temp_dir)
        conversation = Conversation(
            agent=agent,
            max_iteration_per_run=30,
            workspace=workspace
        )
        
        logger.info("Initializing OpenHands conversation...")
        
        # Formulate the message to start the task
        task_msg = (
            f"{prompt}\n\n"
            f"The notebook content has been saved in this workspace as '{temp_dir}/notebook.md'.\n"
            f"Please create the files train.py, inference.py, and requirements.txt "
            f"in this SAME directory based on the instructions. Do not ask for user confirmation. Just write the files!"
        )
        
        conversation.send_message(task_msg)
        
        logger.info("Running OpenHands agent...")
        try:
            start_time = time.time()
            conversation.run()
            end_time = time.time()
            total_time = end_time - start_time
        except Exception as e:
            logger.warning(f"Error during conversation.run(): {e}")

        total_dict = conversation.conversation_stats.get_combined_metrics().model_dump()
        
        logger.info("OpenHands completed. Gathering output files from workspace...")
        
        # Read the generated files from disk
        reqs = (temp_path / "requirements.txt").read_text(encoding="utf-8") if (temp_path / "requirements.txt").exists() else ""
        train = (temp_path / "train.py").read_text(encoding="utf-8") if (temp_path / "train.py").exists() else ""
        inference = (temp_path / "inference.py").read_text(encoding="utf-8") if (temp_path / "inference.py").exists() else ""
        
        # Fallback: Parse from history if disk files are empty
        if not train or not inference or not reqs:
            logger.info("Some files missing on disk, checking conversation history...")
            combined_history = ""
            for event in conversation._state.events:
                source = getattr(event, 'source', 'unknown')
                
                # Check messages and actions from the agent/assistant
                if source in ["assistant", "agent"]:
                    # Try various ways to get text content
                    history_text = ""
                    if hasattr(event, "message") and event.message:
                        history_text = str(event.message)
                    elif hasattr(event, "thought") and event.thought:
                        history_text = str(event.thought)
                    elif hasattr(event, "content") and event.content:
                        history_text = str(event.content)
                    else:
                        # Final fallback: string representation
                        history_text = str(event)
                        
                    combined_history += "\n" + history_text
            
            if combined_history:
                if not reqs:
                    reqs = extract_code_from_text(combined_history, "requirements.txt") or ""
                if not train:
                    train = extract_code_from_text(combined_history, "train.py") or ""
                if not inference:
                    inference = extract_code_from_text(combined_history, "inference.py") or ""
            if not reqs or not train or not inference:
                raise ValueError("Failed to extract code from conversation history")

        return MockAnswer(requirements=reqs, train=train, inference=inference), total_dict, total_time


class OpenHandsRunner(BaseRunner):
    RUNNER_NAME = "agentic"

    def run(
        self,
        params: dict,
        notebook: tuple[str, str],
        temperature: float,
        save_history: bool,
        output_dir: str,
        complexity: int,
        run: int,
    ) -> None:
        model = params.get("model", params.get("name", "unknown_model"))
        api_key = params.get("api_key", "")
        api_base = params.get("api_base")
        prompt = params.get("prompt", "")
        provider = params.get("provider", "openai")

        nb_id, nb_content = notebook
        logger.info(f"[OpenHands - {model}] Processing notebook {nb_id}")

        try:
            answer, metrics, total_time = run_openhands_sync(
                model=model,
                provider=provider,
                api_key=api_key,
                temperature=temperature,
                notebook_content=nb_content,
                prompt=prompt,
                api_base=api_base,
            )
            logger.info(f"[OpenHands - {model}] Successfully processed notebook {nb_id}")
        except Exception as e:
            logger.error(f"[OpenHands - {model}] Failed processing notebook {nb_id}: {e}")
            return

        logger.info(f"[OpenHands - {model}] Finished processing notebook {nb_id}")
        generate_files_from_answer((nb_id, answer, None), output_dir, model, complexity, runner=self.RUNNER_NAME, run=run, usage=metrics, time_taken=total_time)
