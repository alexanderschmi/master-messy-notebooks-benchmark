import dspy
from pydantic import BaseModel, Field
from typing import Optional


class OutputSchema(BaseModel):
    """Output schema for the code generation task."""
    requirements: Optional[str] = Field(
        default="",
        #desc="The full content of the requirements.txt file, listing all dependencies needed to run the scripts.",
    )
    train: Optional[str] = Field(
        default="",
        #desc="The full python code for train.py containing the model training script.",
    )
    inference: Optional[str] = Field(
        default="",
        #desc="The full python code for inference.py containing the model inference script. It should include a function"
        #" inference(feature) that takes one feature of the data set as input and returns the model output.",
    )


class CodeGenSignature(dspy.Signature):
    """
    You are a software developer that works in ai research on NLP tasks. Create the code of the following 3 connected files
    based on a jupyter notebook provided later.
    1. A python file named train that includes the code (classes and functions) of the notebook, needed to train the same model depicted in the notebook and saves the same model artifacts.
    2. A python file named inference that includes the code (classes and functions), needed to run inference on the same model as either
    batch jobs or online given the specific task inside the notebook. It should have a function inference.inference(feature) -> int
    that takes one feature of the data set as input and returns the model output.
    3. A text file named requirements that includes all libraries need to run both python files mentioned before.
    A good output should be well documented, easily understandable and should fix all errors present inside the
    notebook if there are any.
    """

    notebook = dspy.InputField(desc="The Jupyter Notebook")
    train: Optional[str] = dspy.OutputField(default="", desc="Full python code for train.py")
    inference: Optional[str] = dspy.OutputField(default="", desc="Full python code for inference.py")
    requirements: Optional[str] = dspy.OutputField(default="", desc="Full requirements.txt")


def set_prompt(prompt_text: str):
    """Dynamically sets the prompt instruction for the dspy Signature."""
    CodeGenSignature.__doc__ = prompt_text
