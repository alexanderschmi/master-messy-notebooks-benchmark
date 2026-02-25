import dspy
from pydantic import BaseModel, Field
from typing import Optional


class OutputSchema(BaseModel):
    requirements: Optional[str] = Field(
        default="",
        desc="The requirements.txt")
    train: Optional[str] = Field(
        default="",
        desc="The train.py")
    inference: Optional[str] = Field(
        default="",
        desc="The inference.py, it should include a function inference that takes one "
             "feature of the data set as input and returns the specific output (prediction, "
             "classification, etc.)")


class CodeGenSignature(dspy.Signature):
    """
    You are a software developer that works in ai research on NLP tasks. Create the following 3 connected files
    based on a jupyter notebook provided later.
    1. A python file named train.py that includes the script needed to train the same model depicted in the notebook.
    2. A python file named inference.py that includes the script needed to run inference on the same model as either
    batch jobs or online given the specific task inside the notebook.
    3. A text file named requirements.txt that includes all libraries need to run both python files mentioned before.
    A good output should be well documented, easily understandable and should fix all errors present inside the
    notebook if there are any.
    """
    notebook = dspy.InputField(desc="The Jupyter Notebook")
    output: OutputSchema = dspy.OutputField()


# """Generate a requirements.txt, a train.py with specific dockerfile and an inference.py with specific dockerfile
#     given a Jupyter Notebook. For the libraries use the newest version possible. Focus on documentation as well as
#     correctness."""

def set_prompt(prompt_text: str):
    """Dynamically sets the prompt instruction for the dspy Signature."""
    CodeGenSignature.__doc__ = prompt_text
