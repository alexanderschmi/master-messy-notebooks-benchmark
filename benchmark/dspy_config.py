import dspy

class CodeGenSignature(dspy.Signature):
    """Generate a requirements.txt, a train.py with specific dockerfile and a inference.py with specific dockerfile given a Jupyter Notebook"""
    notebook = dspy.InputField(desc="Jupyter Notebook that needs to be deployed")
    requirements = dspy.OutputField(desc="The requirements.txt which includes all libarys and model/training parameters")
    train = dspy.OutputField(desc="The train.py")
    inference = dspy.OutputField(desc="The inference.py")
    docker_file_train = dspy.OutputField(desc="The Dockerfile to run the train.py script")
    docker_file_inference = dspy.OutputField(desc="The Dockerfile to run the inference.py script")