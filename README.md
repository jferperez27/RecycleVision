
  # RecycleVision ♻️
  RecycleVision is a computer vision project designed to classify recyclable and non-recyclable materials
  using the [TrashNet dataset](https://huggingface.co/datasets/garythung/trashnet). This project includes dataset handling and preprocessing.
  Additionally, this project uses the TensorFlow library for model setup and creation, training, and refinement.

  Current baseline: ~74% accuracy on test split.
  
## Features  
- Dataset import via HuggingFace Hub
- Automated train/test data with reproducibility
- TensorFlow/Keras training pipeline
- Single image inference
- Accuracy/Evaluation metrics via `metrics.py`
- Pytest suite
- Interactive REPL for easy inference, training, and testing
- Support to train/modify your own custom models and validate with test suite

## REPL (Read-Eval-Print-Loop)
Start the REPL and type `help` for commands.


`help` - *Show this help message*

`test` - *Run tests*

`train` - *Train a new model*

`load_model` - *Load a trained model*

`test_model` - *Test model predictions with test images*

`inference` - *Run custom single image prediction*

`exit` - *Exit the REPL*


## Run Locally  
Clone the project  

~~~bash  
  git clone https://github.com/jferperez27/RecycleVision.git
~~~

Go to the project directory  

~~~bash  
  cd RecycleVision
~~~

Create/Activate Virtual Environment

~~~bash  
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
#.venv\Scripts\activate      # Windows
~~~

Install Dependencies

~~~bash
pip install -r requirements.txt
~~~

Run the REPL

~~~bash
python3 repl.py
~~~

