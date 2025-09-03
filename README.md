
  # RecycleVision ♻️
  RecycleVision is a computer vision project designed to classify recyclable and non-recyclable materials
  using the [TrashNet dataset](https://huggingface.co/datasets/garythung/trashnet). This project includes dataset handling and preprocessing.
  Additionally, this project uses the TensorFlow library for model setup and creation, training, and refinement.

  ~~Current baseline: ~74% accuracy on test split.~~

  Latest model (v3): **86% test accuracy** on held-out test set
  
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

## Screenshots
Model Predictions

<img width="640" height="480" alt="RecycleVision_prediction" src="https://github.com/user-attachments/assets/e13fb2dd-56da-4786-ab48-4e8cc5c049b5" />

REPL Screenshot (First time loading program)

<img width="771" height="361" alt="RecycleVision_REPL_example" src="https://github.com/user-attachments/assets/902b78b5-c015-4474-9ee3-539cfdae8c04" />

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

