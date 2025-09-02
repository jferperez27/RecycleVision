
  # RecycleVision ♻️
  RecycleVision is a computer vision project designed to classify recyclable and non-recyclable materials
  using the [TrashNet dataset](https://huggingface.co/datasets/garythung/trashnet). This project includes dataset handling and preprocessing.
  Additionally, this project uses the TensorFlow library for model setup and creation, training, and refinement.
  
## Features  
- Dataset import via HuggingFace Hub
- Automated dataset splitting for testing data and training data with reproducability
- Model training with TensorFlow/Keras
- Inference pipeline for single image classification/prediction
- Accuracy/Evaluation metrics via `metrics.py`
- Full test suite using pytest
## Run Locally  
Clone the project  

~~~bash  
  git clone https://link-to-project
~~~

Go to the project directory  

~~~bash  
  cd RecycleVision
~~~

Create/Activate Virtual Environment

~~~bash  
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
~~~

Install Dependencies

~~~bash
pip install -r requirements.txt
~~~