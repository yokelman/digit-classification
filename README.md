This is a simple handwritten digit classifier, trained on the MNIST database and has an HTML canvas frontend.

# Installation
First, clone the repository with:
```
git clone https://github.com/yokelman/digit-classification.git
```
Then inside the folder `digit-classification` create a virtual environment and install requirements with:
```
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Note: A valid version of Python (like 3.10) is required. The latest version most probably won't work for installing TensorFlow.

# Usage
Run index.html on port 5500 as follows:
```
python -m http.server 5500
```
Then start the server:
```
source venv/bin/activate
python server.py
```

Now visit `http://localhost:5500` and voila!
