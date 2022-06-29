# Sign-Language-Detection
An AI model that can detect sign language and translate in real time. The model is trained to recognize three simple gestures as of now:
'Hello', 'Thanks' and 'I Love You'. The data for different gestures is supposed to taken by yourself. The model is then trained on this data on an LSTM neural network to learn different signs.

#Steps to run project:

1) Pip install all dependencies from requirements.txt
2) run make_folder program to create folder structure to store keypoint cordinates for different gestures.
3) run Collect Key points python program and extract key features.
4) Run the jupyter-notebook and train your neural network on an LSTM model.
