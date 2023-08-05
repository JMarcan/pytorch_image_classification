# pytorch_image_classification

This project uses PyTorch framework<br>
and AWS SageMaker to train an image classifier<br>
to classify blood cell images.<br><br>

The classification is based on ResNet-50 neural network that we fine-tune here for the target dataset<br>
while utilizing SageMaker profiling, debugger, and hyperparameter tuning.<br>
![Project-Diagram](assets/project-diagram.png)<br><br>

This project was part of my Machine Learning Nanodegree at Udacity<br>
The dataset I chosen come from [kaggle/paultimothymooney/blood-cells](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)<br>
and contains 4 classes of blood-cells: Eosinophils, Lymphocytes, Monocytes, and Neutrophils.<br><br>

![dataset-cover](assets/dataset-cover.jpg)

The model is then deployed to a Sagemaker Endpoint for inference.