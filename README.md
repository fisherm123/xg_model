# Expected Goals (xG) Model
Fisher Marks & Nicholas Strickler

**Final Report** 
We thought formatting the report as a Medium story would be a clean and nice way of presenting the insights for our project. It is an unlisted story (not public, only accessible through the link below). We also exported as a pdf in the final_report.pdf file.
https://medium.com/@fisher.marks/expected-goals-xg-model-ce4e5ccd3db6

**Getting Started**
First, make sure you are working in a Python environment with the right packages installed. To do so, simply activate a new environment and run the command ```pip install -r requirements.txt``` in the cloned project directory. 

**Data Loading**
In the data directory, we already have a numpy file containing the cleaned shot vectors used by our model, so our results can be replicated by just cloning our repository and running the Jupyter Notebooks. If you would like to select your own features, clone the StatsBomb open data repository here: https://github.com/statsbomb/open-data (or at least the data/events folder as that is all the model needs). Then you can edit the getshots.py file in order to select which shot attributes you would like to include.

**Data Visualizations**
The final project report contains several heatmaps, both of which can be replicated simply by running the Jupyter Notebook entitled data_visualizations.ipynb. The images uploaded to the final report can be found in the images directory.

**Logistic Regression**
Our Logistic Regression model can be found in the Jupyter Notebook entitled logistic_regression.ipynb. The hyper-parameters used to get the results discussed in the final report those currently in the notebook, but for reference here we used
```
model_ours.fit(x_train, 
               y_train, 
               T=6000, 
               alpha=8e-5, 
               eta_decay_factor=1e-5, 
               batch_size=8000, 
               optimizer_type='stochastic_gradient_descent')
```
Simply run the notebook in the project directory in a Python environment with the requirements in requirements.txt. The graphs used in the final report can also be produced in the same notebook, and the exact graph used in the final report can be found in the images directory.

**Neural Network**
We also achieved the results from our Neural Network using the architecture and hyper-parameters currently in the neural_network.ipynb notebook, but for reference here:
```
# Batch size - number of shots within a training batch of one training iteration
N_BATCH = 200

# Training epoch - number of passes through the full training dataset
N_EPOCH = 35

# Learning rate - step size to update parameters
LEARNING_RATE = 0.01

# Learning rate decay - scaling factor to decrease learning rate at the end of each decay period
LEARNING_RATE_DECAY = 0.75

# Learning rate decay period - number of epochs before reducing/decaying learning rate
LEARNING_RATE_DECAY_PERIOD = 4
```
```
def __init__(self, n_input_feature, n_output):
        super(NeuralNetwork, self).__init__()

        self.fully_connected_layer_1 = torch.nn.Linear(n_input_feature, 128)
        self.fully_connected_layer_2 = torch.nn.Linear(128, 256)
        self.fully_connected_layer_3 = torch.nn.Linear(256, 512)
        self.fully_connected_layer_4 = torch.nn.Linear(512, 1024)

        self.output = torch.nn.Linear(1024, n_output)
```
Simply run the notebook in the project directory in a Python environment with the requirements in requirements.txt. The graphs used in the final report can also be produced in the same notebook, and the exact graph used in the final report can be found in the images directory.
