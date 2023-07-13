# Setup plotting
import matplotlib.pyplot as plt
from learntools.deep_learning_intro.dltools import animate_sgd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

# Set Matplotlib defaults
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')


# 1. Read the dataset
fuel = pd.read_csv('../input/dl-course-data/fuel.csv')
X = fuel.copy()
y = X.pop('FE')        # Remove target

# 2. Create a preprocessor for numbers and categorical columns
preprocessor = make_column_transformer( (StandardScaler(), make_column_selector(dtype_include=np.number)),
    									(OneHotEncoder(sparse=False), make_column_selector(dtype_include=object)) )

X = preprocessor.fit_transform(X)
y = np.log(y) # log transform target instead of standardizing
input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))

print('Original data = \n', fuel.head(), end='\n\n\n')
print('Processed data = \n', pd.DataFrame(X[:10,:]).head(), end='\n\n\n')

# 3. Create and compile a DL model using layers and neurons
model = keras.Sequential([  layers.Dense(128, activation='relu', input_shape=input_shape),
    						layers.Dense(128, activation='relu'),    
    						layers.Dense(64, activation='relu'),
    						layers.Dense(1)  ])
model.compile(optimizer='adam', loss='mae')

# 4. Train the model, see the history in epochs, visualize MAE
history = model.fit(X, y, epochs=200, batch_size=128)
history_df = pd.DataFrame(history.history)
# See all the epochs (1-200)
history_df['loss'].plot();
# Start from epoch 5 (5-200)
history_df.loc[5:, ['loss']].plot();


# 5. Evaluate training - Animate SGD

learning_rate = 0.05
batch_size = 128
num_examples = 256

animate_sgd(
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_examples=num_examples,
    # You can also change these, if you like
    steps=50, # total training steps (batches seen)
    true_w=3.0, # the slope of the data
    true_b=2.0) # the bias of the data
