import matplotlib.pyplot as plt
# Set Matplotlib defaults
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping



# 0. Auxiliary Function
def group_split(X, y, group, train_size=0.75):
	"""
	A grouped split to keep all of an artist's songs in one split or the other in order to prevent signal leakage
	"""
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])



# 1. Read the dataset
spotify = pd.read_csv('../input/dl-course-data/spotify.csv')
X = spotify.copy().dropna()
y = X.pop('track_popularity')

# Target and predictors (12 numerical and 1 categorical feature-predictor)
artists = X['track_artist']
features_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']
features_cat = ['playlist_genre']


# 2. Preprocessor for all the columns (scaler + OH encoder)
preprocessor = make_column_transformer( (StandardScaler(), features_num),
    									(OneHotEncoder(), features_cat) )
X_train, X_valid, y_train, y_valid = group_split(X, y, artists)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
# Popularity is on a scale 0-100, so I have to rescale to 0-1.
y_train = y_train / 100 
y_valid = y_valid / 100

input_shape = [X_train.shape[1]]
print("Input shape: {}".format(input_shape))



# 3. Create a linear model with 1 neuron and plot the losses
model = keras.Sequential([ layers.Dense(1, input_shape=input_shape) ])

model.compile(optimizer='adam', loss='mae')

history = model.fit(X_train, y_train,
    				validation_data=(X_valid, y_valid),
    				batch_size=512,
    				epochs=50,
    				verbose=0) # suppress output

history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));



# 4. Zoom in after 10-th epoch
history_df.loc[10:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
print("The gap between these curves is quite small and the validation loss never increases, so it's more likely that the network is underfitting than overfitting. It would be worth experimenting with more capacity to see if that's the case.", end="\n\n\n")



# 5. Create a more complex network with more neurons
model = keras.Sequential([ layers.Dense(128, activation='relu', input_shape=input_shape),
    					layers.Dense(64, activation='relu'),
						layers.Dense(1) ])
model.compile(optimizer='adam', loss='mae')

history = model.fit(X_train, y_train,
    				validation_data=(X_valid, y_valid),
    				batch_size=512,
    				epochs=50)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
print("Now the validation loss begins to rise very early, while the training loss continues to decrease. This indicates that the network has begun to overfit. At this point, we would need to try something to prevent it, either by reducing the number of units or through a method like early stopping. ", end="\n\n\n")



# 6. Create an object about early stopping rounds
early_stopping = EarlyStopping(min_delta = 0.001,
                               patience = 5,
                               restore_best_weights = True)

model = keras.Sequential([ layers.Dense(128, activation='relu', input_shape=input_shape),
    					   layers.Dense(64, activation='relu'),    
    					   layers.Dense(1) ])
model.compile(optimizer='adam', loss='mae')

history = model.fit(X_train, y_train,
    				validation_data=(X_valid, y_valid),
    				batch_size=512,
    				epochs=50,
    				callbacks=[early_stopping] )

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
print("The early stopping callback did stop the training once the network began overfitting. Moreover, by including restore_best_weights we still get to keep the model where validation loss was lowest.", end="\n\n\n")
