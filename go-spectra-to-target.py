#%%
# all libraries needed
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from astropy.cosmology import Planck13
from tqdm import tqdm
import scipy.ndimage
from astropy.io import fits
import random
#%%

"""load the dataset"""

# Path to the FITS file
dataset_path = '/your/path/quasars_spectra_combined.fits'

# Open the FITS file and read the data
with fits.open(dataset_path) as hdu:

    data = hdu[1].data  
    
    # any masking operation can be here
    # data = data[data['REDSHIFT']<2]
    
    # make arrays from spectra and targets
    spectra = data['FLUX']
    loglam = data['LOGLAM']
    dist = data['DIST']
    lo = data['LO']
    lx = data['LX']
    gx = data['GX']
    redshift = data['REDSHIFT']

    # Extract the desired features
    logl1350 = data['LOGL1350']
    logl1700 = data['LOGL1700']
    logl2500 = data['LOGL2500']
    logl3000 = data['LOGL3000']
    logl5100 = data['LOGL5100']
    loglbol = data['LOGLBOL']
    logmbh = data['LOGMBH']
    halpha = data['HALPHA']
    halpha_br = data['HALPHA_BR']
    hbeta = data['HBETA']
    hbeta_br = data['HBETA_BR']
    oiii5007 = data['OIII5007']
    oiii5007c = data['OIII5007C']
    mgii = data['MGII']
    mgii_br = data['MGII_BR']
    civ = data['CIV']

#%%
""" Visualize some relevant plots """

# Visualisation Lo vs Lx    
plt.figure()
plt.scatter(lo, lx, c=dist)
plt.xlabel('LO')
plt.ylabel('LX')

# Visualisation gx vs dist
plt.figure()
plt.scatter(dist, gx, c=dist, alpha=0.1)
plt.xlabel('Dist')
plt.ylabel('GX')

# random spectra visualisation
random_indices = random.sample(range(spectra.shape[0]), 4)

plt.figure(figsize=(12, 8))

for idx in random_indices:
    plt.plot(loglam[idx], spectra[idx], label=f'Object {idx}')

plt.xlabel('Log Lambda')
plt.ylabel('Flux')
plt.title('Spectra for Four Random Objects')
plt.legend()
plt.grid(True)
plt.show()

#%%
""" Start epxeriment """

# chose the target of insterest
# interesting targtes: lo, lx, dist
experiment_target = dist.copy()
# experiment_target = lo.copy()
# experiment_target = lx.copy()

# assign spectra and target to generic X and Y variables
X = spectra.copy()
Y = experiment_target.copy()

#%%
# normalize X 

# not clear what is the best way to normalize the spectra
X = X / 10000.

#%%
from sklearn.model_selection import train_test_split

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp, redshift_train, redshift_temp = train_test_split(
    X, Y, redshift, test_size=0.4, random_state=42)

X_val, X_test, y_val, y_test, redshift_val, redshift_test = train_test_split(
    X_temp, y_temp, redshift_temp, test_size=0.5, random_state=42)
#%%
# normalise targets
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Scaler for y (if y is a continuous target variable)
scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# # # Scaler for redshift
scaler_redshift =  MinMaxScaler()
redshift_train = scaler_redshift.fit_transform(redshift_train.reshape(-1, 1)).ravel()
redshift_val = scaler_redshift.transform(redshift_val.reshape(-1, 1)).ravel()
redshift_test = scaler_redshift.transform(redshift_test.reshape(-1, 1)).ravel()

#%%
""" MLP """

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Spectra input 
spectra_input = tf.keras.Input(shape=(X.shape[1],), name='spectra_input')

# Redshift input
redshift_input = tf.keras.Input(shape=(1,), name='redshift_input')
redshift_layers = tf.keras.layers.Dense(10, activation='relu')(redshift_input)
redshift_layers = tf.keras.layers.Dense(100, activation='relu')(redshift_layers)

# Neural network layers for spectra
spectra_layers = tf.keras.layers.Dense(512, activation='relu')(spectra_input)
spectra_layers = tf.keras.layers.Dense(256, activation='relu')(spectra_layers)
spectra_layers = tf.keras.layers.Dense(128, activation='relu')(spectra_layers)

# Combine spectra and redshift inputs
combined = tf.keras.layers.concatenate([spectra_layers, redshift_layers])

# Continue with more layers starting from the shared layers
combined_layers = tf.keras.layers.Dense(128, activation='relu')(combined)
combined_layers = tf.keras.layers.Dense(64, activation='relu')(combined_layers)
combined_layers = tf.keras.layers.Dense(32, activation='relu')(combined_layers)

# Output layer for regression
output = tf.keras.layers.Dense(1)(combined_layers)  

# Create the model
model = tf.keras.models.Model(inputs=[spectra_input, redshift_input], outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

model.summary()

#%%
# Train the model
history = model.fit(
    [X_train, redshift_train], y_train, 
    validation_data=([X_val, redshift_val], y_val), 
    epochs=20, batch_size=1024,
)

#%%
# Evaluate the model
loss = model.evaluate([X_test, redshift_test], y_test)
print(f"Test Loss: {loss}")

#%%
import matplotlib.pyplot as plt

# Plotting training and validation loss
plt.figure(figsize=(8, 5))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#%%
# Predict values
y_pred = model.predict([X_test, redshift_test]).flatten()

# Inverse transform the predictions and actual values
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Plot
plt.figure(figsize=(8, 8))
plt.scatter(y_test_original, y_pred_original, alpha=0.2, s=50, )  # Adjust alpha for point transparency
plt.xlabel('Actual Values - log(L_UV)')
plt.ylabel('Predicted Values - log(L_UV)')
plt.title('Predicted vs Actual Values on Original Scale')
plt.grid(True)
# plt.colorbar()

# Line of perfect prediction
min_val = min(y_test_original.min(), y_pred_original.min())
max_val = max(y_test_original.max(), y_pred_original.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # Red dashed line

plt.show()
#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp(- (x - mean)**2 / (2 * standard_deviation ** 2))

# Calculate residuals
residuals = y_pred_original - y_test_original

# Create histogram (binned data)
bin_heights, bin_borders, _ = plt.hist(residuals, bins=100, label='Histogram', density=True, alpha=0.6, color='g')
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2

# Fit the Gaussian to the histogram data
popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[0., max(bin_heights), np.std(residuals)])

# Plot the fitted Gaussian
x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 1000)
plt.plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt), label='Fitted Gaussian', color='black')

# Annotate the plot with the Gaussian parameters
plt.title(f'Fitted Gaussian: Mean = {popt[0]:.5f}, Sigma = {popt[2]:.3f}')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.legend()

# plt.xlim(-0.2, 0.2)

plt.show()

#%%