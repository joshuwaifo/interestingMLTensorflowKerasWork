# open the the training input image using rasterio and convert it to a numpy array

# 1. rooftop classifier (rooftop or not)


# 2. import the needed libraries


import rasterio
import numpy as np


# observe that the shape of the ndarray is 8 x 650 x 650


with rasterio.open('data/train/train_input.tif', 'r') as training_input_ds:
    training_input_image_type_ndarray = training_input_ds.read()  # read all raster values


# delete the training_input_ds for memory purposes


del training_input_ds

# (b)


print(training_input_image_type_ndarray.shape)


# note that the datatype is an unsigned integer with 16 bits ie it ranges from 0 to 65,535


print(training_input_image_type_ndarray.dtype)


# (b) extras


# note that it is a 3D tensor at the moment


print(training_input_image_type_ndarray.ndim)


# try the brute force solution for now and then clean it up later


def obtain_min_max_channel(channel_type_ndarray):
    return np.min(channel_type_ndarray), np.max(channel_type_ndarray)

coastal_channel_train = training_input_image_type_ndarray[0]
blue_channel_train = training_input_image_type_ndarray[1]
green_channel_train = training_input_image_type_ndarray[2]
yellow_channel_train = training_input_image_type_ndarray[3]
red_channel_train = training_input_image_type_ndarray[4]
redEdge_channel_train = training_input_image_type_ndarray[5]
nearInfrared1_channel_train = training_input_image_type_ndarray[6]
nearInfrared2_channel_train = training_input_image_type_ndarray[7]

coastal_train_min, coastal_train_max = obtain_min_max_channel(coastal_channel_train)
blue_train_min, blue_train_max = obtain_min_max_channel(blue_channel_train)
green_train_min, green_train_max = obtain_min_max_channel(green_channel_train)
yellow_train_min, yellow_train_max = obtain_min_max_channel(yellow_channel_train)
red_train_min, red_train_max = obtain_min_max_channel(red_channel_train)
redEdge_train_min, redEdge_train_max = obtain_min_max_channel(redEdge_channel_train)
nearInfrared1_train_min, nearInfrared1_train_max = obtain_min_max_channel(nearInfrared1_channel_train)
nearInfrared2_train_min, nearInfrared2_train_max = obtain_min_max_channel(nearInfrared2_channel_train)


# Note that each matrix value of the training normalised channels is now a 64-bit float between 0 and 1


normalised_coastal_channel_train = (coastal_channel_train - coastal_train_min)/(coastal_train_max-coastal_train_min)
normalised_blue_channel_train = (blue_channel_train - blue_train_min)/(blue_train_max-blue_train_min)
normalised_green_channel_train = (green_channel_train - green_train_min)/(green_train_max-green_train_min)
normalised_yellow_channel_train = (yellow_channel_train - yellow_train_min)/(yellow_train_max-yellow_train_min)
normalised_red_channel_train = (red_channel_train - red_train_min)/(red_train_max-red_train_min)
normalised_redEdge_channel_train = (redEdge_channel_train - redEdge_train_min)/(redEdge_train_max-redEdge_train_min)
normalised_nearInfrared1_channel_train = (nearInfrared1_channel_train - nearInfrared1_train_min)/(nearInfrared1_train_max-nearInfrared1_train_min)
normalised_nearInfrared2_channel_train = (nearInfrared2_channel_train - nearInfrared2_train_min)/(nearInfrared2_train_max-nearInfrared2_train_min)

training_input_image_type_ndarray_2D = np.reshape(training_input_image_type_ndarray, (8, -1))
per_channel_max = np.max(training_input_image_type_ndarray_2D, axis=1, keepdims=True)
per_channel_min = np.min(training_input_image_type_ndarray_2D, axis=1, keepdims=True)

test_output_2D = (training_input_image_type_ndarray_2D - per_channel_min)/(per_channel_max - per_channel_min)
test_output_3D = np.reshape(test_output_2D, (8, 650, 650))

normalised_training_input_image_type_ndarray = np.array(
    [
        normalised_coastal_channel_train,
        normalised_blue_channel_train,
        normalised_green_channel_train,
        normalised_yellow_channel_train,
        normalised_red_channel_train,
        normalised_redEdge_channel_train,
        normalised_nearInfrared1_channel_train,
        normalised_nearInfrared2_channel_train
    ]
)


print("part c")

# to be confirmed

# pip install -U scikit-learn

# make use of scikit-learn (also downloads joblib, scipy and threadpoolctl)

# maybe don't use this for now




# pip install matplotlib

# make use of matplotlib also downloads cycler, kiwisolver, pillow, python-dateutil, six)


from matplotlib import pyplot as plt


# Plot

normalised_rgb_train_type_ndarray = np.dstack(
    (
        normalised_red_channel_train,
        normalised_green_channel_train,
        normalised_blue_channel_train
    )
)

# make use of numpy dstack (depth stack, along the third axis)


fig, axs = plt.subplots(2,2)

cax_00 = axs[0,0].imshow(normalised_rgb_train_type_ndarray)
axs[0,0].xaxis.set_major_formatter(plt.NullFormatter())  # kill xlabels
axs[0,0].yaxis.set_major_formatter(plt.NullFormatter())  # kill ylabels

cax_01 = axs[0,1].imshow(normalised_red_channel_train, cmap='Reds')
fig.colorbar(cax_01, ax=axs[0,1])
axs[0,1].xaxis.set_major_formatter(plt.NullFormatter())
axs[0,1].yaxis.set_major_formatter(plt.NullFormatter())

cax_10 = axs[1,0].imshow(normalised_green_channel_train, cmap='Greens')
fig.colorbar(cax_10, ax=axs[1,0])
axs[1,0].xaxis.set_major_formatter(plt.NullFormatter())
axs[1,0].yaxis.set_major_formatter(plt.NullFormatter())

cax_11 = axs[1,1].imshow(normalised_blue_channel_train, cmap='Blues')
fig.colorbar(cax_11, ax=axs[1,1])
axs[1,1].xaxis.set_major_formatter(plt.NullFormatter())
axs[1,1].yaxis.set_major_formatter(plt.NullFormatter())
plt.show()

#  clean up the image plot labels and method, it can be much neater but for now let's leave it as is

#  looks like it's an image of the rooftop of houses


# Number 1 has been completed now




# Number 2

# Reshape the normalised training image to a form of number of samples, 8 features (each normalised channel)

# Visualise the output for now

with rasterio.open('data/train/train_target.tif', 'r') as training_target_ds:
    training_target_image_type_ndarray = training_target_ds.read()  # read all raster values

# mention that the training target image contains integer values ranging from 0 to 255 (uint8)

# as there is only one image but the shape dimension is 1, 650, 650 I need to get a 650, 650 shape by getting the first (and only sample)

#  ensure the plot makes 0 as black and the targets 255 currently as white

plt.imshow(training_target_image_type_ndarray[0], cmap="Greys_r")
plt.title("Target output")
plt.axis("off")
plt.show()


# try and merge both together

plt.imshow(normalised_rgb_train_type_ndarray, cmap="Greys_r")
plt.imshow(training_target_image_type_ndarray[0], cmap="Greys_r", alpha=0.35)
#Greys_r, Greys, cividis, plasma, jet, magma, inferno, viridis
# this is enough it seems like it's showing the rooftops of buildings

plt.title("Merge of both images")
plt.axis("off")
plt.show()

# rescale the labels to be either 0 or 1 instead of 0 or 255
cleaned_training_target_image_type_ndarray = training_target_image_type_ndarray.copy()
cleaned_training_target_image_type_ndarray[cleaned_training_target_image_type_ndarray == 255] = 1


logistic_train_data_before_transpose = np.reshape(normalised_training_input_image_type_ndarray, (8, -1))
logistic_train_data = logistic_train_data_before_transpose.T
logistic_train_target = np.reshape(cleaned_training_target_image_type_ndarray, (650 * 650))

# understand the data distribution

unique_classes, counts = np.unique(logistic_train_target, return_counts=True)
print( dict(zip(unique_classes, counts)) )

# get the perecentage of values
class_percentages = (counts/np.sum(counts))*100

plt.hist(logistic_train_target)
plt.show()

# observation  is that the majority of the outputs are black pixels, 87% is in class 0 and the rest is in class 1 so it's more appropriate to look at the confusion matrix and use a different metric like precision or recall

# now do this properly afterwards

# traindata split NEXT THING TO DO

# Try iterating on different optimizers/solvers
# newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’



# # newton-cg

# lbfgs

# liblinear

# sag

# saga

# class_weightdict or ‘balanced’, default=None

# # Try stratified sampling of the dataset when splitting
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(logistic_train_data, logistic_train_target, shuffle=True, stratify=logistic_train_target, test_size=0.4, random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, average_precision_score, roc_auc_score, f1_score, precision_score
logisticRegression_sklearn = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)



# logisticRegression_classifier1 = LogisticRegression(random_state=0, solver='newton-cg').fit(X_train, y_train)
# logisticRegression_classifier2 = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
# logisticRegression_classifier3 = LogisticRegression(random_state=0, solver='liblinear').fit(X_train, y_train)
# logisticRegression_classifier4 = LogisticRegression(random_state=0, solver='sag').fit(X_train, y_train)
# logisticRegression_classifier5 = LogisticRegression(random_state=0, solver='saga').fit(X_train, y_train)


# print(logisticRegression_classifier.score(logistic_train_data, logistic_train_target))


# There are 9 trainable parameters 8 for the coefficients of each satellite band and 1 serving as a bias term/ y-intercept

# JUST FOR TESTING PURPOSES, IT'S NOT GOOD THEORY OR PRACTICE, BUT JUST WANT TO QUICKLY ENSURE THINGS ARE WORKING FROM A PIPELINE PERSPECTIVE


y_val_predSklearn = logisticRegression_sklearn.predict(X_val)


# y_val_pred1 = logisticRegression_classifier1.predict(X_val)
# y_val_pred2 = logisticRegression_classifier2.predict(X_val)
# y_val_pred3 = logisticRegression_classifier3.predict(X_val)
# y_val_pred4 = logisticRegression_classifier4.predict(X_val)
# y_val_pred5 = logisticRegression_classifier5.predict(X_val)
#
print("1")
# average recall
print(balanced_accuracy_score(y_val, y_val_predSklearn))
# average precision
print(average_precision_score(y_val, y_val_predSklearn))
# TRY TO IMPROVE THIS
print(roc_auc_score(y_val, y_val_predSklearn))
# harmonic mean between precision and recall
print(f1_score(y_val, y_val_predSklearn))

#
# print("2")
# print(balanced_accuracy_score(y_val, y_val_pred2))
# print(average_precision_score(y_val, y_val_pred2))
# print(roc_auc_score(y_val, y_val_pred2))
# print(f1_score(y_val, y_val_pred2))
#
#
# print("3")
# print(balanced_accuracy_score(y_val, y_val_pred3))
# print(average_precision_score(y_val, y_val_pred3))
# print(roc_auc_score(y_val, y_val_pred3))
# print(f1_score(y_val, y_val_pred3))
#
#
# print("4")
# print(balanced_accuracy_score(y_val, y_val_pred4))
# print(average_precision_score(y_val, y_val_pred4))
# print(roc_auc_score(y_val, y_val_pred4))
# print(f1_score(y_val, y_val_pred4))






# I think average precision might be the one but let's look at the output visually to see if it looks appropriate, if it looks nice visually we call use the 83% + if it's bad visually we'll use the 73% one



# chosen model
# the default is okay
# logisticRegression_classifier2 = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)

# 9 parameters, 8 from coef_ and 1 from intercept_


with rasterio.open('data/validation/validation_input.tif', 'r') as validation_input_ds:
    validation_input_image_type_ndarray = validation_input_ds.read()  # read all raster values


with rasterio.open('data/validation/validation_target.tif', 'r') as validation_target_ds:
    validation_target_image_type_ndarray = validation_target_ds.read()  # read all raster values

coastal_channel_val = validation_input_image_type_ndarray[0]
blue_channel_val = validation_input_image_type_ndarray[1]
green_channel_val = validation_input_image_type_ndarray[2]
yellow_channel_val = validation_input_image_type_ndarray[3]
red_channel_val = validation_input_image_type_ndarray[4]
redEdge_channel_val = validation_input_image_type_ndarray[5]
nearInfrared1_channel_val = validation_input_image_type_ndarray[6]
nearInfrared2_channel_val = validation_input_image_type_ndarray[7]


normalised_coastal_channel_val = (coastal_channel_val - coastal_train_min)/(coastal_train_max-coastal_train_min)
normalised_blue_channel_val = (blue_channel_val - blue_train_min)/(blue_train_max-blue_train_min)
normalised_green_channel_val = (green_channel_val - green_train_min)/(green_train_max-green_train_min)
normalised_yellow_channel_val = (yellow_channel_val - yellow_train_min)/(yellow_train_max-yellow_train_min)
normalised_red_channel_val = (red_channel_val - red_train_min)/(red_train_max-red_train_min)
normalised_redEdge_channel_val = (redEdge_channel_val - redEdge_train_min)/(redEdge_train_max-redEdge_train_min)
normalised_nearInfrared1_channel_val = (nearInfrared1_channel_val - nearInfrared1_train_min)/(nearInfrared1_train_max-nearInfrared1_train_min)
normalised_nearInfrared2_channel_val = (nearInfrared2_channel_val - nearInfrared2_train_min)/(nearInfrared2_train_max-nearInfrared2_train_min)

normalised_validation_input_image_type_ndarray = np.array(
    [
        normalised_coastal_channel_val,
        normalised_blue_channel_val,
        normalised_green_channel_val,
        normalised_yellow_channel_val,
        normalised_red_channel_val,
        normalised_redEdge_channel_val,
        normalised_nearInfrared1_channel_val,
        normalised_nearInfrared2_channel_val
    ]
)


# rescale the labels to be either 0 or 1 instead of 0 or 255
cleaned_validation_target_image_type_ndarray = validation_target_image_type_ndarray.copy()
cleaned_validation_target_image_type_ndarray[cleaned_validation_target_image_type_ndarray == 255] = 1


logistic_val_data_before_transpose = np.reshape(normalised_validation_input_image_type_ndarray, (8, -1))
logistic_val_data = logistic_val_data_before_transpose.T
logistic_val_target = np.reshape(cleaned_validation_target_image_type_ndarray, (650 * 650))

logistic_val_pred = logisticRegression_sklearn.predict(logistic_val_data)

print("Logistic Regression Sklearn Model validation performance")
print(balanced_accuracy_score(logistic_val_target, logistic_val_pred))
print(average_precision_score(logistic_val_target, logistic_val_pred))
print(roc_auc_score(logistic_val_target, logistic_val_pred))
print(f1_score(logistic_val_target, logistic_val_pred))



from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D
from keras.metrics import Precision, Recall


from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


input_shape = (8)

logistic_regression_model = Sequential()
logistic_regression_model.add( Dense(1, activation='sigmoid') )

batch_size = 128
# epochs = 240 (79.3% recall validation)
epochs = 50 # (~75% validation recall)

logistic_regression_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[precision_m, recall_m, "accuracy"])

logistic_regression_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
logistic_regression_model.summary()

#  clip the outputs

logistic_val_pred_keras = logistic_regression_model.predict(logistic_val_data)
logistic_val_pred_keras[logistic_val_pred_keras > 0.5] = 1
logistic_val_pred_keras[logistic_val_pred_keras <= 0.5] = 0


print("Logistic Regression Keras Model validation performance")
print(balanced_accuracy_score(logistic_val_target, logistic_val_pred_keras))
print(average_precision_score(logistic_val_target, logistic_val_pred_keras))
print(roc_auc_score(logistic_val_target, logistic_val_pred_keras))
print(f1_score(logistic_val_target, logistic_val_pred_keras))

#  save the model with the best validation_recall



# maximise the validation recall


# print("Actual Validation outputs")
# # average recall
# print(balanced_accuracy_score(logistic_val_target, logistic_val_pred1))
# # average precision
# print(average_precision_score(logistic_val_target, logistic_val_pred1))
# # TRY TO IMPROVE THIS
# print(roc_auc_score(logistic_val_target, logistic_val_pred1))
# # harmonic mean between precision and recall
# print(f1_score(logistic_val_target, logistic_val_pred1))


#  try a simple keras model mlp as a replacement to the logistic regression model to see what the results are like

with rasterio.open('data/test/test_input.tif', 'r') as test_input_ds:
    test_input_image_type_ndarray = test_input_ds.read()  # read all raster values


coastal_channel_test = test_input_image_type_ndarray[0]
blue_channel_test = test_input_image_type_ndarray[1]
green_channel_test = test_input_image_type_ndarray[2]
yellow_channel_test = test_input_image_type_ndarray[3]
red_channel_test = test_input_image_type_ndarray[4]
redEdge_channel_test = test_input_image_type_ndarray[5]
nearInfrared1_channel_test = test_input_image_type_ndarray[6]
nearInfrared2_channel_test = test_input_image_type_ndarray[7]


normalised_coastal_channel_test = (coastal_channel_test - coastal_train_min)/(coastal_train_max-coastal_train_min)
normalised_blue_channel_test = (blue_channel_test - blue_train_min)/(blue_train_max-blue_train_min)
normalised_green_channel_test = (green_channel_test - green_train_min)/(green_train_max-green_train_min)
normalised_yellow_channel_test = (yellow_channel_test - yellow_train_min)/(yellow_train_max-yellow_train_min)
normalised_red_channel_test = (red_channel_test - red_train_min)/(red_train_max-red_train_min)
normalised_redEdge_channel_test = (redEdge_channel_test - redEdge_train_min)/(redEdge_train_max-redEdge_train_min)
normalised_nearInfrared1_channel_test = (nearInfrared1_channel_test - nearInfrared1_train_min)/(nearInfrared1_train_max-nearInfrared1_train_min)
normalised_nearInfrared2_channel_test = (nearInfrared2_channel_test - nearInfrared2_train_min)/(nearInfrared2_train_max-nearInfrared2_train_min)

normalised_test_input_image_type_ndarray = np.array(
    [
        normalised_coastal_channel_test,
        normalised_blue_channel_test,
        normalised_green_channel_test,
        normalised_yellow_channel_test,
        normalised_red_channel_test,
        normalised_redEdge_channel_test,
        normalised_nearInfrared1_channel_test,
        normalised_nearInfrared2_channel_test
    ]
)


normalised_rgb_test_type_ndarray = np.dstack(
    (
        normalised_red_channel_test,
        normalised_green_channel_test,
        normalised_blue_channel_test
    )
)

# # rescale the labels to be either 0 or 1 instead of 0 or 255
# cleaned_validation_target_image_type_ndarray = validation_target_image_type_ndarray.copy()
# cleaned_validation_target_image_type_ndarray[cleaned_validation_target_image_type_ndarray == 255] = 1


logistic_test_data_before_transpose = np.reshape(normalised_test_input_image_type_ndarray, (8, -1))
logistic_test_data = logistic_test_data_before_transpose.T


# logistic_val_target = np.reshape(cleaned_validation_target_image_type_ndarray, (650 * 650))

logistic_test_pred1 = logistic_regression_model.predict(logistic_test_data)
logistic_test_pred1[logistic_test_pred1 > 0.5] = 1
logistic_test_pred1[logistic_test_pred1 <= 0.5] = 0
logistic_test_predicted_imageKeras = np.reshape(logistic_test_pred1, (650, 650))

logistic_test_pred2 = logisticRegression_sklearn.predict(logistic_test_data)
logistic_test_predicted_imageSklearn = np.reshape(logistic_test_pred2, (650, 650))


print("we get here")

plt.imshow(normalised_rgb_test_type_ndarray, cmap="Greys_r")
plt.title("Actual RGB test image ")
plt.axis("off")
plt.show()


plt.imshow(logistic_test_predicted_imageSklearn, cmap="Greys_r")
plt.title("Test predicted output Sklearn")
plt.axis("off")
plt.show()

plt.imshow(logistic_test_predicted_imageKeras, cmap="Greys_r")
plt.title("Test predicted output Keras")
plt.axis("off")
plt.show()




#  empty plot visually

logistic_test_pred1proba = logistic_regression_model.predict(logistic_test_data)
logistic_test_predicted_proba_imageKeras = np.reshape(logistic_test_pred1proba, (650, 650))

logistic_test_pred2proba = logisticRegression_sklearn.predict_proba(logistic_test_data)[:,1]
logistic_test_predicted_proba_imageSklearn = np.reshape(logistic_test_pred2proba, (650, 650))


print("we get here")

plt.imshow(normalised_rgb_test_type_ndarray, cmap="Greys_r")
plt.title("Actual RGB test image")
plt.axis("off")
plt.show()


plt.imshow(logistic_test_predicted_proba_imageSklearn, cmap="Greys_r")
plt.title("Test predicted probability output Sklearn")
plt.axis("off")
plt.show()

plt.imshow(logistic_test_predicted_proba_imageKeras, cmap="Greys_r")
plt.title("Test predicted probability output Keras")
plt.axis("off")
plt.show()


# for a better plot maybe look at the predict_probabilities plot instead

# # # try and merge both together
# #
# plt.imshow(normalised_rgb_test_type_ndarray, cmap="Greys_r")
# plt.imshow(logistic_test_predicted_image, cmap="Greys_r", alpha=0.35)

# plot the values

# I choose for now the sklearn model


sklearn_paramters = np.concatenate([logisticRegression_sklearn.coef_.flatten(),logisticRegression_sklearn.intercept_])
print(sklearn_paramters)


keras_parameters = np.concatenate([logistic_regression_model.get_weights()[0].flatten(), logistic_regression_model.get_weights()[1]])
print(keras_parameters)





N = 9
list_of_values = sklearn_paramters
ind = np.arange(N) # the x locations for the groups
width = 0.35 # the width of the bars: can also be len(x) sequence


fig, ax = plt.subplots()
p1 = ax.bar(ind, list_of_values, width, label='weight/bias values')
ax.axhline(0, color='grey', linewidth=0.8)

ax.set_title('Satellite band coefficient values')
ax.set_xticks(ind)
ax.set_xticklabels(('Coastal', 'Blue', 'Green', 'Yellow', 'Red', 'Red-edge', 'Near-infrared 1', 'Near-infrared 2', 'constant value'))
ax.legend()
plt.xticks(rotation=90)
plt.show()


# positive/negative effect all other things being equal
# increasing blah does p

# The impact of blue satellite band and near-edge satellite band seems to overpower the other values

# general speaking there are 2 groups here
# the positive values and the negative ones
# increasing the presence/intensity of these values like near-edge, coastal and so on for a given pixel would cause the binary classifier to more likely suggest that that given pixel is a rooftop and vice versa reducing the intensity of these values would make the classifier lean more towards the no-rooftop class
# the opposite appears to hold true for the negative weighted satellite bands and the bias (like Blue, near-infrared 2 and so on)


# Numero 3

cnn_model = keras.Sequential()
# input of size 8
cnn_model.add( Conv2D(32, kernel_size=(3,3), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform') )
cnn_model.add( Conv2D(1, kernel_size=(3,3), activation='linear', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform') )
Conv1D
# batch_size = 128
# # epochs = 240 (79.3% recall validation)
# epochs = 50 # (~75% validation recall)

cnn_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[precision_m, recall_m, "accuracy"])
print(cnn_model.summary())

# # logistic_regression_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
# logistic_regression_model.summary()
#
#
#     [
#         keras.Input(shape=input_shape),
#         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation="softmax"),
#     ]
# )


# (b)

# (((3 x 3) x 8) x 32) + 32
#
# ( ((3 x 3) x 32) x 1) + 1
#
# Including biases the total number is ((((3 x 3) x 8) x 32) + 32) + ( ( ((3 x 3) x 32) x 1) + 1 )
#  = 2336 + 289  = 2625 trainable parameters

# channel_last_normalised_training_input_image_type_ndarray = np.transpose(normalised_training_input_image_type_ndarray, (1,2,0))
# added_axis_channel_last_normalised_training_input_image_type_ndarray = np.reshape(channel_last_normalised_training_input_image_type_ndarray, (1,650,650,8))
# channel_last_normalised_training_input_image_type_ndarray[3,6,:]

# print(normalised_training_input_image_type_ndarray[:,3,649])
# print(channel_last_normalised_training_input_image_type_ndarray[3,649,:])
# print(added_axis_channel_last_normalised_training_input_image_type_ndarray[0,3,649,:])

# (c)

# just apply the modell to it


# (d) # it appears like this filter gets activated more when it receives as input the presence of diagonal lines


# Note that for the same task one requires 9 parameters (logistic regression model whereas the other requires 2.6 thousand parameters)






#
# names1 = ["Ava", "Emma", "Olivia"]
# names2 = ["Olivia", "Sophia", "Emma"]
#
# print(list(set(names1 + names2)))
#
#
#
#
#
#
#
#
# files_type_dict = {
#     'Input.txt' : 'Randy',
#     'Code.py': 'Stan',
#     'Output.txt': 'Randy'
# }
#
# new_dict = dict.fromkeys(files_type_dict.values(), None)
#
# for key, value in new_dict.items():
#     new_dict[key] = [ other_dict_key for other_dict_key in files_type_dict.keys() if files_type_dict[other_dict_key]==key ]
#
#
#
#
#
#
#
#
# def find_roots(a, b, c):
#     x1 = ((-1*b) + (b**2 - 4*a*c)**(0.5))/(2*a)
#     x2 = ((-1*b) - (b**2 - 4*a*c)**(0.5))/(2*a)
#     return x1, x2
#
# print(find_roots(2, 10, 8));
#
#
#
#
#
#
#
#
# class IceCreamMachine:
#
#     def __init__(self, ingredients, toppings):
#         self.ingredients = ingredients
#         self.toppings = toppings
#
#     def scoops(self):
#         combination = []
#         for ingredient in self.ingredients:
#             for topping in self.toppings:
#                 combination.append([ingredient, topping])
#         return combination
#
#
# if __name__ == "__main__":
#     machine = IceCreamMachine(["vanilla", "chocolate"], ["chocolate sauce"])
#     print(machine.scoops())  # should print[['vanilla', 'chocolate sauce'], ['chocolate', 'chocolate sauce']]
#
#
# # Test dome gold certificate
#
