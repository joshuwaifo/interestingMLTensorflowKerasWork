# !pip install rasterio

# after running this restart the kernel and move to the next cell and comment the line above

# ROOFTOP segmentation of satellite imagery using logistic regression and an investigation towards the use of CNNs



# import the needed library
from rasterio import open as rOpen

with rOpen('data/train/train_input.tif', 'r') as training_input_ds:
    training_input_image_ndarray = training_input_ds.read()

del training_input_ds # clean up objects

print(f"Array shape: {training_input_image_ndarray.shape}")
print(f"Array type: {training_input_image_ndarray.dtype}")

# import the needed library
import numpy as np


# create a function here for normalising and reuse that function







# FUNCTION HERE








training_input_image_ndarray_2D = np.reshape(training_input_image_ndarray, (8, -1))
per_channel_max = np.max(training_input_image_ndarray_2D, axis=1, keepdims=True)
per_channel_min = np.min(training_input_image_ndarray_2D, axis=1, keepdims=True)

test_output_2D = (training_input_image_ndarray_2D - per_channel_min)/(per_channel_max - per_channel_min)
normalised_training_input_image_ndarray = np.reshape(test_output_2D, (8, 650, 650))

# !python3 -m pip install --upgrade pip
# !python3 -m pip install--upgrade Pillow
# !pip install matplotlib

# after running this restart the kernel and move to the next cell and comment the line above

# import the needed library
from matplotlib import pyplot as plt

normalised_red_channel_train = normalised_training_input_image_ndarray[4]
normalised_green_channel_train = normalised_training_input_image_ndarray[2]
normalised_blue_channel_train = normalised_training_input_image_ndarray[1]



normalised_rgb_train_ndarray = np.dstack(
    (
        normalised_red_channel_train,
        normalised_green_channel_train,
        normalised_blue_channel_train
    )
)

fig, axs = plt.subplots(2,2,figsize=(15,15))

rgb_subplot = axs[0,0].imshow(normalised_rgb_train_ndarray)
axs[0,0].xaxis.set_major_formatter(plt.NullFormatter())  # remove the xlabels
axs[0,0].yaxis.set_major_formatter(plt.NullFormatter())  # remove the ylabels
axs[0,0].title.set_text('RGB Image')

rChannel_subplot = axs[0,1].imshow(normalised_red_channel_train, cmap='Reds')
fig.colorbar(rChannel_subplot, ax=axs[0,1])
axs[0,1].xaxis.set_major_formatter(plt.NullFormatter())
axs[0,1].yaxis.set_major_formatter(plt.NullFormatter())
axs[0,1].title.set_text('Red Channel')

gChannel_subplot = axs[1,0].imshow(normalised_green_channel_train, cmap='Greens')
fig.colorbar(gChannel_subplot, ax=axs[1,0])
axs[1,0].xaxis.set_major_formatter(plt.NullFormatter())
axs[1,0].yaxis.set_major_formatter(plt.NullFormatter())
axs[1,0].title.set_text('Green Channel')


bChannel_subplot = axs[1,1].imshow(normalised_blue_channel_train, cmap='Blues')
fig.colorbar(bChannel_subplot, ax=axs[1,1])
axs[1,1].xaxis.set_major_formatter(plt.NullFormatter())
axs[1,1].yaxis.set_major_formatter(plt.NullFormatter())
axs[1,1].title.set_text('Blue Channel')

plt.show()

# !pip install scikit-learn

# after running this restart the kernel and move to the next cell and comment the line above


# get the training targets/labels

with rOpen('data/train/train_target.tif', 'r') as training_target_ds:
    training_target_image_ndarray = training_target_ds.read()

del training_target_ds # clean up memory

# rescale the labels to be either 0 or 1 instead of 0 or 255
cleaned_training_target_image_ndarray = training_target_image_ndarray.copy() / 255

# make the training data 2D and transpose it to be of shape (number of samples, number of features)
logistic_train_data_before_transpose = np.reshape(normalised_training_input_image_ndarray, (8, -1))
logistic_train_data = logistic_train_data_before_transpose.T

# convert the shape from (1,650,650) to (650*650 =  422500)
logistic_train_target = np.reshape(cleaned_training_target_image_ndarray, (650 * 650))


# try making use of the balanced weight_setting that DINO mentioned


# try making use of f1 score instead and compare the difference



# train the logistic regression model
from sklearn.linear_model import LogisticRegression
logisticRegression_sklearn = LogisticRegression(random_state=0, solver='lbfgs').fit(logistic_train_data, logistic_train_target)

sklearn_paramters = np.concatenate([logisticRegression_sklearn.coef_.flatten(),logisticRegression_sklearn.intercept_])
print(sklearn_paramters)
print(f"Total number of parameters including bias is {len(sklearn_paramters)}")

from sklearn.metrics import average_precision_score

with rOpen('data/validation/validation_input.tif', 'r') as validation_input_ds:
    validation_input_image_ndarray = validation_input_ds.read()  # read all raster values


with rOpen('data/validation/validation_target.tif', 'r') as validation_target_ds:
    validation_target_image_ndarray = validation_target_ds.read()  # read all raster values

validation_input_image_ndarray_2D = np.reshape(validation_input_image_ndarray, (8, -1))
validation_input_2D = (validation_input_image_ndarray_2D - per_channel_min)/(per_channel_max - per_channel_min)
normalised_validation_input_image_ndarray = np.reshape(validation_input_2D, (8, 650, 650))


# rescale the labels to be either 0 or 1 instead of 0 or 255
cleaned_validation_target_image_ndarray = validation_target_image_ndarray.copy()
cleaned_validation_target_image_ndarray[cleaned_validation_target_image_ndarray == 255] = 1


logistic_val_data_before_transpose = np.reshape(normalised_validation_input_image_ndarray, (8, -1))
logistic_val_data = logistic_val_data_before_transpose.T
logistic_val_target = np.reshape(cleaned_validation_target_image_ndarray, (650 * 650))

logistic_val_pred = logisticRegression_sklearn.predict(logistic_val_data)

print("Logistic Regression Sklearn Model validation performance")
print(f"Average validation precision: {average_precision_score(logistic_val_target, logistic_val_pred)}")


with rOpen('data/test/test_input.tif', 'r') as test_input_ds:
    test_input_image_ndarray = test_input_ds.read()  # read all raster values


test_input_image_ndarray_2D = np.reshape(test_input_image_ndarray, (8, -1))
test_input_2D = (test_input_image_ndarray_2D - per_channel_min)/(per_channel_max - per_channel_min)
normalised_test_input_image_ndarray = np.reshape(test_input_2D, (8, 650, 650))

normalised_red_channel_test = normalised_test_input_image_ndarray[4]
normalised_green_channel_test = normalised_test_input_image_ndarray[2]
normalised_blue_channel_test = normalised_test_input_image_ndarray[1]


normalised_rgb_test_ndarray = np.dstack(
    (
        normalised_red_channel_test,
        normalised_green_channel_test,
        normalised_blue_channel_test
    )
)

logistic_test_data_before_transpose = np.reshape(normalised_test_input_image_ndarray, (8, -1))
logistic_test_data = logistic_test_data_before_transpose.T

logistic_test_pred = logisticRegression_sklearn.predict(logistic_test_data)
logistic_test_predicted_imageSklearn = np.reshape(logistic_test_pred, (650, 650))


plt.imshow(normalised_rgb_test_ndarray, cmap="Greys_r")
plt.title("Actual RGB test image ")
plt.axis("off")
plt.show()


plt.imshow(logistic_test_predicted_imageSklearn, cmap="Greys_r")
plt.title("Test predicted output Sklearn")
plt.axis("off")
plt.show()


logistic_test_pred_proba = logisticRegression_sklearn.predict_proba(logistic_test_data)[:,1]
logistic_test_predicted_proba_imageSklearn = np.reshape(logistic_test_pred_proba, (650, 650))


plt.imshow(normalised_rgb_test_ndarray, cmap="Greys_r")
plt.title("Actual RGB test image")
plt.axis("off")
plt.show()

plt.imshow(logistic_test_predicted_proba_imageSklearn, cmap="Greys_r")
plt.title("Test predicted probability output Sklearn")
plt.axis("off")
plt.show()


np.max(logistic_test_pred_proba)

unique_classes, counts = np.unique(logistic_train_target, return_counts=True)

# get the perecentage of values
class_percentages = (counts/np.sum(counts))*100
print(class_percentages)

plt.title("Class distribution of training labels")
plt.xticks([0,1])
plt.hist(logistic_train_target)
plt.show()


plt.figure(figsize=(25, 25))
plt.imshow(normalised_rgb_test_ndarray, cmap="jet")
plt.imshow(logistic_test_predicted_proba_imageSklearn, cmap="jet", alpha=0.8)
plt.show()


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

# !pip install tensorflow

# after running this restart the kernel and move to the next cell and comment the line above


# import the needed libraries
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from keras import backend as K


cnn_model = Sequential()
cnn_model.add( Conv2D(32, kernel_size=(3,3), activation='relu'))
cnn_model.add( Conv2D(1, kernel_size=(3,3), activation='linear'))


# build model
cnn_model_random_weights = Sequential()
cnn_model_random_weights.add( Conv2D(32, kernel_size=(3,3), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform') )
cnn_model_random_weights.add( Conv2D(1, kernel_size=(3,3), activation='linear', kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform') )


# ensure data is represented in a way Keras appreciates (1,650,650,8)
channel_last_normalised_training_input_image_ndarray = np.transpose(normalised_training_input_image_ndarray, (1,2,0))
added_axis_channel_last_normalised_training_input_image_ndarray = np.reshape(channel_last_normalised_training_input_image_ndarray, (1,650,650,8))


# apply model with randomly initialised weights
cnn_model_random_weights(added_axis_channel_last_normalised_training_input_image_ndarray)


print("Helps to emphasise trainable parameter calculation above")
print(cnn_model_random_weights.summary())



