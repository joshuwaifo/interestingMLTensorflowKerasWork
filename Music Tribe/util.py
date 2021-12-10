# import libraries
import os, json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def dataLoadingUtil():

    # initialise list of json files across all categories
    json_files = [] 

    # store categories for later as a list of strings
    path='music_tribe_ml_test/data/'
    categories = [ item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item)) ]


    # obtain the paths to all the json files across categories
    for category in categories:
        # this finds our json files in a given category
        # (sub)folders are seperated by '/'
        path_to_json = 'music_tribe_ml_test/data/'+category+'/'


        # list files in path
        # check if file ends with a .json extension
        # if so add to list
        category_files = [category+'/'+json_file for json_file in os.listdir(path_to_json) if json_file.endswith('.json')]

        # add the elements of category files to the end of json files
        json_files.extend(category_files) 

    # features and label columns
    featuresLabel = ['centroid', 'end', 'energy', 'flatness', 'flux', 'spectralComplexity', 'start', 'zeroCrossingRate', 
                     'onsetTime', 'gfcc', 'lpc', 'mfcc', 'category']

    # here I define my pandas Dataframe with the columns I want to get from the json
    jsons_data = pd.DataFrame(columns = featuresLabel)
    path_to_dataFolder = 'music_tribe_ml_test/data/'
    index = 0 # sample number index
    for js in json_files:

        # join path and js together and open
        with open(os.path.join(path_to_dataFolder, js)) as json_file:
            # get the first part of the json file name 
            category = js.split('/')[0] 
            json_text = json.load(json_file)

            # counter to retrieve corresponding onset time for sample
            onsetCount = 0 
            for name in json_text: 
                if "sample_" in name:                
                    centroid = json_text[name]['centroid'][0]
                    end = json_text[name]['end'][0]
                    energy = json_text[name]['energy'][0]
                    flatness = json_text[name]['flatness'][0]
                    flux = json_text[name]['flux'][0]
                    spectralComplexity = json_text[name]['spectralComplexity'][0]
                    start = json_text[name]['start'][0]
                    zeroCrossingRate = json_text[name]['zeroCrossingRate'][0]
                    onsetTime = json_text['onsetTimes'][0][onsetCount]
                    # increment onsetCount
                    onsetCount += 1

                    gfcc = json_text[name]['gfcc'][0]
                    lpc = json_text[name]['lpc'][0]
                    mfcc = json_text[name]['mfcc'][0]

                    # here I push a list of data into a pandas DataFrame at row given by 'index'
                    jsons_data.loc[index] = [ centroid, end, energy, flatness, flux, spectralComplexity, start, 
                                             zeroCrossingRate, onsetTime, gfcc, lpc, mfcc, category ]
                    index += 1 # increment sample index counter


    uncleanDataMatrix = pd.DataFrame(jsons_data) # unclean matrix as list still present in the gfcc, lpc and mfcc columns   

    # expand out gfcc, lpc and mfcc into their own dataframes
    gfcc = uncleanDataMatrix['gfcc'].apply(pd.Series)
    lpc = uncleanDataMatrix['lpc'].apply(pd.Series)
    mfcc = uncleanDataMatrix['mfcc'].apply(pd.Series)


    # rename each variable ie gfcc_0, gfcc_1, ...
    gfcc = gfcc.rename(columns = lambda x : 'gfcc_' + str(x))
    lpc = lpc.rename(columns = lambda x : 'lpc_' + str(x))
    mfcc = mfcc.rename(columns = lambda x : 'mfcc_' + str(x))

    # remove gfcc, lpc and mfcc columns
    uncleanDataMatrix = uncleanDataMatrix.drop(['gfcc', 'lpc', 'mfcc'], axis = 1) 

    # concatenate matrices into a final cleaned data matrix
    dataMatrix = pd.concat([uncleanDataMatrix, gfcc, lpc, mfcc], axis=1) 

    # obtain features for X, labels for y
    # features: pandas dataframe object containing all attributes except the label 'category'
    features = dataMatrix.drop('category', axis=1) 
    features = features.astype(float) # all features can be and are converted into floats


    # Pandas series object, labels containing only the label 'category'
    labels = dataMatrix['category']

    # use object to retrieve corresponding categories later
    encodeObj = LabelEncoder().fit(labels)

    # Categorical attributes are encoded to numeric categories [0,1,2,...,8] in the form of a numpy array, using sklearn's label encoder function
    numeric_labels = encodeObj.transform(labels)

    # categoric_labels = encode.inverse_transform(numeric_labels)
    categoric_labels = labels # above is how to reverse transformation

    # Here the pandas dataframe is converted to a numpy array
    X = features.values # shape: 8869 samples, 46 features
    y = numeric_labels # shape: 8869 samples
    
    return X,y, features, labels, encodeObj