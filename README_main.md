<div align=”center”>

# MODULE 3 - PROJECT 

# Expanding Machine Learning into Formation Evaluation

</div>

## Introduction

Formation Evaluation has played an important role in many industries including but not limited to oil & gas, geothermal, and mining. These industries for long have utilized large and complex data to analyze and make their predictions. This analysis is especially important in the exploration stage of most processes. Till now, certain tools and techniques have been used across various industries to deal with big data. However these processes are cumbersome and is difficult to achieve utmost accuracy while also speeding up the process. 
> This is where the demand of data science rises in hopes of developing a tool/model that can accurately fill in the gaps.

One aspect of Formation evaluation is determining the facies of a certain rock formation based on the log readings measured during the exploration process. Facies determine the properties of that formation, the result of which is important to determine the presence of oil/gas/water, geothermal properties, etc. The most common logs used for facies determination are Gamma Ray, Resistivity, Neutron Density, and photoelectric.
>Different facies show different values for each log measurement.

Thus, analyzing the combined values will help classify each data point at a given depth and thus determine the facies at that depth.
>The final model from this project will help accurately classify each point at a given depth into different facies that would further help to separate the formation into layers and aid the exploration process.


## The Data

Our data consist of information gathered from 8 different wells and 3232 data points. We have 11 coulumns including the target variable.

The target variable has 9 classes numbered 1-9, each denoting a facies. These are described as below:

>**1** : Non-Marine Sandstone <br /> 
**2**: Non-Marine Coarse Siltstone <br />
**3**: Non-Marine Fine Siltstone <br />
**4**: Marine siltstone and shale <br />
**5**: Mudstone (Limestone) <br />
**6**: Wackestone (Limestone) <br />
**7**: Dolomite <br />
**8**: Packstone - Grainstone (Limestone) <br />
**9**: Phylloid-algal bafflestone (Limestone) <br />

The predictors used are the log reading acquired during logging operations. 
>The logs included in this data are: <br />
- Gamma Ray (GR)
- Resistivity (ILD_log10)
- Photoelectric Effect (PE)
- Neutron-Density Porosity Difference (DeltaPHI)
- Neutron-Density Porosity (PHID) <br />

Other Predictors include Depth, Nonmarine-Marine Indicator (NM_M), and relative position.

Formation and well names have also been included for the knowledge of geographical location. This will ensure the extension of this project to other geographical locations.


## Methodology

Most of the data that we get from the field require cleaning, processing and noise elimination. This Project utilizes scikit learn's preprocessing classes to accomplish this. The data is split into training and testing set to be able to test the final model. The training data is further divided into training and validation to test each model and leave the test data un-biased.

Once this is accomplished, the data is fit through five differnt models and compared on the basis of f1 score. The errors for each class and the macro-averaged f1-score is analysed to select the final model.

Lastly, feature analyses is performed based on the feature ranking, importances and weights.

The libraries used in this project include scikit learn, pandas, numpy, yellowbrick, eli5, matplotlib, seaborn, ipywidgets



## Functions Used to Build this Project

1. Column Exploration

```
def column_type_exploration(df, column):
    
    '''For a given column in the given dataframe, displays an output of number of unique values and
    statistical summary for number type
    -----------------------------------------------
    Input:
    df (DataFrame): DataFrame for the column to evaluate
    column (str): column name
    -----------------------------------------------
    Output:
    statistical summary of the numerical column
    unique value counts'''
    
    if df[column].dtype=='int64' or df[column].dtype=='float64':
        display(df[column].describe())
    
    print('\n')
    print(f'Number of Unique Values: {len(df[column].unique())}')
    print('\n')
    print(f'Example Unique Values: {df[column].unique()[:5]}')
```

2. Preprocessing

```
def preprocessing_trial(num_cols, cat_cols,
                  cat_imputer=KNNImputer(weights='distance'), 
                  encoder=OneHotEncoder(sparse=False, drop='first',handle_unknown='error'), 
                  num_imputer=KNNImputer(weights='distance'), 
                  transformation=PowerTransformer()):
    
    '''Builds a preprocessing pipeline and column transformation to data containing numerical and/or categorical data 
    based on the chosen classes/preprocessing methods
    --------------------------------
    Inputs:
    
    cat_imputer (class): Imputer class for categorical data. Default - KNNImputer() with weights as distance
    encoder (class): encoding class for categorical data. Default - OneHotEncoder() ignoring the unknowns and dropping first
    num_imputer (class): Imputer class for numerical data. Default - KNNImputer() with weights as distance
    transformation (class): linear scaling or non-linear transformation class. Default - PowerTransformer()
    num_cols (list): numerical columns
    cat_cols (list): categorical columns
    --------------------------------
    Output:
    
    ColumnTransformer pipeline to preprocess a given data
    --------------------------------'''
    
    cat_transformer = Pipeline(steps=[('ohe', encoder),
                                      ('impute', cat_imputer)])
    
    num_transformer = Pipeline(steps=[('impute', num_imputer),
                                  ('scaler', transformation)])
    
    preprocessing = ColumnTransformer(transformers=[('num', num_transformer,num_cols),
                                                ('cat', cat_transformer,cat_cols)])
    return preprocessing
```

3. modeling pipeline

```
def model_pipeline(model,preprocessor):
    
    '''Returns a model object for a given estimator and preprocessor
    ------------------------------------------
    Inputs:
    
    model (sklearn classifier class)
    preprocessor (pipeline or class): sklearn preprocessing class or pipeline
    ------------------------------------------
    
    Outputs:
    
    sklearn modelling pipeline
    -------------------------------------------'''
    
    model = Pipeline(steps = [('preprocessor', preprocessor),
                              ('model', model)])
    return model
```

4. Model Evaluation

```
def model_evaluation(model,X,y,cm_normalize = 'true', cm_cmap = 'BuGn_r' ):
    '''Displays classification matrix and visual evaluation (confusion matrix) for a given scikit learn model
    for test data
    ------------------------------
    Inputs:
    
    model (sklearn model)
    X (DataFrame, series or array): test data for features
    y (series or array): dependent variable
    ------------------------------
    
    Outputs:
    
    sklearn confusion matrix (DataFrame)
    sklearn confusion matrix plot
    ------------------------------'''
    
    print('\n')
    print(formating.bold + formating.underline+ formating.blue + 'MODEL EVALUATION' + formating.normal)
    print('\n')
    
    #classification report
    
    print(formating.bold + formating.underline + 'Classification Report' + formating.normal)
#     print('\n')
    
    y_hat = model.predict(X)
    display(pd.DataFrame(metrics.classification_report(y,y_hat,output_dict=True)))
    
    
    #Visual Separation
    
    print('\n')
    print('--'*20)
    print('\n')
    
    
    #Visual Evaluation
    
    print(formating.bold + formating.underline + 'Visual Evaluation' + formating.normal)
#     print('\n')
    
    fig,axes = plt.subplots(figsize = (7,7))
#     axes = axes.flatten()
    
    metrics.plot_confusion_matrix(model,X,y,normalize = cm_normalize, cmap = cm_cmap,ax=axes)#, ax=axes[0])
    axes.set_title('Confusion Matrix')
    
#     metrics.plot_roc_curve(model, X,y, ax=axes[1])
#     axes[1].set_title('ROC Curve')
#     axes[1].legend()
#     axes[1].plot([0,1],[0,1], ls = ':')
    
    fig.tight_layout()
#     plt.show()
```

5. Model Visuals

```
def model_visuals(model,X,y,X_test,y_test,num_cols,cat_cols,labels):
    
    '''Yellowbrick visual analysis of the model. Includes Class prediction error, feature correlation,
    and Feature ranks
    --------------------------------------
    Input:
    
    model (sklearn model)
    X (DataFrame or array): Training features data
    y (series or array): Training Target
    X_test (DataFrame or array): Testing features data
    y_test (series or array): Testing Target
    num_cols (array or list): numerical columns
    cat_cols (array or list): categorical columns
    labels (array or list): feature names
    --------------------------------------
    Output:
    
    Class prediction plot
    feature correlation plots, with and without mutual information
    feature ranks
    '''
    
    #Class Prediction Error
    
    print('\n')
    print(formating.bold+formating.underline+formating.green+'CLASS PREDICTION ERROR'+formating.normal+'\n')
    class_prediction_error(model,X, y, X_test, y_test);
    
    print('\n')
    print('--'*50)
    print('\n')
    
    #Feature Correlation with target variable
    
    print(formating.bold+formating.underline+formating.green+'FEATURE CORRELATION WITH TARGET'+formating.normal+'\n')
    
    fig,axes = plt.subplots(ncols=2,figsize=(15,6))
    
    visualize = FeatureCorrelation(sort = True,ax = axes[0],labels = labels)
    X_train_feature = preprocessing_trial(numerical,categorical).fit_transform(X)
    visualize.fit(X_train_feature,y)
    visualize.show();
    
    visualize2 = FeatureCorrelation(method='mutual_info-classification',sort = True,ax = axes[1],labels = labels)
    X_train_feature = preprocessing_trial(num_cols,cat_cols).fit_transform(X)
    visualize2.fit(X_train_feature,y)
    visualize2.show();
    
    print('\n')
    print('--'*50)
    print('\n')
    
    #Rank Features 1d and 2d
    
    print(formating.bold+formating.underline+formating.green+'RANK FEATURES'+formating.normal+'\n')
    
    fig,axes = plt.subplots(ncols=2, figsize=(15,6))
    rank1d(X_train_feature, ax=axes[0], show=False, features = cols_j)
    rank2d(X_train_feature, ax=axes[1], show=False,features = cols_j)
    plt.show();
```

6. Feature Importances

```
def feature_importances(model_classifier,preprocessed_X,y):
    
    '''Feature importances relative and actual
    ----------------------------
    Input:
    
    model_classifier (sklearn classifier)
    preprocessed_X (Data_frame or array): Preprosessed training X
    y (array, series): target variable
    ----------------------------
    Output:
    
    Relative and actual feature importances plots
    -----------------------------
    '''
    
    print(formating.bold+formating.underline+formating.green+'FEATURE IMPORTANCES'+formating.normal+'\n')
    
    try:
        fig,ax=plt.subplots(figsize=(7,6))
        viz = FeatureImportances(model_classifier,ax=ax)
        viz.fit(preprocessed_X,y_train_model)
        viz.show();
        fig,ax1=plt.subplots(figsize=(7,6))
        viz2 = FeatureImportances(model_classifier,relative=False,ax=ax1)
        viz2.fit(preprocessed_X,y_train_model)
        viz2.show();
    except:
        print('Feature Importances is not compatable with this classifier')
```

7. Explained Weights

```
def explained_weights(model_classifier,feature_names=cols_j,top=30):
    
    '''Table of explained weights of the model features
    ---------------------------
    Input:
    
    model_classifier (sklearn classifier)
    feature_names (series, array, list): Feature names
    top (int): total number of features to display
    ---------------------------
    Output:
    
    Table of feature weights color coded
    ---------------------------'''
    
    return eli5.explain_weights(model_classifier,feature_names = feature_names,top = top)
```

8. model comparison on scores

```
def score_compare(models, X_test, y_test, parameters,parameter_name, target_variable,
                  classification_score = 'f1-score', palette='mako'):
    
    '''Compares and plots the average score of the models
    ------------------------------------
    Inputs:
    
    models (sklearn model): models that are to be compared
    X_test (DataFrame or array): Test features data
    y_test (array or series): target test data
    parameters (list or array): parameters on the basis of which comparison is made
    parameter_name (str): name of the parameters on the basis of which comparison is made. This is
    used to name the column.
    target_variable (str)
    classification_score (str): score on which comparison is performed
    pallette (str or seaborn color pallette): plot color pallette
    ------------------------------------
    Output:
    
    bar plot comparing these scores
    ------------------------------------'''
    
    #Dictionary of all scores
    
    scores = {}
    for i in range(len(parameters)):
        y_hat = models[i].predict(X_test)
        report = pd.DataFrame(metrics.classification_report(y_test,y_hat,output_dict=True))
        scores[parameters[i]]=report.loc[classification_score]
    
    #table
    
    f1_table = pd.DataFrame(scores).reset_index()
    f1_table = f1_table.rename(columns = {'index':target_variable})
    f1_table_formation = f1_table.loc[:(len(f1_table)-4)]
    
    table = pd.melt(f1_table_formation, id_vars=[target_variable],
                    value_name=classification_score, var_name = parameter_name)
    
    #plot
    
    fig,axes = plt.subplots(figsize = (10,6))
    sns.barplot(target_variable, classification_score, hue = parameter_name, data = table, palette=palette, ax=axes);
    axes.set_title('Individual Analysis')
```

9. F1 score comparison

```
def f1_compare(models, X_test, y_test, parameters,average='macro'):
    
    '''Compares average f1 score of the models
    ------------------------------------
    Inputs:
    
    models (sklearn model): models that are to be compared
    X_test (DataFrame or array): Test features data
    y_test (array or series): target test data
    parameters (list or array): parameters on the basis of which comparison is made
    pallette (str or seaborn color pallette): plot color pallette
    ------------------------------------
    Output:
    
    dictionary of the avg score for each model
    ------------------------------------'''
    
    print('\n'+formating.underline+f'Average f1_score'+'\n')
    
    f1_macro = {}
    for i in range(len(parameters)):
        y_hat_ = models[i].predict(X_test)
        f1 = round(metrics.f1_score(y_test,y_hat_,average=average),5)
        f1_macro[parameters[i]]=f1
    
    return f1_macro
```


## Data Splitting and Preprocessing


### Data Splitting

Most of the supervised learning models rely on and learn from the given target variables. Therefore, it becomes paramount to seperate out and keep aside a part of data that you could test to analyse the model performance. The train-test split was performed using he default 80-20 split.

Further, since we are analysing multiple models, using test data multiple times could aslo risk bias since the data would already be exposed during the execution. So, it is wise to keep our test data aside and further split-our training data into two. The train-validation split was also performed using the default 80-20 split.

Once the split has been performed, we test for any imblance. The image below shows the class imbalance before the split, and after. Since we have similar proportions, we are good to proceed.

**Class Imbalance before the split**
img:CI_values_before, CI_Before

**Class Imbalance after the split**
img:CI_values_after, CI_after


### Data Preprocessing

The property of a rock formation is such that they are layered. Which means, that with relation to depth, the log values would be similar to the neighboring data points since logs determine the rock/formation properties. For this reason, all missing values are set to be filled using K-Nearest Neighbors weighed on distance.

Further, since most of our data is partially skewed, the numerical data is scaled on the basis of the column's medians using Scikit Learn's `RhobustScaler()`. Power and Quantile Transformers are also explored. Scaling is performed to be able to determine the relative relation with the target variable and make the interpretation easier.

To make the workflow more efficient, pipelines were created to be used for all preprocessing:

```
cat_transformer = Pipeline(steps=[('impute', KNNImputer(weights='distance')),
                                  ('ohe', OneHotEncoder(sparse=False, drop='first',handle_unknown='ignore'))])

num_transformer = Pipeline(steps=[('impute', KNNImputer(weights='distance')),
                                  ('scaler', RobustScaler())])

preprocessing = ColumnTransformer(transformers=[('num', num_transformer,numerical),
                                                ('cat', cat_transformer,categorical)])
```

A final pipeline is created and built in a function to incorporate the entire modeling process in one line of code:

```
def model_pipeline(model,preprocessor):
    
    '''Returns a model object for a given estimator and preprocessor
    ------------------------------------------
    Inputs:
    
    model (sklearn classifier class)
    preprocessor (pipeline or class): sklearn preprocessing class or pipeline
    ------------------------------------------
    
    Outputs:
    
    sklearn modelling pipeline
    -------------------------------------------'''
    
    model = Pipeline(steps = [('preprocessor', preprocessor),
                              ('model', model)])
    return model
```

## Comparing the Different Models

### KNN Model

Since the nature of the dataset is such that most of the immediate data points in depth have the same characteristics, except on the boundaries, our best first approach could be using the KNN model. Scikit Learn's `KNeighbourClassifier()` was used for the purpose. The best KNN model was seleced using gridsearch on parameters `n_neighbours` and `weight`. The f1-macro-average score was 0.82.

Looking at the individual errors, Facies 1, 5 and 9 seem to perform well with less errors (low false negatives) Facies 2 seem so show some errors while separating from 3 since both these facies share some similar properties. Similarly, 6 and 8 seem to show some mix ups. Facies 4 seem to have a few false negatives belonging to 6 and 8. A possible explaination of this relies on the size of the grains of these formations. Some of the logs determine porosity of the formation which depends largly on brain size and volumes. 

img:KNN_matrix, KNN_error

Feature correlation using KNN show that Marine-Non-Marine, PE log values, one of the formations, resistivity log, and depth have high positive influence on the classification inthat respective order, while Gamma ray values, N-D porosity, Delta N-D negatively influence the classification.

img: feature_corr_KNN

Shapiro feature ranking shows the importance of these features to lie in the line of Resistivity N-D Porosity having the greatest influence regardless of the direction, followed by depth, GR, PE, and Relative position.

ing: KNN_shapiro

### Decision Tree Classifier

Next we explore Desision Tree Classification using scikit Learn's DecisionTreeClassifier() with a gridsearch on parameters `crierion`, `min_samples_leaf`, and `class_weights`. The final decision tree model also took into consideration scaling or transforming method on the numerical data. The f1-macro-average score was 0.73.

The results of this model showed similar trends in individual errors but larger in value than the KNN Model.

img: error_DT, DT_matrix

Feature correlation and and shapiro ranking proved to be exactly the same as KNN Model.

Img: feature_corr_DT, DT_shapiro

eli15's explained weights show a slightly different feaure ranking in a way that the heighest influenceer was Marine-Non-Marine, followed by Depth, resistivity, relative position, PE and GR.

img: eli5_DT

Scikit Learn's feature importances show the same results. Based on this, we would go forward and use these results for feature ranking and importances.

img: DT_importances

### Random Forest Classifier

The results from the scikit Learn's RandomForestClassifier() show a similar trend in errors as KNN and Decision tree, however it shows an improvement from the Desision tree classifier. The f1-macro-average score was 0.81

img:error_RF, RF_matrix

While the explained weights, and shapiro ranking show the same ranks in features, the the result of feature importances slightly changed to Marine-Non Marine still being on the top followed by PE, GR, Depth, Rsisitivity, N-D porosity, Delta N-D, and relative porosity.

img: all other

### SVM

Since SVM also uses distances in their calculation, this might give better results as well. The f1-macro-average score was 0.75

While other classes do relatively well, 2, 8, and 6 do not perform that well.

Feature correlation and shapiro importances remain the same.

### Stacking Classifier

The final classifier used was the stacking classifier which would take the predictions of all the models above and predict accordingly. The f1-macro-average score was 0.82.

The individual errors were moderate, with more errors in class 3 and 6.



## Final Model and Test Results

Based on the f1-macro-average, Knn performs heighest with 0.82 score followed by Stacking classifier 0.81.

Test data was fit in these models and both performed at an f1score of 0.85.

The KNn Model performs fairly on layers 2,3,6,8 with f1 scores of 0.83, 0.86, 0.83,0.75.

img: KNN_test_error, KNN_test_matrix

The Stacking classifier model performs well with 2, moderately for 3 and fairly on 6 and 8 with scores of 0.88, 0.91, 0.79, and 0.82 respectively.

img: SC,SC

Therefore, our Final model selected was the **Stacking Classifier**. The time taken to execute this model on test data with 808 data points was about 11 secs!


## Feature Ranking and Importances

Random Forest classifier results were used for feature importances and ranking. Please refer to the Random Forest classifier results and analysis.

The plots below show a relation between these features and the facies classification.

Marine non Marine plot show that each classification accurately corresponds to fixed facies in the mix.

img:

The plot below shows the trends for each log plot corresponding to a particular facies.

img:

img:closeup


## Conclusion and Recommendations

- The best model took about 11 secs to classify the data into separate facies with aa f-1 score of. 

- Using a Machine learning model can thus prove to be a very efficient and speedy method for facies classification as compared to the combursome manual techniques currently used which may take days to generate the results.

-The most important features that determine the accurate classification include the property of being marine or non-marine, and the log values generated from PE, GR, N-D logs, Resistivity and relative position along with depth. These show great influence since each of these values are unique to certain properties which define a facies.

- It is paramount, therefore, that these operations and data preprocessing is conducted meticulously before feeding in the data into the machine learning model.


## Future Work

Future work will include but not limited to:

- Further improving the model to include other methods of distance calculation since distance is proved to be a major factor in the results.

- Analyse the effect of class imbalance to further improve our model.

- Expand and test this model for wells at different geological locations with other facies present to make this model applicable globally

- Use these predictions as a feature in machine learning models to predict the main goal of facies classification.
