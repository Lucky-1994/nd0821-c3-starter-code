# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

RandomForestClassifier with the following parameters:
- n_estimators=100
- max_depth=50 
- min_samples_leaf=2
- min_samples_split=3 
- random_state=42
- other parameters are the defaults values of scikit learn version 1.1.0

## Intended Use

This model can be used to decide if a person is likely to earn above or below 50.000 $. This model was not carefully fine-tuned. Therefore, it should be carefully used.

## Training Data
The census income data set was used to train the model it consists in total of 32561 examples. For the training data were 26048 examples used.

## Evaluation Data

The test data consists of the remaining 6513 examples.

## Metrics

The following metrics were used to analyze the model performance:
- Precision
- Recall
- F1 Score

The following scores were created for the training set:
- Precision: 0.87 		
- Recall: 0.72 	 
- F1: 0.78  

The following scores were created for the test set:
- Precision: 0.78 		
- Recall: 0.63 	 
- F1: 0.7

## Ethical Considerations

It might help to use ML models to get rough ideas about income of people, but there can be a lot of problems resulting from using this model in the wrong places. Therefore, it should be made sure that the results of this model are also checked by humans after wards to ensure there are no concerns resulting from mistakes.

## Caveats and Recommendations

This is just a simple baseline model. If u want to have a model to do this task in a real setting, u should put more effort in the fine-tuning process, but this version can be a good starting point.
