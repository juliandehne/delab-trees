import pandas as pd
from keras import Sequential
from keras.layers import Dense
from pandas import DataFrame

from delab_trees.training_alg_rb import take_features
from sklearn.model_selection import train_test_split

"""
# Author Prediction

It is possible to predict an author or "new author" at same time by defining categories as 1 if a author is to be predicted but
only if it is not a new author. 

The probability of predicting an author is calculated for each relationship (root distance to another node, reply distance to other nodes, and reply distance to nodes with the same author. In future also the author follower network will be included in the feature set.

The overall sum of the probability of predicting an author (in average) will be interpreted as the likelihood of any author writing in any time in the conversation (again, because it is not a new author). This will then seen as the author being present in the conversation because it is another measure of a author being available in all branches and positions in the conversation.

#### Create a one hot vector representation of the possible authors
- create an artificial user that represents a new user in a conversation up to that point
- get a matrix with the authors as columns and a 1 if the author wrote the post
- join it with the feature matrix
- drop the author column


#### Training NN to predict the author that would write next
- included a "new author" category to capture predicting unknown authors
- using multi-class classification (instead of multi-label)
- relu/sigmoid activation functions have same effect
- precision grew significantly when adding more than 3-5 layers

#### Predicting the author presence based on prediction probabilities
- compute predictions for the whole dataframe
- drop features and non-features except conversation and platform
- wide to long the authors to make them a index
- groupby conversation and platform

#### Notes
- inserting the new author column increased precision times 10
- categorical accuracy and regular accuracy match (which is weird)
"""


def train_pb(data: DataFrame):
    prediction_result = calculate_author_predictions(data)
    result = prediction_result.groupby(["conversation_id", "author"]).mean().reset_index()
    return result


def calculate_author_predictions(df):
    # compute a fake user that symbolizes that the given user has not been seen at a given stage in the conversation
    df_conversation_authors = df[["conversation_id", "author", "current_time"]]
    first_times = df_conversation_authors.groupby(["conversation_id", "author"]).min()

    def is_new_author(row):
        earliest_author_post = first_times.loc[row["conversation_id"], row["author"]]
        current_post_time = row["current_time"]
        return earliest_author_post >= current_post_time

    new_author_column = df[["conversation_id", "author", "current_time"]].apply(is_new_author, axis=1)
    new_author_column = new_author_column.rename(columns={'current_time': "Author_is_new"})
    new_author_column.value_counts()

    def compute_new_author_column(df):
        import pandas as pd
        author_one_hot = pd.get_dummies(df.author, prefix="Author", sparse=True)
        # make author cells 0 that are now represented as "new author"
        author_one_hot = author_one_hot.astype(bool).apply(lambda x: x & ~new_author_column.Author_is_new).astype(int)
        # delete columns that are all 0
        author_one_hot = author_one_hot.loc[:, (author_one_hot != 0).any(axis=0)]
        # join the new author column to the labels
        labels = author_one_hot.join(new_author_column.astype(int))
        features = take_features(df, ["current_time", "beam_node_time"])
        combined_set = features.join(labels)
        return combined_set, features, labels

    combined_set, features, labels = compute_new_author_column(df)

    # from keras.optimizer_v2.rmsprop import RMSprop
    # selecting train and test datasets
    train, test = train_test_split(combined_set, test_size=0.2, shuffle=False)
    print("split training and test set")

    # train the model
    y = train.drop(features.columns, axis=1)
    x = train.drop(labels.columns, axis=1)
    print("seperated features and y with shapes:")
    print(x.shape)
    print(y.shape)

    # import tensorflow and train the model
    # print(tf.__version__)
    input_shape = (x.shape[1],)
    output_shape = y.shape[1]
    print("inputshape is {}".format(input_shape))
    model = Sequential([
        Dense(output_shape, activation='relu', input_shape=input_shape),
        Dense(output_shape, activation='relu', input_shape=input_shape),
        Dense(output_shape, activation='relu', input_shape=input_shape),
        Dense(output_shape, activation='softmax', input_shape=input_shape)
    ])
    print("defined model as {}".format(model.layers))
    # stochastic gradient descend as a classifier seem appropriate
    model.compile(
        optimizer="rmsprop",
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy', 'accuracy', 'mae']
    )
    print("compiled model")
    # model.fit(x, y, epochs=3)
    model.fit(x, y)
    # model.fit(x, y, epochs=10, shuffle=True)
    # evaluate the model on the test set
    test_y = test.drop(features.columns, axis=1)
    test_x = test.drop(labels.columns, axis=1)
    # test_x = test_x.drop("timedelta", axis=1)

    loss, cat_accuracy, accuracy, mae = model.evaluate(test_x, test_y)
    print("the accuracy on the training set is cat acc {}, reg acc {} and the mae is {}".format(cat_accuracy, accuracy,
                                                                                                mae))

    all_features = take_features(df, ["current_time", "beam_node_time"])
    print("start generating author predictions for the whole data set")
    predictions = model.predict(all_features, use_multiprocessing=True)
    print("end generating author predictions for the whole data set")
    column_names = labels.columns
    predictions = pd.DataFrame(predictions, columns=column_names)
    print(type(predictions))
    print(predictions.shape)

    all_non_features = df[["conversation_id"]]
    print(type(all_non_features))
    print(all_non_features.shape)
    all_non_features.reset_index(drop=True, inplace=True)
    joined_dataframe = all_non_features.join(predictions)
    # print(joined_dataframe.Author_is_new.describe()) # no idea why that is the same prediction of all the rows

    joined_dataframe = joined_dataframe.groupby("conversation_id").mean()
    author_predictions_existing = joined_dataframe.drop(["Author_is_new"], axis=1)
    author_predictions_existing.reset_index(level="conversation_id", inplace=True)
    print("start converting author hot vectors back to one author column")
    # author_predictions_existing_reshaped = pd.wide_to_long(author_predictions_existing, stubnames="Author_",
    #                                                       i='conversation_id', j="author_id")
    author_predictions_existing_reshaped = pd.wide_to_long(author_predictions_existing,
                                                           stubnames="Author",
                                                           i="conversation_id", j="author_id", sep="_")
    # TODO: Figure out why the wide to long is not working for only one column
    assert author_predictions_existing_reshaped.empty is False, "Not enough data in order to compute pb values!"

    author_predictions_existing_reshaped = author_predictions_existing_reshaped.rename(
        columns={"Author": "predictions"})
    return author_predictions_existing_reshaped
