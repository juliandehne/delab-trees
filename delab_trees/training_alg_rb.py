import tensorflow as tf
from pandas import DataFrame
from sklearn.model_selection import train_test_split
# There seems to be an import bug in Pycharm for keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def train_rb(data: DataFrame, inspect=False) -> DataFrame:
    tw_x, tw_y, tw_test_x, tw_test_y, tw_model = __train_model(data)
    if inspect:
        inspect_model(tw_x, tw_y, tw_test_x, tw_test_y, tw_model)

    tw_non_features = take_non_features(data)
    tw_features_y = take_features(data)
    tw_features = tw_features_y.drop("y", axis=1)
    tw_predictions = tw_model.predict(tw_features)

    tw_vision = tw_non_features.assign(predictions=tw_predictions)
    not_needed_list = ["beam_node_author", "beam_node", "has_followed_path", "has_follow_path", "current"]
    author_vision_result = tw_vision.drop(not_needed_list, axis=1)
    applied_model = author_vision_result.groupby(["conversation_id", "author"]).mean().reset_index()
    return applied_model, tw_model, tw_features


non_feature_list = ["current", "beam_node", "conversation_id", "has_followed_path", "has_follow_path",
                    "beam_node_author", "author"]


def take_features(df, additional_non_features=[]):
    """
    Some utility functions to take the columns that are used as features
    :param df:
    :param additional_non_features:
    :return:
    """
    non_feature_list2 = non_feature_list + additional_non_features
    df = df.drop(non_feature_list2, axis=1)
    return df


def take_non_features(df, additional_non_features=[]):
    non_feature_list2 = non_feature_list + additional_non_features
    column_names = df.columns.values
    feature_list = [column_name for column_name in column_names if column_name not in non_feature_list2]
    df = df.drop(feature_list, axis=1)
    return df


def normalize_timedelta(df):
    # normalize timedelta (put between 0 and 1)
    dt = df.timedelta
    timedelta_normalized = (dt - dt.min()) / (dt.max() - dt.min())
    df = df.assign(timedelta=timedelta_normalized)
    return df


# training functions
def __train_model(df):
    # dropping non-reddit non-twitter data
    df = take_features(df)

    # selecting train and test datasets
    train, test = train_test_split(df, test_size=0.2)
    train.describe()

    # train the model
    y = train.y
    x = train.drop("y", axis=1)
    print(x.shape)
    print(y.shape)

    # import tensorflow and train the model

    print(tf.__version__)
    input_shape = (x.shape[1],)
    model = Sequential([
        Dense(1, activation='sigmoid', input_shape=input_shape)
    ])

    # stochastic gradient descend as a classifier seem appropriate
    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy', 'mae']
    )

    # model.fit(x, y, epochs=3)
    model.fit(x, y)
    # evaluate the model on the test set
    test_y = test.y
    test_x = test.drop("y", axis=1)

    loss, accuracy, mae = model.evaluate(test_x, test_y)
    print("the accuracy on the training set is {} and the mae is {}".format(accuracy, mae))

    return x, y, test_x, test_y, model


def inspect_model(x, y, test_x, test_y, model):
    # have a look at some prediction
    reply_distance_2 = test_x[test_x["reply_distance_2"] == 1]
    first_rows = reply_distance_2.head(2)
    print(first_rows)
    model.predict(first_rows)

    # let's have a look at the weights and biases of the hidden layer
    first_layer_weights = model.layers[0].get_weights()[0]
    first_layer_biases = model.layers[0].get_weights()[1]
    # print(first_layer_weights)
    column_names = x.columns.values
    for i in range(len(column_names[:5])):
        print("feature {} has weight {} \n".format(column_names[i], first_layer_weights[i]))
