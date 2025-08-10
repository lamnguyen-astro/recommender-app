import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_fscore_support
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from surprise import accuracy, Dataset, KNNBasic, NMF, Reader
from surprise.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras


models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")


def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


def clustering_recommendations(ratings_df, params, user_id, pca=False):
    cluster_no = params.get('cluster_no', 20)
    popularity_threshold = params.get('popularity_threshold', 10)
    ratings_sparse_df = ratings_df.pivot(
        index='user', columns='item', values='rating'
    ).fillna(0).reset_index().rename_axis(index=None, columns=None)
    feature_names = list(ratings_sparse_df.columns[1:])
    # Use StandardScaler to make each feature with mean 0, standard deviation 1
    # Instantiating a StandardScaler object
    scaler = StandardScaler()
    # Standardizing the selected features (feature_names) in the ratings_sparse_df DataFrame
    ratings_sparse_df[feature_names] = scaler.fit_transform(ratings_sparse_df[feature_names])
    features = ratings_sparse_df.loc[:, ratings_sparse_df.columns != 'user']
    user_ids = ratings_sparse_df.loc[:, ratings_sparse_df.columns == 'user']
    km = KMeans(n_clusters=cluster_no)
    if pca:
        n_components = params.get('n_components', 9)
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(features)
        components_df = pd.DataFrame(data=components, columns=['PC' + str(i) for i in range(components.shape[1])])
        features_transformed = pd.merge(user_ids, components_df, left_index=True, right_index=True)
        km.fit(features_transformed)
    else:
        km.fit(features)
    cluster_labels = km.labels_
    # Convert labels to a DataFrame
    labels_df = pd.DataFrame(cluster_labels)
    # Merge user_ids DataFrame with labels DataFrame based on index
    cluster_df = pd.merge(user_ids, labels_df, left_index=True, right_index=True)
    # Rename columns to 'user' and 'cluster'
    cluster_df.columns = ['user', 'cluster']

    users_df = ratings_df[['user', 'item']]
    users_labelled = pd.merge(users_df, cluster_df, left_on='user', right_on='user')
    # Extracting the 'item' and 'cluster' columns from the test_users_labelled DataFrame
    courses_cluster = users_labelled[['item', 'cluster']]
    # Adding a new column 'count' with a value of 1 for each row in the courses_cluster DataFrame
    courses_cluster['count'] = [1] * len(courses_cluster)
    # Grouping the DataFrame by 'cluster' and 'item', aggregating the 'count' column with the sum function,
    # and resetting the index to make the result more readable
    courses_cluster_grouped = courses_cluster.groupby(['cluster', 'item']).agg(
        enrollments=('count', 'sum')).reset_index()
    user_subset = users_labelled[users_labelled['user'] == user_id]
    ## - For each user, first finds its cluster label
    cluster_label = user_subset['cluster'].iloc[0]
    ## - First get all courses belonging to the same cluster and figure out what are the popular ones
    # (such as course enrollments beyond a threshold like 100)
    all_courses = set(courses_cluster[courses_cluster['cluster'] == cluster_label]['item'])
    ## - Get the user's current enrolled courses
    enrolled_course_ids = set(user_subset['item'])
    ## - Check if there are any courses on the popular course list which are new/unseen to the user.
    new_course_ids = all_courses.difference(enrolled_course_ids)
    ## If yes, make those unseen and popular courses as recommendation results for the user
    recommended = (
            (courses_cluster_grouped['cluster'] == cluster_label) &
            (courses_cluster_grouped['item'].isin(new_course_ids)) &
            (courses_cluster_grouped['enrollments'] > popularity_threshold)
    )
    courses_recommended = courses_cluster_grouped.loc[
        recommended, ['item', 'enrollments']
    ].sort_values('enrollments', ascending=False)
    return courses_recommended


def collab_recommendations(ratings_df, params, user_id, algo):
    reader = Reader(rating_scale=(2, 3))
    course_dataset = Dataset.load_from_df(ratings_df, reader=reader)
    trainset, testset = train_test_split(course_dataset, test_size=.3)
    # Create a dictionary to store your recommendation results
    res = {}
    if algo == 'knn':
        # - Define a KNNBasic() model
        k = params.get('k', 40)
        user_based = params.get('user_based', True)
        model = KNNBasic(k=k, sim_option={'user_based': user_based})
    elif algo == 'nmf':
        # - Define a NMF model
        n_factors = params.get('n_factors', 15)
        model = NMF(n_factors=n_factors)
    else:
        return res

    # - Train the model on the trainset, and predict ratings for the testset
    model.fit(trainset)
    predictions = model.test(testset)
    # - Then compute RMSE
    accuracy.rmse(predictions)

    user_ratings = ratings_df[ratings_df['user'] == user_id]
    enrolled_course_ids = user_ratings['item'].to_list()
    all_courses = set(ratings_df['item'])
    unselected_course_ids = all_courses.difference(enrolled_course_ids)

    for unselect_course in unselected_course_ids:
        prediction = model.predict(str(user_id), unselect_course)
        res[unselect_course] = prediction.est
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


class RecommenderNet(keras.Model):
    """
        Neural network model for recommendation.

        This model learns embeddings for users and items, and computes the dot product
        of the user and item embeddings to predict ratings or preferences.

        Attributes:
        - num_users (int): Number of users.
        - num_items (int): Number of items.
        - embedding_size (int): Size of embedding vectors for users and items.
    """
    def __init__(self, num_users, num_items, embedding_size=16, **kwargs):
        """
            Constructor.

            Args:
            - num_users (int): Number of users.
            - num_items (int): Number of items.
            - embedding_size (int): Size of embedding vectors for users and items.
         """
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size

        # Define a user_embedding vector
        # Input dimension is the num_users
        # Output dimension is the embedding size
        # A name for the layer, which helps in identifying the layer within the model.

        self.user_embedding_layer = keras.layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            name='user_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define a user bias layer
        # Bias is applied per user, hence output_dim is set to 1.
        self.user_bias = keras.layers.Embedding(
            input_dim=num_users,
            output_dim=1,
            name="user_bias")

        # Define an item_embedding vector
        # Input dimension is the num_items
        # Output dimension is the embedding size
        self.item_embedding_layer = keras.layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            name='item_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define an item bias layer
        # Bias is applied per item, hence output_dim is set to 1.
        self.item_bias = keras.layers.Embedding(
            input_dim=num_items,
            output_dim=1,
            name="item_bias")

    def call(self, inputs):
        """
            Method called during model fitting.

            Args:
            - inputs (tf.Tensor): Input tensor containing user and item one-hot vectors.

            Returns:
            - tf.Tensor: Output tensor containing predictions.
        """
        # Compute the user embedding vector
        user_vector = self.user_embedding_layer(inputs[:, 0])
        # Compute the user bias
        user_bias = self.user_bias(inputs[:, 0])
        # Compute the item embedding vector
        item_vector = self.item_embedding_layer(inputs[:, 1])
        # Compute the item bias
        item_bias = self.item_bias(inputs[:, 1])
        # Compute dot product of user and item embeddings
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        # Add all the components (including bias)
        x = dot_user_item + user_bias + item_bias
        # Apply ReLU activation function
        return tf.nn.relu(x)


def process_dataset(raw_data):
    """
        Preprocesses the raw dataset by encoding user and item IDs to indices.

        Args:
        - raw_data (DataFrame): Raw dataset containing user, item, and rating information.

        Returns:
        - encoded_data (DataFrame): Processed dataset with user and item IDs encoded as indices.
        - user_idx2id_dict (dict): Dictionary mapping user indices to original user IDs.
        - course_idx2id_dict (dict): Dictionary mapping item indices to original item IDs.
    """

    encoded_data = raw_data.copy() # Make a copy of the raw dataset to avoid modifying the original data.

    # Mapping user ids to indices
    user_list = encoded_data["user"].unique().tolist() # Get unique user IDs from the dataset.
    user_id2idx_dict = {x: i for i, x in enumerate(user_list)} # Create a dictionary mapping user IDs to indices.
    user_idx2id_dict = {i: x for i, x in enumerate(user_list)} # Create a dictionary mapping user indices back to original user IDs.

    # Mapping course ids to indices
    course_list = encoded_data["item"].unique().tolist() # Get unique item (course) IDs from the dataset.
    course_id2idx_dict = {x: i for i, x in enumerate(course_list)} # Create a dictionary mapping item IDs to indices.
    course_idx2id_dict = {i: x for i, x in enumerate(course_list)} # Create a dictionary mapping item indices back to original item IDs.

    # Convert original user ids to idx
    encoded_data["user"] = encoded_data["user"].map(user_id2idx_dict)
    # Convert original course ids to idx
    encoded_data["item"] = encoded_data["item"].map(course_id2idx_dict)
    # Convert rating to int
    encoded_data["rating"] = encoded_data["rating"].values.astype("int")
    # Return the processed dataset and dictionaries mapping indices to original IDs.
    return encoded_data, user_id2idx_dict, user_idx2id_dict, course_id2idx_dict, course_idx2id_dict


def generate_train_test_datasets(dataset, scale=True):
    """
        Splits the dataset into training, validation, and testing sets.

        Args:
        - dataset (DataFrame): Dataset containing user, item, and rating information.
        - scale (bool): Indicates whether to scale the ratings between 0 and 1. Default is True.

       Returns:
        - x_train (array): Features for training set.
        - x_val (array): Features for validation set.
        - x_test (array): Features for testing set.
        - y_train (array): Labels for training set.
        - y_val (array): Labels for validation set.
        - y_test (array): Labels for testing set.
    """

    min_rating = min(dataset["rating"]) # Get the minimum rating from the dataset
    max_rating = max(dataset["rating"]) # Get the maximum rating from the dataset

    dataset = dataset.sample(frac=1) # Shuffle the dataset to ensure randomness
    x = dataset[["user", "item"]].values # Extract features (user and item indices) from the dataset
    if scale:
        # Scale the ratings between 0 and 1 if scale=True
        y = dataset["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    else:
        # Otherwise, use raw ratings
        y = dataset["rating"].values

    # Assuming training on 80% of the data and testing on 10% of the data
    train_indices = int(0.8 * dataset.shape[0])
    test_indices = int(0.9 * dataset.shape[0])
    # Assigning subsets of features and labels for each set
    x_train, x_val, x_test, y_train, y_val, y_test = (
        x[:train_indices], # Training features
        x[train_indices:test_indices], # Validation features
        x[test_indices:], # Testing features
        y[:train_indices], # Training labels
        y[train_indices:test_indices], # Validation labels
        y[test_indices:], # Testing labels
    )
    return x_train, x_val, x_test, y_train, y_val, y_test # Return the training, validation, and testing sets


def predictive_recommendations(ratings_df, params, user_id, algo='tensordot'):
    num_users = len(ratings_df['user'].unique())
    num_items = len(ratings_df['item'].unique())
    embedding_size = params.get('embedding_size', 16)
    # Process the raw dataset using the process_dataset function
    # The function returns three values: encoded_data, user_idx2id_dict, and course_idx2id_dict
    # encoded_data: Processed dataset with user and item IDs encoded as indices
    # user_idx2id_dict: Dictionary mapping user indices to original user IDs
    # course_idx2id_dict: Dictionary mapping item indices to original item IDs
    encoded_data, user_id2idx_dict, user_idx2id_dict, course_id2idx_dict, course_idx2id_dict = process_dataset(
        ratings_df
    )
    x_train, x_val, x_test, y_train, y_val, y_test = generate_train_test_datasets(encoded_data)
    model = RecommenderNet(num_users, num_items, embedding_size)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.losses.MeanSquaredError()],
    )

    ## - call model.fit() to train the model
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=10,
        batch_size=64,
        validation_data=(x_val, y_val),
        verbose=1,
    )
    ### - call model.evaluate() to evaluate the model
    model.evaluate(x_test, y_test)

    user_ratings = ratings_df[ratings_df['user'] == user_id]
    enrolled_course_ids = user_ratings['item'].to_list()
    all_courses = set(ratings_df['item'])
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    unselected_course_idx = [
        course_id2idx_dict[course_id]
        for course_id in unselected_course_ids
        if course_id in course_id2idx_dict
    ]
    user_idx = user_id2idx_dict.get(user_id)
    courses_recommended = pd.DataFrame(
        {
            'user': [user_idx] * len(unselected_course_idx),
            'item': unselected_course_idx
        }
    )

    if algo in ('regression', 'classification'):
        user_emb = model.get_layer('user_embedding_layer').get_weights()[0]
        item_emb = model.get_layer('item_embedding_layer').get_weights()[0]
        # Define column names for user and course embedding features
        u_features = [f"UFeature{i}" for i in range(embedding_size)]  # Assuming there are 16 user embedding features
        c_features = [f"CFeature{i}" for i in range(embedding_size)]  # Assuming there are 16 course embedding features

        user_emb_df = pd.DataFrame(user_emb, columns=u_features)
        user_emb_df['user'] = user_emb_df.index

        item_emb_df = pd.DataFrame(item_emb, columns=c_features)
        item_emb_df['item'] = item_emb_df.index

        # Merge user embedding features
        user_emb_merged = pd.merge(encoded_data, user_emb_df, how='left', left_on='user', right_on='user').fillna(0)
        # Merge course embedding features
        merged_df = pd.merge(user_emb_merged, item_emb_df, how='left', left_on='item', right_on='item').fillna(0)

        # Extract user embedding features
        user_embeddings = merged_df[u_features]
        # Extract course embedding features
        course_embeddings = merged_df[c_features]
        # Extract ratings
        ratings = merged_df['rating']

        # Aggregate the two feature columns using element-wise add
        interaction_dataset = user_embeddings + course_embeddings.values
        # Rename the columns of the resulting DataFrame
        interaction_dataset.columns = [f"Feature{i}" for i in range(16)]  # Assuming there are 16 features
        # Add the 'rating' column from the original DataFrame to the regression dataset
        interaction_dataset["rating"] = ratings

        # Extract features (X) from the interaction_dataset DataFrame
        # Selects all rows and all columns except the last column (features)
        X = interaction_dataset[[f"Feature{i}" for i in range(16)]]

        if algo == 'regression':
            y = interaction_dataset["rating"]
            alpha = params.get("alpha", 10)
            model = Ridge(alpha=alpha)
        else:
            # Extract the target variable (y_raw) from the interaction_dataset DataFrame
            # Selects all rows and only the last column (target variable)
            y_raw = interaction_dataset["rating"]
            # Initialize a LabelEncoder object to encode the target variable
            label_encoder = LabelEncoder()
            # Encode the target variable (y_raw) using the LabelEncoder
            # .values.ravel() converts the target variable to a flattened array before encoding
            # The LabelEncoder fits and transforms the target variable, assigning encoded labels to y
            y = label_encoder.fit_transform(y_raw.values.ravel())
            max_depth = params.get("max_depth")
            model = RandomForestClassifier(max_depth=max_depth)

        X_train, X_test, y_train, y_test = sklearn_train_test_split(X, y, test_size=0.2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if algo == 'regression':
            ### The main evaluation metric is RMSE but you may use other metrics as well
            print(f"RMSE: {mean_squared_error(y_test, y_pred) ** 0.5}")
        else:
            ### The main evaluation metrics could be accuracy, recall, precision, F score, and AUC.
            print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
            print(precision_recall_fscore_support(y_test, y_pred))

        merged_course_recommended = pd.merge(
            courses_recommended, user_emb_df, how='left', left_on='user', right_on='user'
        ).fillna(0)
        merged_course_recommended = pd.merge(
            merged_course_recommended, item_emb_df, how='left', left_on='item', right_on='item'
        ).fillna(0)
        merged_course_recommended[[f"Feature{i}" for i in range(16)]] = (
            merged_course_recommended[u_features] + merged_course_recommended[c_features].values
        )
        prediction = model.predict(merged_course_recommended[[f"Feature{i}" for i in range(16)]])
        if algo == 'classification':
            prediction = label_encoder.inverse_transform(prediction)

        courses_recommended['rating'] = prediction
    elif algo == 'tensordot':
        prediction = model.predict(courses_recommended)
        courses_recommended['rating'] = prediction + 2
    else:
        return pd.DataFrame(columns=['user', 'item', 'rating'])

    courses_recommended['user'] = courses_recommended['user'].map(user_idx2id_dict)
    courses_recommended['item'] = courses_recommended['item'].map(course_idx2id_dict)
    courses_recommended.sort_values('rating', ascending=False, inplace=True)
    return courses_recommended


# Model training
def train(model_name, params):
    pass


# Prediction
def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == models[0]:
            top_courses = params.get('top_courses', 10)
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)

            users = users[:top_courses]
            courses = courses[:top_courses]
            scores = scores[:top_courses]
        # User profile model
        elif model_name == models[1]:
            pass
        # Clustering model
        elif model_name == models[2]:
            top_courses = params.get('top_courses', 10)
            ratings_df = load_ratings()
            courses_recommended = clustering_recommendations(ratings_df, params, user_id)
            courses += courses_recommended['item'][:top_courses].tolist()
            scores += courses_recommended['enrollments'][:top_courses].tolist()
            users += [user_id]*len(courses)
        # Clustering with PCA model
        elif model_name == models[3]:
            top_courses = params.get('top_courses', 10)
            ratings_df = load_ratings()
            courses_recommended = clustering_recommendations(ratings_df, params, user_id, pca=True)
            courses += courses_recommended['item'][:top_courses].tolist()
            scores += courses_recommended['enrollments'][:top_courses].tolist()
            users += [user_id] * len(courses)
        # KNN
        elif model_name == models[4]:
            top_courses = params.get('top_courses', 10)
            ratings_df = load_ratings()
            res = collab_recommendations(ratings_df, params, user_id, 'knn')
            courses += list(res.keys())[:top_courses]
            scores += list(res.values())[:top_courses]
            users += [user_id] * len(courses)
        # NMF
        elif model_name == models[5]:
            top_courses = params.get('top_courses', 10)
            ratings_df = load_ratings()
            res = collab_recommendations(ratings_df, params, user_id, 'nmf')
            courses += list(res.keys())[:top_courses]
            scores += list(res.values())[:top_courses]
            users += [user_id] * len(courses)
        # Neural Network
        elif model_name == models[6]:
            top_courses = params.get('top_courses', 10)
            ratings_df = load_ratings()
            courses_recommended = predictive_recommendations(ratings_df, params, user_id, 'tensordot')
            courses += courses_recommended['item'][:top_courses].tolist()
            scores += courses_recommended['rating'][:top_courses].tolist()
            users += [user_id]*len(courses)
        # Regression with Embedding Features
        elif model_name == models[7]:
            top_courses = params.get('top_courses', 10)
            ratings_df = load_ratings()
            courses_recommended = predictive_recommendations(ratings_df, params, user_id, 'regression')
            courses += courses_recommended['item'][:top_courses].tolist()
            scores += courses_recommended['rating'][:top_courses].tolist()
            users += [user_id]*len(courses)
        # Classification with Embedding Features
        elif model_name == models[8]:
            top_courses = params.get('top_courses', 10)
            ratings_df = load_ratings()
            courses_recommended = predictive_recommendations(ratings_df, params, user_id, 'classification')
            courses += courses_recommended['item'][:top_courses].tolist()
            scores += courses_recommended['rating'][:top_courses].tolist()
            users += [user_id]*len(courses)

    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df
