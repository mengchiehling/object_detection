def optimization_process(fn, pbounds: Dict, verbose: int = 2) -> Tuple[Dict, np.ndarray]:

    """
    Bayesian optimization process interface. Returns hyperparameters of machine learning algorithms and the
    corresponding out-of-fold (oof) predictions

    Args:
        fn: functional that will be optimized
        pbounds: a dictionary having the boundary of parameters of fn

    Returns:
        A tuple of dictionary containing optimized hyperparameters and oof-predictions
    """

    optimizer = BayesianOptimization(
        f=fn,
        pbounds=pbounds,
        random_state=1,
        verbose=verbose)

    optimizer.maximize(
        **bayesianOptimization
    )

    if verbose == 0:
        print(f"target: {np.round(optimizer.max['target'], 6)}")

    optimized_parameters = optimizer.max['params']

    _, y_oof = fn(model_stacking=True, **optimized_parameters)

    return optimized_parameters, y_oof


def prepare_bert_input(input_df: pd.DataFrame, label: str = 'is_relevant_human'):

    """
    Preprocessing the input data.

    Args:
        input_df:
            evaluation dataframe containing review info as text stored in title_cleaned and text_cleaned columns
        label:
            Options; target variable to be predicted

    Return:
        A dataframe containing texts used for predictions.
    """

    df = input_df.copy()

    hash_human_labeled = df.index[df[label].isin([0.0, 1.0])]

    # Concatenate title and text
    df = df.loc[hash_human_labeled]

    df['text'] = df['title_cleaned'] + " " + df['text_cleaned']
    df = df[['text', label]]

    # clean text with nltk stop words
    try:
        stop = stopwords.words('german')
    except LookupError as e:
        nltk.download('stopwords')
        stop = stopwords.words('german')

    # df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if ((word not in stop)
    #                                                   and (word.lower() not in stop))]))

    df[label] = df[label].astype(int)

    # Follow the simpletransformers convention
    df.rename(columns={label: 'labels'}, inplace=True)

    return df