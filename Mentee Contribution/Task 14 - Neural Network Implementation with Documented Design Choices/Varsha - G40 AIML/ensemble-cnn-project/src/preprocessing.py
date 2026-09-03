from sklearn.model_selection import train_test_split


def preprocess_data(x_train, y_train, x_test, y_test):

    # Normalize pixel values from 0-255 to 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Convert labels from shape (n, 1) to (n,)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Split training data into training and validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.1765,
        random_state=42,
        stratify=y_train
    )

    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test
    )