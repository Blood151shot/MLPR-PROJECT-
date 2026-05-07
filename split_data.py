from sklearn.model_selection import train_test_split
from predict import classify

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    y_labels = [classify(v) for v in y]

    X_train, X_temp, y_train, y_temp, ylab_train, ylab_temp = train_test_split(
        X, y, y_labels,
        test_size=(test_size + val_size),
        stratify=y_labels,
        random_state=random_state
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=ylab_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test