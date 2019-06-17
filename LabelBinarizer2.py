import numpy as np
from sklearn.preprocessing import LabelBinarizer

'''
For two classes LabelBinarizer returns only one column.
If explicitly two columns are desired, this workaround must be used
'''

class LabelBinarizer2:

    def __init__(self):
        self.lb = LabelBinarizer()

    def fit(self, X):
        # Convert X to array
        X = np.array(X)
        # Fit X using the LabelBinarizer object
        self.lb.fit(X)
        # Save the classes
        self.classes_ = self.lb.classes_

    def fit_transform(self, X):
        # Convert X to array
        X = np.array(X)
        # Fit + transform X using the LabelBinarizer object
        Xlb = self.lb.fit_transform(X)
        # Save the classes
        self.classes_ = self.lb.classes_
        if len(self.classes_) == 2:
            Xlb = np.hstack((Xlb, 1 - Xlb))
        return Xlb

    def transform(self, X):
        # Convert X to array
        X = np.array(X)
        # Transform X using the LabelBinarizer object
        Xlb = self.lb.transform(X)
        if len(self.classes_) == 2:
            Xlb = np.hstack((Xlb, 1 - Xlb))
        return Xlb

    def inverse_transform(self, Xlb):
        # Convert Xlb to array
        Xlb = np.array(Xlb)
        if len(self.classes_) == 2:
            X = self.lb.inverse_transform(Xlb[:, 0])
        else:
            X = self.lb.inverse_transform(Xlb)
        return X