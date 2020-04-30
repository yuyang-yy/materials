#!/usr/bin/env python

# synthesize dataset for classification and output to a csv file
import pandas as pd
from sklearn import datasets
X, y = datasets.make_classification(n_samples=8000, n_features=8,
                                    n_informative=3, n_redundant=3, weights=[0.95],
                                    random_state=42)

data = pd.DataFrame(X, columns = ["X%d"%i for i in np.arange(1, 9)])
data['y'] = y
data.to_csv('../imbalance.csv', index=False)
