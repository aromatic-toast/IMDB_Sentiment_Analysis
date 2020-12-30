# Author: Lesley Miller
# Date: 2020/12/29
"""
The purpose of this script is pull a random sample of movie reviews from the
raw training dataset for display on the dashboard. The full train set is too large
to push to GitHub so a subset will be created here that can be pushed to the dev branch
 of the dashboard.
"""
import pandas as pd

# load raw train set
train = pd.read_parquet("data/Train.parquet")

# take a sample of 30 reviews
train_subset = train.sample(n=30, random_state=42)

# write out the data subset
train_subset.to_parquet("data/Train_subset.parquet", index=False)