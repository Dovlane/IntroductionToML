# Introduction to ML - predicting house prices

I programmed in Python using the Jupyter platform, applying libraries such as NumPy, Pandas, and Scikit-learn. I utilized correlation matrices and various statistical methods such as Z-score and R-Squared. I presented to my colleagues at the Mathematical Institute the basics of machine learning, referencing a specific problem from the link below. For training the model, I employed multilinear regression.

Link to problem:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

# Intro to problem

Main goal is to based on this data to predict house prices.

Dataset has 79 descriptive columns:
1. 57 categorical
2. 20 quantitative
3. 4 temporal

Examples of categorical columns:
1. OverallQual: Rates the overall material and finish of the house
2. LotShape: General shape of property
3. HeatingQC: Heating quality and condition


Examples of quantitative columns:
1. LotArea: Lot size in square feet
2. GrLivArea: Above grade (ground) living area square feet
3. FullBath: Full bathrooms above grade


Examples of temporal columns:
1. YearBuilt: Original construction date
2. YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
3. GarageYrBlt: Year garage was built
4. YrSold: Year Sold

# Conclusion
After reprocessing this data some of columns were dropped, and categorical columns are one-hot encoded. After applying multilinear regression, vital columns has excpected to be:
  1. GrLivArea
  2. Condition 1
  3. BldgType
  4. YearBuilt
  5. Fireplaces

# Results:
- Score on Kaggle: 0.16938
- MAE = 18798.120 (Mean absolute error)
- MAPE = 0.10874900729951524 (Mean absolute percentage error)
