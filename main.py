import matplotlib.pyplot as plt
import kagglehub
import pandas as pd
import os
import seaborn as sns
import numpy as np
import numbers

# Select which plots are needed
printHistograms = False
printCorrMatrix = False
printBoxPlots = True

# Downloading latest version of the analyzed dataset
path = kagglehub.dataset_download("jtrofe/beer-recipes")

print("Path to dataset files: ", path)

recipeData = os.path.join(path, "recipeData.csv")
recipes = pd.read_csv(recipeData, encoding='ISO-8859-1')

print("Amount of recipes: ", len(recipes))

# Attributes that don't need to be analyzed mainly due to being unique for each row or there being too many
unnecessaryAttributes = ["BeerID", "Name", "URL", "UserId", "StyleID", "Style"]
# Attributes that aren't standardised
nonStandardised = ["PrimingMethod", "PrimingAmount"]

# Dropping the unnecessary attributes
for attribute in unnecessaryAttributes:
    recipes = recipes.drop(attribute, axis=1, errors='ignore')
for attribute in nonStandardised:
    recipes = recipes.drop(attribute, axis=1, errors='ignore')

# Only leaving columns with number values
numeric_recipes = recipes.select_dtypes(include=['number'])

# Histograms
if printHistograms:
    for attribute in recipes.keys():
            min = recipes[attribute].min()
            max = recipes[attribute].max()

            if isinstance(min, numbers.Number):
                bins = np.arange(min, max, (max-min)/20)
                print("-----------\n", attribute, ":\nmin: ", min, "\nmax: ", max, "\n-----------")
                plt.hist(recipes[attribute].dropna(), rwidth=0.9, bins=bins, edgecolor='black')
            else:
                print("-----------\n", attribute, ":\nNon-numerical\n-----------")
                plt.hist(recipes[attribute].dropna(), rwidth=0.9, edgecolor='black')

            plt.title("Histogram atrybutu " + attribute)
            plt.xlabel(attribute)
            plt.ylabel("Liczba próbek")
            plt.show()


# Correlation matrix
if printCorrMatrix:
    corrMatrix = numeric_recipes.corr(method="spearman")
    plt.figure()
    sns.heatmap(corrMatrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Macierz korelacji metodą Spearmana")
    plt.show()

# Box plots
if printBoxPlots:
    for attribute in numeric_recipes.keys():
        # Filtering below the 5th and above the 95th percentile for most attributes so that the plot is readable
        if attribute != "PitchRate" and attribute != "MashThickness":
            qLow = numeric_recipes[attribute].quantile(0.05)
            qHigh = numeric_recipes[attribute].quantile(0.95)
            filtered = numeric_recipes[(numeric_recipes[attribute] >= qLow) & (numeric_recipes[attribute] <= qHigh)]
        else:
            filtered = numeric_recipes

        plt.boxplot(filtered[attribute].dropna())
        plt.title("Wykres pudełkowy atrybutu " + attribute)
        plt.ylabel(attribute)
        plt.show()
