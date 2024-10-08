def distribution_plot(df, x, kind="hist", kde=False):
    """
    Function to generate distribution plots: histogram, KDE, boxplot, violinplot.
    
    Parameters
    ----------
    df: pd.DataFrame
        The data to plot.
    x : str
        The column name to plot.
    kind : str
        Type of plot to create. Options: 'hist', 'kdeplot', 'boxplot', 'violinplot'.
    kde : bool
        Whether to add KDE curve to the histogram (only applicable for 'hist').
    hue : str
        Grouping variable that will produce different colors in the plot.
    
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    
    if kind == "hist":
        sns.histplot(data=df, x=x, kde=kde, ax=ax)
        ax.set_title(f'Histogram of {x}')
        
    elif kind == "kdeplot":
        sns.kdeplot(data=df, x=x, ax=ax)
        ax.set_title(f'KDE Plot of {x}')
        
    elif kind == "boxplot":
        sns.boxplot(data=df, x=x, ax=ax)
        ax.set_title(f'Boxplot of {x}')
        
    elif kind == "violinplot":
        sns.violinplot(data=df, x=x, ax=ax)
        ax.set_title(f'Violin Plot of {x}')
    
    elif kind == "piechart":
        distribution_count = df[x].value_counts()
        ax.pie(distribution_count, labels=distribution_count.index, autopct='%1.1f%%', startangle=140, 
                colors=sns.color_palette('Dark2', len(distribution_count)))
        ax.set_title(f'Pie Chart of {x}')
    
    elif kind == "countplot":
        sns.countplot(data=df, x=x, ax=ax, palette='Dark2')
        ax.set_title(f'Countplot of {x}')
    
    else:
        raise ValueError(f"Invalid 'kind' parameter. Supported types are 'hist', 'kdeplot', 'boxplot', 'violinplot'.")
    

def categorical_plot(df, x, y=None, kind="barplot", hue=None):
    """
    Function to generate categorical plots: barplot, countplot, pointplot, stripplot, swarmplot.
    
    Parameters
    ----------
    df : DataFrame
        The data to plot.
    x : str
        The column name for categories.
    y : str, optional
        The column name for values (if applicable).
    kind : str
        Type of plot to create. Options: 'barplot', 'countplot', 'pointplot', 'stripplot', 'swarmplot'.
    hue : str, optional
        Grouping variable that will produce different colors in the plot.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    
    if kind == "barplot":
        sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax)
        ax.set_title(f'Barplot of {x} vs {y}')
        plt.xticks(rotation=90)
        
    elif kind == "pointplot":
        sns.pointplot(data=df, x=x, y=y, hue=hue, ax=ax)
        ax.set_title(f'Point Plot of {x} vs {y}')
        
    elif kind == "stripplot":
        sns.stripplot(data=df, x=x, y=y, hue=hue, ax=ax)
        ax.set_title(f'Stripplot of {x} vs {y}')
        
    elif kind == "swarmplot":
        sns.swarmplot(data=df, x=x, y=y, hue=hue, ax=ax)
        ax.set_title(f'Swarmplot of {x} vs {y}')
    
    else:
        raise ValueError(f"Invalid 'kind' parameter. Supported types are 'barplot', 'countplot', 'pointplot', 'stripplot', 'swarmplot'.")
    
def relationship_plot(df, x, y, kind="scatterplot", hue=None):
    """
    Function to generate relationship plots: scatterplot, lineplot, pairplot, jointplot.
    
    Parameters
    ----------
    df : DataFrame
        The data to plot.
    x : str
        The column name for the x-axis.
    y : str
        The column name for the y-axis.
    kind : str
        Type of plot to create. Options: 'scatterplot', 'lineplot', 'pairplot', 'jointplot'.
    hue : str, optional
        Grouping variable that will produce different colors in the plot.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    
    if kind == "scatterplot":
        sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
        ax.set_title(f'Scatterplot of {x} vs {y}')
        
    elif kind == "lineplot":
        sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax)
        ax.set_title(f'Line Plot of {x} vs {y}')
    
    elif kind == "jointplot":
        sns.jointplot(data=df, x=x, y=y, hue=hue)
        plt.suptitle(f'Jointplot of {x} vs {y}', y=1.02)
        return
    
    else:
        raise ValueError(f"Invalid 'kind' parameter. Supported types are 'scatterplot', 'lineplot', 'jointplot'.")


def roc_curve_plot(y, y_preds):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    roc_auc = roc_auc_score(y, y_preds)

    fpr, tpr, thresholds = roc_curve(y, y_preds)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

def confusion_matrix_plot(y_true, y_pred, cmap="Blues"):
    """
    Function to plot confusion matrix using Seaborn and Matplotlib.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    cmap : str, optional
        Colormap for the heatmap. Default is 'Blues'.
    
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
