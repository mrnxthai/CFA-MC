import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def CFA(X, y, stddev_percent=10, good_native_fd=2, verify_with_baseline_model=True, baseline_model=None, visualize_with_pca=True, pca=None):
    majority_label = y.value_counts().index[0]
    minority_label = y.value_counts().index[1]
    
    # some code needed for visualization
    if visualize_with_pca and X.shape[1] > 2 and pca is None:
        raise ValueError("pca model must be given if visualize_with_pca is True")
    if visualize_with_pca:
        if X.shape[1] > 2:
            X_2d = pca.transform(X)
            X_2d = pd.DataFrame(X_2d, columns=[f"PC_{i}" for i in range(X_2d.shape[1])]).set_index(X.index)
        else:
            # data is already 2D
            X_2d = X

        X_maj_2d = X_2d[y == majority_label].copy()
        X_min_2d = X_2d[y == minority_label].copy()

    ## 1. Divide Training Data into majority and minority subsets:
    X_maj = X[y == majority_label].copy()
    X_min = X[y == minority_label].copy()

    ## 2. Compute CF-Set
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_min)
    distances, indices = nbrs.kneighbors(X_maj)
    nearest_neighbor_for_each_X_maj = X_min.iloc[indices[:, 0]].set_index(X_maj.index)

    ## 3. Compute the tolerance limit
    upper_limit = X_maj + stddev_percent/100 * X.std()
    lower_limit = X_maj - stddev_percent/100 * X.std()

    ## 4. Compute number of feature-differences
    # check whether values are close to each other
    nearest_neighbor_for_each_X_maj_within_tol = nearest_neighbor_for_each_X_maj[
        (nearest_neighbor_for_each_X_maj >= lower_limit) & (nearest_neighbor_for_each_X_maj <= upper_limit)
    ]
    num_difference_features = nearest_neighbor_for_each_X_maj_within_tol.isna().sum(axis=1)

    ## 5. Select all "good" native counterfactuals (with number of feature-differences <= good_native_fd)
    num_good_native_counterfactuals = (((num_difference_features > 0)) & (num_difference_features <= good_native_fd)).sum()
    print(f"Number of 'good' native counterfactuals in data: {num_good_native_counterfactuals}")
    if num_good_native_counterfactuals == 0:
        print("CF-Set is empty, no augmentation possible! Not applying CFA...")
        return X, y
    if (((num_difference_features > 0)) & (num_difference_features > good_native_fd)).sum() == 0:
        print("No unpaired instances found, no augmentation possible! Not applying CFA...")
        return X, y
    X_maj_unpaired = X_maj[(num_difference_features > 0) & (num_difference_features > good_native_fd)].copy()
    X_maj_in_cf_set = X_maj[(num_difference_features > 0) & (num_difference_features <= good_native_fd)].copy()

    ## 6. Find the nearest-neighbour for each unpaired instance
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_maj_in_cf_set)
    distances, indices = nbrs.kneighbors(X_maj_unpaired)
    index_of_nearest_majority_neighbor_in_cf_set = X_maj_in_cf_set.iloc[indices[:, 0]].index
    index_of_nearest_majority_neighbor_in_cf_set

    ## 7. Generate a synthetic counterfactual instance for each unpaired instance
    # for match-features, set the value of the new synthetic counterfactual to the value of the unpaired instance
    synthetic_counterfactuals = X_maj_unpaired.copy()
    match_feature_mask = nearest_neighbor_for_each_X_maj_within_tol.reindex(index_of_nearest_majority_neighbor_in_cf_set).set_index(X_maj_unpaired.index).notna()
    synthetic_counterfactuals = synthetic_counterfactuals[match_feature_mask] # set all difference-features to NaN

    # for difference-features, set the value of the new synthetic counterfactuals to the value of the nearest native counterfactual
    synthetic_counterfactuals = synthetic_counterfactuals.fillna(
        nearest_neighbor_for_each_X_maj.loc[X_maj_in_cf_set.iloc[indices[:, 0]].index].set_index(X_maj_unpaired.index)
    )

    if verify_with_baseline_model:
        if baseline_model is None:
            raise ValueError("baseline_model must be specified if verify_with_baseline_model is True")
        # verify that the new counterfactuals are actually in the minority class
        synthetic_counterfactual_predicted_class = baseline_model.predict(synthetic_counterfactuals) 
        if synthetic_counterfactual_predicted_class.all():
            print("Classifier predicted all new synthetic counterfactuals to be in the majority class! => No new minority instances => Terminating...")
            return X, y
        synthetic_counterfactuals = synthetic_counterfactuals[(synthetic_counterfactual_predicted_class == minority_label)]

    ## 8. Add synthetic counterfactuals to data 
    missing_minority_instances = len(X_maj) - len(X_min) 
    if len(synthetic_counterfactuals) > missing_minority_instances:
        # we want the data to be balanced, which means we only need to add as many minority instances as there are majority instances
        synthetic_counterfactuals = synthetic_counterfactuals[:missing_minority_instances]
    X_cfa = pd.concat([X, synthetic_counterfactuals])
    y_cfa = pd.concat([y, pd.Series([minority_label] * len(synthetic_counterfactuals), index=synthetic_counterfactuals.index)])

    if visualize_with_pca:
        ## VISUALIZE RESULTS WITH PCA
        # red   = line connecting native cf-pair
        # blue  = line connecting each unpaired instance to the next majority instance in a native cf-pair
        # green = line from unpaired instance to new synthetic counterfactual

        if X.shape[1] > 2:
            synthetic_counterfactuals_2d = pca.transform(synthetic_counterfactuals)
            synthetic_counterfactuals_2d = pd.DataFrame(synthetic_counterfactuals_2d, columns=[f"PC_{i}" for i in range(synthetic_counterfactuals_2d.shape[1])]).set_index(synthetic_counterfactuals.index)    
        else:
            # data is already 2D
            synthetic_counterfactuals_2d = synthetic_counterfactuals

        if X.shape[1] > 2:
            nearest_neighbor_for_each_X_maj_2d = pca.transform(nearest_neighbor_for_each_X_maj)
            nearest_neighbor_for_each_X_maj_2d = pd.DataFrame(nearest_neighbor_for_each_X_maj_2d, columns=[f"PC_{i}" for i in range(nearest_neighbor_for_each_X_maj_2d.shape[1])]).set_index(nearest_neighbor_for_each_X_maj.index)
        else:
            # data is already 2D
            nearest_neighbor_for_each_X_maj_2d = nearest_neighbor_for_each_X_maj

        X_maj_unpaired_2d = X_maj_2d[(num_difference_features > 0) & (num_difference_features > good_native_fd)].copy()
        X_maj_in_cf_set_2d = X_maj_2d[(num_difference_features > 0) & (num_difference_features <= good_native_fd)].copy()

        fig = plt.figure()
        plt.scatter(X_2d.values[:, 0], X_2d.values[:, 1], marker="o", c=y.values, s=25, edgecolor="k");
        plt.scatter(synthetic_counterfactuals_2d.values[:, 0], synthetic_counterfactuals_2d.values[:, 1], marker="o", c="c", s=25, edgecolor="k");
        for i in range(len(X_maj_in_cf_set_2d)):
            x1, y1 = X_maj_in_cf_set_2d.iloc[i]
            x2, y2 = nearest_neighbor_for_each_X_maj_2d.reindex(X_maj_in_cf_set_2d.index).iloc[i]
            plt.plot([x1, x2], [y1, y2], 'r-', alpha=0.2)
        for i in range(len(X_maj_unpaired_2d)):
            x1, y1 = X_maj_unpaired_2d.iloc[i]
            x2, y2 = X_maj_in_cf_set_2d.iloc[indices[:, 0]].iloc[i]
            plt.plot([x1, x2], [y1, y2], 'b-', alpha=0.2)
        for idx in synthetic_counterfactuals_2d.index:
            x1, y1 = X_maj_unpaired_2d.loc[idx]
            x2, y2 = synthetic_counterfactuals_2d.loc[idx]
            plt.plot([x1, x2], [y1, y2], 'g-', alpha=0.2)
        plt.title("Augmented data (with new synthetic counterfactuals in cyan blue)")
        plt.show();

    return X_cfa, y_cfa

def Iterative_CFA(X, y, stddev_percent=10, good_native_fd=2, verify_with_baseline_model=True, baseline_model=None, visualize_with_pca=True):
    X_cfa = X.copy()
    y_cfa = y.copy()
    
    it = 0
    
    min_count = y_cfa.value_counts()[0]
    maj_count = y_cfa.value_counts()[1]
    
    majority_label = y_cfa.value_counts().index[0]
    minority_label = y_cfa.value_counts().index[1]
    
    if visualize_with_pca:
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            pca.fit(X)
        else:
            # data is 2D, pca is not needed
            pca = None
    else:
        pca = None
    
    print(f"Data distribution before CFA:")
    print(f"\tMajority ({majority_label}): {maj_count}, Minority (label {minority_label}): {min_count}")
    
    while not np.allclose(min_count, maj_count, rtol=0.01):
        X_cfa, y_cfa = CFA(X_cfa, y_cfa, 
                           stddev_percent=stddev_percent,
                           good_native_fd=good_native_fd,
                           verify_with_baseline_model=verify_with_baseline_model, 
                           baseline_model=baseline_model, 
                           visualize_with_pca=visualize_with_pca, 
                           pca=pca)
        
        new_min_count = y_cfa.value_counts()[0]
        
        if new_min_count == min_count:
            # CFA was unable to add any synthetic counterfactuals, stop loop
            break
        else:
            min_count = new_min_count
        print(f"Data distribution after iteration {it}:")
        print(f"\tMajority ({majority_label}): {maj_count}, Minority (label {minority_label}): {min_count}")
        it += 1
        
    print("====================================================================")
    print(f"Data distribution after CFA:")
    print(f"\tMajority ({majority_label}): {maj_count}, Minority (label {minority_label}): {min_count}")

    if visualize_with_pca:
        if X_cfa.shape[1] > 2:
            X_cfa_2d = pca.transform(X_cfa)
            X_cfa_2d = pd.DataFrame(X_cfa_2d, columns=[f"PC_{i}" for i in range(X_cfa_2d.shape[1])]).set_index(X_cfa.index)
        else:
            X_cfa_2d = X_cfa

        fig = plt.figure()
        plt.scatter(X_cfa_2d.values[:, 0], X_cfa_2d.values[:, 1], marker="o", c=y_cfa.values, s=25, edgecolor="k");
        plt.title("Final augmented data")
        plt.show();

    return X_cfa, y_cfa
