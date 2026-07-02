from sklearn.decomposition import PCA

def apply_pca(X_train_scaled, X_test_scaled, n_components):
    
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    return pca, X_train_pca, X_test_pca
