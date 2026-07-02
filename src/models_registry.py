from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_model(model_name):
    
    
    
    return {
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(random_state=42, n_estimators=100)
    }