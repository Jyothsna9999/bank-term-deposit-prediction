from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from data_preprocess import data_preprocess
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
def model_selection():
    X_train, y_train, X_test, y_test = data_preprocess() 
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    print(dt.score(X_test, y_test))
    rt = RandomForestClassifier(n_estimators=100, n_jobs=1)
    rt.fit(X_train, y_train)
    print(rt.score(X_test, y_test))
    lr= make_pipeline(StandardScaler(),LogisticRegression())
    lr.fit(X_train, y_train)
    print(lr.score(X_test, y_test))

model_selection()
