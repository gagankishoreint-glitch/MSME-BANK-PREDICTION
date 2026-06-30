import joblib
import xgboost as xgb
import shap
from test_all_models import make_msme_data
from sklearn.impute import SimpleImputer
import os

def retrain():
    print("Generating MSME data...")
    msme_df = make_msme_data()
    features = [f for f in msme_df.columns if f != 'default_flag']
    X = msme_df[features].values
    y = msme_df['default_flag'].values

    print("Fitting imputer...")
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)

    print("Training XGBoost...")
    spw = float((y == 0).sum() / (y == 1).sum())
    xgb_model = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, 
                                  subsample=0.8, colsample_bytree=0.85, 
                                  reg_lambda=2, reg_alpha=0.5, scale_pos_weight=spw,
                                  eval_metric='auc', random_state=42, n_jobs=-1)
    xgb_model.fit(X_imp, y)

    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(xgb_model)

    print("Saving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_model, 'models/xgb_model.joblib')
    joblib.dump(explainer, 'models/shap_explainer.joblib')
    joblib.dump(imputer, 'models/imputer.joblib')
    joblib.dump(features, 'models/feature_list.joblib')
    print("Successfully retrained and saved MSME models for XGBoost version:", xgb.__version__)

if __name__ == '__main__':
    retrain()
