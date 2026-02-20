import sys
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score


def main():
    # Generate sample data
    np.random.seed(42)
    
    # Create categorical features
    categories = ['A', 'B', 'C', 'D', 'E']
    n_samples = 1000
    
    # Generate categorical data
    cat_data = np.random.choice(categories, size=(n_samples, 3))
    
    # Generate numerical features
    num_features = np.random.randn(n_samples, 5)
    
    # Combine features
    X = np.hstack([cat_data, num_features])
    
    # Generate target variable (regression)
    y_reg = (num_features[:, 0] * 2 + num_features[:, 1] * 1.5 + 
             np.random.randn(n_samples) * 0.5)
    
    # Generate target for classification
    y_class = (num_features[:, 0] + num_features[:, 1] > 0).astype(int)
    
    # Feature hashing for categorical features
    fh = FeatureHasher(n_features=16, input_type='string')
    
    # Split data
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
        X, y_reg, y_class, test_size=0.2, random_state=42
    )
    
    # Process training data with feature hashing
    X_train_cat = fh.transform(X_train[:, :3].tolist()).toarray()
    X_train_num = X_train[:, 3:].astype(float)
    X_train_processed = np.hstack([X_train_cat, X_train_num])
    
    # Process test data with feature hashing
    X_test_cat = fh.transform(X_test[:, :3].tolist()).toarray()
    X_test_num = X_test[:, 3:].astype(float)
    X_test_processed = np.hstack([X_test_cat, X_test_num])
    
    # Train regression model
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train_processed, y_reg_train)
    
    # Train classification model
    class_model = RandomForestClassifier(n_estimators=100, random_state=42)
    class_model.fit(X_train_processed, y_class_train)
    
    # Get predictions
    train_pred_reg = reg_model.predict(X_train_processed)
    val_pred_reg = reg_model.predict(X_test_processed)
    
    train_pred_class = class_model.predict(X_train_processed)
    val_pred_class = class_model.predict(X_test_processed)
    
    # Calculate metrics
    best_train_metrics = {
        'r2': r2_score(y_reg_train, train_pred_reg),
        'mse': mean_squared_error(y_reg_train, train_pred_reg),
        'accuracy': accuracy_score(y_class_train, train_pred_class),
        'f1_macro': f1_score(y_class_train, train_pred_class, average='macro')
    }
    
    best_val_metrics = {
        'r2': r2_score(y_reg_test, val_pred_reg),
        'mse': mean_squared_error(y_reg_test, val_pred_reg),
        'accuracy': accuracy_score(y_class_test, val_pred_class),
        'f1_macro': f1_score(y_class_test, val_pred_class, average='macro')
    }
    
    # Quality checks
    print("\n" + "=" * 60)
    print("Quality Checks")
    print("=" * 60)
    
    # Check 1: Best model should have good performance
    quality_pass = True
    
    # Check train R2 > 0.8
    train_r2_pass = best_train_metrics['r2'] > 0.8
    print(f"{'✓' if train_r2_pass else '✗'} Train R2 > 0.8: {best_train_metrics['r2']:.4f}")
    quality_pass = quality_pass and train_r2_pass
    
    # Check val R2 > 0.7
    val_r2_pass = best_val_metrics['r2'] > 0.7
    print(f"{'✓' if val_r2_pass else '✗'} Val R2 > 0.7: {best_val_metrics['r2']:.4f}")
    quality_pass = quality_pass and val_r2_pass
    
    # Check val MSE < 1.0
    val_mse_pass = best_val_metrics['mse'] < 1.0
    print(f"{'✓' if val_mse_pass else '✗'} Val MSE < 1.0: {best_val_metrics['mse']:.4f}")
    quality_pass = quality_pass and val_mse_pass
    
    # Check accuracy > 0.7
    accuracy_pass = best_val_metrics['accuracy'] > 0.7
    print(f"{'✓' if accuracy_pass else '✗'} Val Accuracy > 0.7: {best_val_metrics['accuracy']:.4f}")
    quality_pass = quality_pass and accuracy_pass
    
    # Check F1 macro > 0.7
    f1_pass = best_val_metrics['f1_macro'] > 0.7
    print(f"{'✓' if f1_pass else '✗'} Val F1 Macro > 0.7: {best_val_metrics['f1_macro']:.4f}")
    quality_pass = quality_pass and f1_pass
    
    # Check R2 difference < 0.15 (avoid overfitting)
    r2_diff = abs(best_train_metrics['r2'] - best_val_metrics['r2'])
    overfit_pass = r2_diff < 0.15
    print(f"{'✓' if overfit_pass else '✗'} R2 difference < 0.15: {r2_diff:.4f}")
    quality_pass = quality_pass and overfit_pass
    
    # Final summary
    print("\n" + "=" * 60)
    if quality_pass:
        print("PASS: All quality checks passed!")
        print("=" * 60)
        return 0
    else:
        print("FAIL: Some quality checks failed!")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
