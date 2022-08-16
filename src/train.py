SCRIPT_DIR = os.path.dirname('../src/')
sys.path.append(os.path.dirname(SCRIPT_DIR))

from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold



def cross_validate_smoth(k, model, X, y, over_sampling=False):
    kfold = KFold(n_splits=k) 
    accuracy_split = []
    
    # interando sobre os splits
    for idx, (idx_train, idx_validate) in enmerate(kfold.split(X)):
        X_split_train = X.iloc[idx_train, :]
        y_split_train = y.iloc[idx_train, :]
        
        if over_sampling:
            sm = SMOTE(random_state=42)
            X_split_train, y_split_train = sm.fit_resample(X_split_train, y_split_train)
            
        # com os dados balanceados SÓ NO TREINO, vamos treinar 
        model.fit(X_split_train, y_split_train.values.flatten())
        
        X_split_validate = X.iloc[idx_validate, :]
        y_split_validate = y.iloc[idx_validate, :]
        
        # validacao SEM oversampling, amostra do mundo real com dados desbalanceados
        predictions_val = model.predict(X_split_validate)
        
        accuracy_split = accuracy_score(y_split_validate, predictions_val)
        
        print(f'Acuracia do split {idx}: {accuracy_split}')
    
    return accuracy_split
        