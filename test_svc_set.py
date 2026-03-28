from sklearn.svm import SVC
import pickle
path='skin_disease_detection/models/svm_model_optimized.pkl'
with open(path,'rb') as f: loaded=pickle.load(f)
print(type(loaded))
model=SVC()
try:
    for k,v in loaded.items():
        try:
            setattr(model, k, v)
        except Exception:
            model.__dict__[k] = v
    print('reconstructed via model.__dict__.update allowed')
except Exception as e:
    print('reconstruction failed', e)

print('has predict', hasattr(model,'predict'))
print('dual_coef_', getattr(model,'dual_coef_', None))
print('support_', getattr(model,'support_', None))
print('classes_', getattr(model,'classes_', None))
