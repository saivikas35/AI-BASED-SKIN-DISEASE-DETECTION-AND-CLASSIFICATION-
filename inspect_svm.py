import pickle
path = 'skin_disease_detection/models/svm_model_optimized.pkl'
with open(path, 'rb') as f:
    m = pickle.load(f)
print('type:', type(m))
print('has predict:', hasattr(m, 'predict'))
if isinstance(m, dict):
    print('keys:', m.keys())
else:
    print('object repr', repr(m))
