from fastcoref import FCoref

model = FCoref(device='cpu')

preds = model.predict(
   texts=['We are so happy to see you using our coref package. This package is very fast!']
)
print(preds)