from models.baseline_cnn import build_baseline_cnn
from models.regularized_cnn import build_regularized_cnn
from models.deep_cnn import build_deep_cnn


model1 = build_baseline_cnn()
model2 = build_regularized_cnn()
model3 = build_deep_cnn()


print("\nCNN 1")
model1.summary()

print("\nCNN 2")
model2.summary()

print("\nCNN 3")
model3.summary()