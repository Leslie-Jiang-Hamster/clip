from clip import Classifier
from PIL import Image
from prompt import observe, observe_
import os

categories = [*observe, *observe_]
assert(len(categories) == 20)

classifier = Classifier(categories)
TP = FP = TN = FN = cnt = 0
image_names = os.listdir("images")
for image_name in image_names:
  image = Image.open(f"images/{image_name}")
  category = classifier.predict(image)
  if "观察口" in image_name and category in observe:
    TP += 1
  elif "观察口" not in image_name and category in observe_:
    TN += 1
  elif "观察口" in image_name and category in observe_:
    FN += 1
  elif "观察口" not in image_name and category in observe:
    FP += 1
  else:
    assert False
  cnt += 1
  print(f"{cnt}/{len(image_names)}")
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
print(f"precision: {precision}, recall: {recall}, F1: {F1}")