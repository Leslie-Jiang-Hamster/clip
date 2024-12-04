from PIL import Image

from transformers import CLIPProcessor, CLIPModel

class Classifier:
  def __init__(self, categories):
    self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    self.categories = categories
  
  def predict(self, image):
    inputs = self.processor(text=self.categories, images=image, return_tensors="pt", padding=True)
    outputs = self.model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    category = self.categories[probs.argmax()]
    return category

classifier = Classifier(["cat", "dog", "elephant"])

image = Image.open("cat.jpg")

print(classifier.predict(image))

