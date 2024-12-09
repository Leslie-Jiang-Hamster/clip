from transformers import ChineseCLIPProcessor, ChineseCLIPModel

class Classifier:
  def __init__(self, categories):
    self.model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    self.processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
    self.categories = categories
  
  def predict(self, image):
    inputs = self.processor(text=self.categories, images=image, return_tensors="pt", padding=True)
    outputs = self.model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    # print(probs)
    category = self.categories[probs.argmax()]
    return category