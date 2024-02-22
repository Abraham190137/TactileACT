
import torchvision
from clip_pretraining import modified_resnet18
resnet18 = getattr(torchvision.models, 'resnet18')(weights=False)

print(resnet18)
print(modified_resnet18())