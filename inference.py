import torch
from src.model import SimpleECGCNN

model = SimpleECGCNN()
model.load_state_dict(torch.load("outputs/cnn_model.pth", map_location='cpu'))
model.eval()

# 예시 입력 (1개 샘플, 12채널, 5000 길이)
dummy_input = torch.randn(1, 12, 5000)
with torch.no_grad():
    output = model(dummy_input)
    pred = torch.argmax(output, dim=1)
    print("Predicted class:", pred.item())

