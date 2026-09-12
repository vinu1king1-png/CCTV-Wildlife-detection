import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn
import torch
from torchvision import datasets, models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 4 )


model.load_state_dict(torch.load(r"C:\Users\Asus\New folder (2)\resnet18_4class.pth", map_location=device))
model.to(device)
model.eval()

# ResNet standard image preprocessing pipeline
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class_names = ['cheetah', 'leopard', 'lion', 'tiger']

cap = cv2.VideoCapture("http://192.168.0.158:8080/video")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    label_text = f"{class_names[predicted_idx.item()]}: {confidence.item():.2f}"

    # 4. Display result on frame
    cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Phone Feed - ResNet Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()