import torch 
import torch.nn as nn 
import torchvision.models as models
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torchaudio
import cv2
from PIL import Image
import sounddevice as sd
import time
import gc  # Added for memory management
from torchvision import transforms, datasets
device = torch.device("cpu")
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class FaceEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 6)   
    def forward(self, images):
        return self.resnet(images)
class VoiceEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Linear(self.wav2vec.config.hidden_size, 6)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        return self.classifier(pooled)
class CombinedEmotionModel(nn.Module):
    def __init__(self, voice_model, face_model):
        super().__init__()
        self.face_model = face_model
        self.voice_model = voice_model
        self.fusion = nn.Linear(12, 6)
    def forward(self, *, input_values=None, attention_mask=None, images=None):
        voice_output = self.voice_model(input_values=input_values, attention_mask=attention_mask)
        face_output = self.face_model(images)
        combined = torch.cat((voice_output, face_output), dim=1)
        return self.fusion(combined)
def load_model_safe(model_class, path):
    model = model_class().eval().to(device)
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        del state_dict  # Free memory
        gc.collect()
        return model
    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        raise
gc.collect()  # Clean up before loading
print("Loading face model...")
face_model = load_model_safe(FaceEmotionModel,  "models/face_emotion_model.pth")
print("Loading voice model...")
voice_model = load_model_safe(VoiceEmotionModel, "models/voice_emotion_model.pth")
print("Creating combined model...")
combined_model = CombinedEmotionModel(voice_model, face_model).eval().to(device)
try:
    combined_model.load_state_dict(torch.load("models/combined_emotion_model.pth", map_location=device))
except:
    print("Note: Couldn't load combined model weights, using component models only")
def capture_and_predict():
    cap = cv2.VideoCapture(0)  
    time.sleep(2)  # Reduced from 5 to 2 seconds
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "Failed to capture frame"
    print("Recording audio...")
    audio = sd.rec(int(4 * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    inputs = processor(
        audio.squeeze(), sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=16000*4, truncation=True).to(device)
    with torch.no_grad():
        outputs = combined_model(
            input_values=inputs.input_values,  images=image_tensor)
        pred = torch.argmax(outputs).item()
    del inputs, image_tensor
    gc.collect()
    
    idx_to_label = {
        0: "HAP", 1: "SAD", 2: "ANG",
        3: "FEA", 4: "DIS", 5: "NEU"
    }
    return idx_to_label[pred]
print("Predicted emotion:", capture_and_predict())