import torch
import torch.nn as nn
import timm
from pathlib import Path

class GenConViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=False, num_classes=0)
        self.embedder = timm.create_model('swin_small_patch4_window7_224', pretrained=False, num_classes=0, 
                                         embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24])
        self.encoder = nn.Sequential(
            nn.Linear(1536, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2000)
        )
        self.fc = nn.Linear(2000, 500)
        self.fc2 = nn.Linear(500, 2)
    
    def forward(self, x):
        b_feat = self.backbone(x)
        e_feat = self.embedder(x)
        combined = torch.cat([b_feat, e_feat], dim=1)
        enc = self.encoder(combined)
        dec = self.decoder(enc)
        out = self.fc(dec)
        logits = self.fc2(out)
        return torch.softmax(logits, dim=1)[:, 1:2]

class GenConViTDetector:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if GenConViTDetector._initialized:
            return
        
        print("Loading GenConViT models...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        weights_dir = Path(__file__).parent / 'weights'
        ed_path = weights_dir / 'genconvit_ed_inference.pth'
        vae_path = weights_dir / 'genconvit_vae_inference.pth'
        
        if not ed_path.exists() or not vae_path.exists():
            raise FileNotFoundError(f"Model weights not found in {weights_dir}/")
        
        self.ed_model = GenConViTModel().to(self.device)
        self.vae_model = GenConViTModel().to(self.device)
        
        ed_state = torch.load(ed_path, map_location=self.device)
        vae_state = torch.load(vae_path, map_location=self.device)
        
        model_state = self.ed_model.state_dict()
        filtered_ed = {k: v for k, v in ed_state.items() if k in model_state and v.shape == model_state[k].shape}
        filtered_vae = {k: v for k, v in vae_state.items() if k in model_state and v.shape == model_state[k].shape}
        
        self.ed_model.load_state_dict(filtered_ed, strict=False)
        self.vae_model.load_state_dict(filtered_vae, strict=False)
        
        self.ed_model.eval()
        self.vae_model.eval()
        
        GenConViTDetector._initialized = True
        print(f"✅ Models loaded on {self.device}")
    
    def predict(self, frames_tensor):
        with torch.inference_mode():
            ed_pred = self.ed_model(frames_tensor)
            vae_pred = self.vae_model(frames_tensor)
            combined = (ed_pred + vae_pred) / 2
            return combined.mean().item()
