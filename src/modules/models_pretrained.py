import torch
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel

class ResnetEncoder(torch.nn.Module):
    def __init__(self, resnet_version="resnet18", pretrained=True):
        super().__init__()
        # Pobieramy klasę modelu na podstawie nazwy
        resnet_constructor = getattr(models, resnet_version)
        self.model = resnet_constructor(pretrained=pretrained)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        for param in self.model.parameters():   #NA PEWNO W TYM MIEJSCU?
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: tensor [B, 3, H, W] na odpowiednim urządzeniu
        returns: embeddings [B, feature_dim] (feature_dim zależy od wersji ResNet)
        """
        return self.model(images)

class TextEncoder(torch.nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", max_length=128, device="cpu"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.max_length = max_length
        for param in self.model.parameters():   #NA PEWNO W TYM MIEJSCU?
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, descs):
        """
        descs: list of str
        returns: last_hidden_state [B, L, D], pooler_output (if exists) or None
        """
        if isinstance(descs, str):
            descs = [descs]
        enc = self.tokenizer(
            list(descs),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.model(**enc)
        return out

