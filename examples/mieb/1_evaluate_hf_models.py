import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor, SiglipModel, SiglipProcessor, BlipModel, BlipProcessor
from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm

import gc

# ───────────── user‑tweakables ─────────────
N_IMAGES   = 8_192
BATCH_SIZE = 512
NUM_WORKERS = 0
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_DICT = {
  'EVA02-CLIP-bigE-14-plus': 'QuanSun/EVA-CLIP',
  'google/siglip-so400m-patch14-384': 'google/siglip-so400m-patch14-384',
  'google/siglip-large-patch16-384': 'google/siglip-large-patch16-384',
  'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k': 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k',
  'laion/CLIP-ViT-g-14-laion2B-s34B-b88K': 'laion/CLIP-ViT-g-14-laion2B-s34B-b88K',
  'EVA02-CLIP-bigE-14': 'QuanSun/EVA-CLIP',
  'google/siglip-large-patch16-256': 'google/siglip-large-patch16-256',
  'laion/CLIP-ViT-H-14-laion2B-s32B-b79K': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
  'laion/CLIP-ViT-L-14-laion2B-s32B-b82K': 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K',
  'royokong/e5-v': 'royokong/e5-v',
  'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K': 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',
  'google/siglip-base-patch16-512': 'google/siglip-base-patch16-512',
  'google/siglip-base-patch16-256': 'google/siglip-base-patch16-256',
  'google/siglip-base-patch16-384': 'google/siglip-base-patch16-384',
  'google/siglip-base-patch16-256-multilingual': 'google/siglip-base-patch16-256-multilingual',
  'google/siglip-base-patch16-224': 'google/siglip-base-patch16-224',
  'openai/clip-vit-large-patch14': 'openai/clip-vit-large-patch14',
  'blip2-finetune-coco': None,
  'laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K': 'laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K',
  'laion/CLIP-ViT-B-32-laion2B-s34B-b79K': 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K',
  'blip2-pretrain': None,
  'laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K': 'laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K',
  'voyage-multimodal-3': None,
  'openai/clip-vit-base-patch16': 'openai/clip-vit-base-patch16',
  'facebook/dinov2-giant': 'facebook/dinov2-giant',
  'facebook/dinov2-base': 'facebook/dinov2-base',
  'facebook/dinov2-large': 'facebook/dinov2-large',
  'openai/clip-vit-base-patch32': 'openai/clip-vit-base-patch32',
  'facebook/dinov2-small': 'facebook/dinov2-small',
  'Salesforce/blip-itm-large-coco': 'Salesforce/blip-itm-large-coco',
  'TIGER-Lab/VLM2Vec-Full': 'TIGER-Lab/VLM2Vec-Full',
  'TIGER-Lab/VLM2Vec-LoRA': 'TIGER-Lab/VLM2Vec-LoRA',
  'BAAI/bge-visualized-base': 'BAAI/bge-visualized',
  'Salesforce/blip-itm-base-coco': 'Salesforce/blip-itm-base-coco',
  'kakaobrain/align-base': 'kakaobrain/align-base',
  'Salesforce/blip-itm-large-flickr': 'Salesforce/blip-itm-large-flickr',
  'jinaai/jina-clip-v1': 'jinaai/jina-clip-v1',
  'nyu-visionx/moco-v3-vit-l': 'nyu-visionx/moco-v3-vit-l',
  'BAAI/bge-visualized-m3': 'BAAI/bge-visualized',
  'EVA02-CLIP-L-14': 'QuanSun/EVA-CLIP',
  'nyu-visionx/moco-v3-vit-b': 'nyu-visionx/moco-v3-vit-b',
  'Salesforce/blip-itm-base-flickr': 'Salesforce/blip-itm-base-flickr',
  'Salesforce/blip-image-captioning-large': 'Salesforce/blip-image-captioning-large',
  'EVA02-CLIP-B-16': 'QuanSun/EVA-CLIP',
  'nomic-ai/nomic-embed-vision-v1.5': 'nomic-ai/nomic-embed-vision-v1.5',
  'Salesforce/blip-image-captioning-base': 'Salesforce/blip-image-captioning-base'
}

MODEL_NAMES = list(MODEL_DICT.values())
MODEL_NAMES = [m for m in MODEL_NAMES if m is not None]


OUT_DIR = Path("kernels_out")
OUT_DIR.mkdir(exist_ok=True)
# ───────────────────────────────────────────

# 1) Dataset -------------------------------------------------------------------
print("📦  loading timm/mini‑imagenet …")
hf_ds = load_dataset("timm/mini-imagenet", split="train")

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),   # ensure 3‑ch
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class HFWrapper(Dataset):
    def __init__(self, ds, tfm):
        self.ds, self.tfm = ds, tfm
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[int(i)]
        return self.tfm(item["image"]), item["label"]

# sample once, reuse for every model
full_ds   = HFWrapper(hf_ds, transform)
indices   = random.sample(range(len(full_ds)), N_IMAGES)
subset_ds = Subset(full_ds, indices)
loader    = DataLoader(subset_ds, batch_size=BATCH_SIZE,
                       shuffle=False, num_workers=NUM_WORKERS,
                       pin_memory=True)
print(f"✓ dataset ready — {len(subset_ds)} images\n")

# 2) Feature → kernel → save  (one loop per model) -----------------------------
print("📝  computing features ...")

@torch.no_grad()
def features(model_name: str) -> torch.Tensor:
    # Load model and processor
    try:
        # Try to load as CLIP model first
        if 'clip' in model_name.lower() or 'eva' in model_name.lower():
            model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()
            processor = CLIPProcessor.from_pretrained(model_name)
        # Try to load as SigLIP model
        elif 'siglip' in model_name.lower():
            model = SiglipModel.from_pretrained(model_name).to(DEVICE).eval()
            processor = SiglipProcessor.from_pretrained(model_name)
        # Try to load as BLIP model
        elif 'blip' in model_name.lower():
            model = BlipModel.from_pretrained(model_name).to(DEVICE).eval()
            processor = BlipProcessor.from_pretrained(model_name)
        # Try to load as generic AutoModel
        else:
            model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
            processor = AutoProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None
    
    vecs = []
    for imgs, _ in tqdm(loader, desc=f"{model_name:>24}", leave=False):
        try:
            # Convert PIL images to the format expected by the processor
            pil_images = [transforms.ToPILImage()(img) for img in imgs]
            
            # Process images with the appropriate processor
            if hasattr(processor, 'image_processor'):
                # For newer processors
                processed = processor.image_processor(pil_images, return_tensors="pt")
            else:
                # For older processors
                processed = processor(pil_images, return_tensors="pt")
            
            # Move to device
            processed = {k: v.to(DEVICE) for k, v in processed.items()}
            
            # Get features from the model
            if hasattr(model, 'get_image_features'):
                # For CLIP/SigLIP models
                batch_output = model.get_image_features(**processed)
            elif hasattr(model, 'get_vision_features'):
                # For BLIP models
                batch_output = model.get_vision_features(**processed)
            elif hasattr(model, 'vision_model'):
                # For models with vision_model attribute
                vision_outputs = model.vision_model(**processed)
                batch_output = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs.last_hidden_state.mean(dim=1)
            elif hasattr(model, 'embeddings'):
                # For DINOv2 and similar models
                outputs = model(**processed)
                batch_output = outputs.last_hidden_state.mean(dim=1) if hasattr(outputs, 'last_hidden_state') else outputs.pooler_output
            else:
                # Generic approach
                outputs = model(**processed)
                if hasattr(outputs, 'last_hidden_state'):
                    batch_output = outputs.last_hidden_state.mean(dim=1)
                elif hasattr(outputs, 'pooler_output'):
                    batch_output = outputs.pooler_output
                elif hasattr(outputs, 'image_embeds'):
                    # For some models like BLIP
                    batch_output = outputs.image_embeds
                else:
                    batch_output = outputs[0].mean(dim=1)
            
            # Ensure we have a valid tensor
            if batch_output is None:
                print(f"Warning: No output from model {model_name} for batch")
                continue
                
            # Flatten and convert to float32
            batch_output = batch_output.flatten(1).to(torch.float32)
            vecs.append(batch_output)
            
        except Exception as e:
            print(f"Error processing batch for {model_name}: {e}")
            continue
        
        # Explicitly free memory
        torch.cuda.empty_cache()
    
    print('done with looping')
    
    if not vecs:
        return None
    
    return torch.cat(vecs)


for m in MODEL_NAMES:
    print(f"🚀  processing {m} …")
    try:
        F_m = features(m)                            # (N, D_m)
        if F_m is None:
            print(f"   ↳ skipped {m} (failed to load/process)\n")
            continue
            
        print('features computed')
        Z_m = F.normalize(F_m, p=2, dim=1)           # row‑norm
        K_m = Z_m @ Z_m.T                            # (N, N)

        m_safe = m.replace('/', '__')
        torch.save(
            {"K": K_m.cpu(),                         # kernel
             "Z": F_m.cpu(),                         # normalised feats
             "dim": F_m.shape[1],                    # feature length of this model
             "indices": indices},
            OUT_DIR / f"K_{m_safe}_{N_IMAGES}.pt"
        )
        print(f"   ↳ saved  {OUT_DIR / f'K_{m_safe}_{N_IMAGES}.pt'}\n")

        # Clean up memory before next modeltmu
        del F_m, Z_m, K_m
        torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection
    except Exception as e:
        print(f"Error processing model {m}: {e}")
        torch.cuda.empty_cache()
        gc.collect()  # Still try to clean up memory on error

print("✅  all kernels done.")