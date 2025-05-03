import torch
import yaml
from PIL import Image
from torchvision import transforms
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
from models import CDiT_models
from isolated_nwm_infer import model_forward_wrapper
import os

# ==== config ====
EXP_NAME = 'nwm_cdit_xl'
MODEL_PATH = '/datadrive/sdc/yifei/nwm/logs/nwm_cdit_xl/checkpoints/latest.pth.tar'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("config/data_config.yaml", "r") as f:
    default_config = yaml.safe_load(f)
with open(f'config/nwm_cdit_ha.yaml', "r") as f:
    user_config = yaml.safe_load(f)
config = default_config
config.update(user_config)

context_size = config['context_size']
image_size = config['image_size']
latent_size = image_size // 8

# ==== model ====
model = CDiT_models[config['model']](
    input_size=latent_size,
    context_size=context_size,
    use_instruction=config.get("use_instruction", False)
).to(DEVICE).eval()

ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt["ema"], strict=True)

# 🚫 Disable torch.compile to avoid tokenizer-related FakeTensor device issues
# model = torch.compile(model)  # Commented out due to embedding crash

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(DEVICE).eval()
diffusion = create_diffusion(str(250))

# ==== image preproc ====
def load_image_tensor(path):
    _transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    img = Image.open(path).convert('RGB')
    return _transform(img)

# ==== Put only 1 context frames here ====
image_paths = [
    "samples/episode_id_00002_2_t/16.jpg"
]
obs_images = torch.stack([load_image_tensor(p) for p in image_paths]).unsqueeze(0).to(DEVICE)  # [1, 2, 3, 224, 224]

# ==== delta, rel_t, instruction ====
delta = torch.tensor([[[-0.1177, -0.0984, 0, -0.2618, 0, 0]]], dtype=torch.float32).to(DEVICE)  # [1, 1, 3]
rel_t = torch.tensor([[0.0703125]]).to(DEVICE)  # [1, 1]
instruction = ["Start by moving straight ahead, navigating between the fireplace and the surrounding furniture."]

# ==== run inference ====
with torch.no_grad():
    samples = model_forward_wrapper(
        (model, diffusion, vae),
        obs_images,
        delta,
        num_timesteps=None,
        latent_size=latent_size,
        device=DEVICE,
        num_cond=context_size,
        num_goals=1,
        rel_t=rel_t.flatten(0, 1),
        instruction=instruction,
        progress=True
    )

samples = samples * 0.5 + 0.5
samples = (samples * 255).clamp(0, 255).permute(0, 2, 3, 1).to(torch.uint8).cpu()

# ==== save result ====
os.makedirs("output_pred", exist_ok=True)
Image.fromarray(samples[0].numpy()).save("output_pred/predicted_next_step.png")
print("✅ Saved prediction to output_pred/predicted_next_step.png")