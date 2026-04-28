import torch, os

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from torchvision import transforms
from Student.osunet_student import LEDSR
from transformers import CLIPTextModel, AutoTokenizer

parser = ArgumentParser()
parser.add_argument(
    "--sd_path",
    type=str,
    default="stabilityai/sd-turbo",
    help="Path to the stable diffusion model",
)
parser.add_argument(
    "--taesd_path",
    type=str,
    default="madebyollin/taesd",
    help="Path to the taesd model",
)
parser.add_argument(
    "--prune_depth", type=int, default=2, help="depth to prune the unet"
)
parser.add_argument("--fp16", type=bool, default=False)
parser.add_argument("--upscale", type=int, default=4)
parser.add_argument("--model_dir", type=str, default="exp/V3.11.1/weight")
parser.add_argument(
    "--LR_dir",
    type=str,
    default="/mnt/massive/wangce/LightSD/dataset/val-dior/images",
)
parser.add_argument("--SR_dir", type=str, default="result/Ours-final/DIOR")
args = parser.parse_args()

device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    "stabilityai/sd-turbo", subfolder="text_encoder"
).cuda()
text_encoder.requires_grad_(False)
with torch.no_grad():
    text_token = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.cuda()
    text_embedding = text_encoder(text_token)[0]
# === FP16 推理 ===
model = torch.nn.DataParallel(LEDSR(args, text_embedding)).to(device)
model.load_state_dict(args.model_dir, weights_only=False)

if args.fp16:
    model = model.half()
model.module.set_eval()

test_LR_paths = sorted(os.listdir(args.LR_dir))
test_LR_paths = [os.path.join(args.LR_dir, name) for name in test_LR_paths]

os.makedirs(args.SR_dir, exist_ok=True)

if args.fp16:
    autocast_context = torch.cuda.amp.autocast()
else:
    # 一个“空的 context manager”，这样写可以保证结构一致
    from contextlib import nullcontext

    autocast_context = nullcontext()

with torch.no_grad(), autocast_context:
    for i, path in enumerate(test_LR_paths):
        # 读取 & 预处理
        LR = Image.open(path).convert("RGB")
        LR = transforms.ToTensor()(LR).to(device).unsqueeze(0) * 2 - 1
        LR = F.interpolate(LR, scale_factor=args.upscale, mode="bicubic")

        # padding 到 256 的倍数
        _, _, h, w = LR.shape
        pad_h = (256 - h % 256) % 256
        pad_w = (256 - w % 256) % 256
        LR = F.pad(LR, (0, pad_w, 0, pad_h), mode="reflect")

        # 推理（根据精度切换输入 dtype）
        inp = LR.half() if args.fp16 else LR
        z0_student, SR = model(inp)

        # 裁剪回原尺寸
        SR = SR[:, :, :h, :w]

        # 亮度归一化
        SR = (SR - SR.mean(dim=[2, 3], keepdim=True)) / SR.std(
            dim=[2, 3], keepdim=True
        ) * LR.std(dim=[2, 3], keepdim=True) + LR.mean(dim=[2, 3], keepdim=True)

        # 保存结果
        SR = transforms.ToPILImage()((SR[0] / 2 + 0.5).clamp(0, 1).float().cpu())
        SR.save(os.path.join(args.SR_dir, os.path.basename(path)))
