import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from diffusers.models import AutoencoderKL
import yaml
import argparse
import logging
import os

from models import CDiT_models
from diffusion import create_diffusion
from datasets import TrainingDataset
from misc import transform
from distributed import init_distributed
from train import evaluate

def create_logger(logging_dir):
    """
    创建日志记录器
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/eval_log.txt")]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--save-dir", type=str, required=True, help="评估结果保存路径")
    parser.add_argument("--bfloat16", type=int, default=1, help="是否使用bfloat16")
    parser.add_argument("--global-seed", type=int, default=0, help="全局随机种子")
    args = parser.parse_args()

    # 初始化分布式训练
    _, rank, device, _ = init_distributed()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    
    # 加载配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # 创建保存目录和日志
    os.makedirs(args.save_dir, exist_ok=True)
    logger = create_logger(args.save_dir)
    
    # 初始化模型
    tokenizer = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    latent_size = config['image_size'] // 8
    num_cond = config['context_size']
    
    model = CDiT_models[config['model']](
        context_size=num_cond, 
        input_size=latent_size,
        in_channels=4,
        use_instruction=config.get("use_instruction", False)
    ).to(device)
    
    # 加载checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    map_location_str = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=map_location_str, weights_only=False)  # 显式设置 weights_only=False
    if "ema" in checkpoint:
        model.load_state_dict({k.replace('_orig_mod.', ''):v for k,v in checkpoint['ema'].items()})
        logger.info("Loaded EMA weights")
    else:
        model.load_state_dict({k.replace('_orig_mod.', ''):v for k,v in checkpoint['model'].items()})
        logger.info("Loaded model weights")
    
    # 创建测试数据集
    test_dataset = []
    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "test" in data_config:
            dataset = TrainingDataset(
                data_folder=data_config["data_folder"],
                data_split_folder=data_config["test"],
                dataset_name=dataset_name,
                image_size=config["image_size"],
                min_dist_cat=data_config.get("distance", {}).get("min_dist_cat", config["distance"]["min_dist_cat"]),
                max_dist_cat=data_config.get("distance", {}).get("max_dist_cat", config["distance"]["max_dist_cat"]),
                len_traj_pred=data_config.get("len_traj_pred", config["len_traj_pred"]),
                context_size=config["context_size"],
                normalize=config["normalize"],
                goals_per_obs=4,  # 标准化测试设置
                transform=transform,
                predefined_index=None,
                instruction_file=data_config.get("instruction_file", None),
                traj_stride=1,
            )
            test_dataset.append(dataset)
            logger.info(f"Dataset: {dataset_name} (test), size: {len(dataset)}")
    
    test_dataset = ConcatDataset(test_dataset)
    diffusion = create_diffusion(timestep_respacing="50")
    
    # 运行评估
    model.eval()
    with torch.no_grad():
        sim_score = evaluate(
            model=model,
            vae=tokenizer,
            diffusion=diffusion,
            test_dataloaders=test_dataset,
            rank=rank,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            latent_size=latent_size,
            device=device,
            save_dir=args.save_dir,
            seed=args.global_seed,
            bfloat_enable=bool(args.bfloat16),
            num_cond=num_cond,
            train_steps=0,
            eval_fraction=1.0  # 评估整个测试集
        )
    
    logger.info(f"Evaluation completed. Final score: {sim_score:.4f}")
    
    # 清理分布式进程
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()