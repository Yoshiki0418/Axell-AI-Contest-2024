import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torchmetrics
from torchmetrics import Precision, Recall
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
import onnxruntime as ort

from src.utils import set_seed
from src.datasets import get_dataset
from src.models import get_model
from src.psnr_calculator import calc_psnr

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir= hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="axell-ai-contest-2024")

    #-----------------------------
    #        DataLoader
    #-----------------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    train_dataset, validation_dataset = get_dataset()

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, pin_memory=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=4)

    #----------------------------
    #         model
    #----------------------------
    ModelClass = get_model(args.model.name)

    model = ModelClass().to(args.device)

    checkpoint_path = "model_epoch_40.pth"
    model.load_state_dict(torch.load(checkpoint_path))

    #----------------------------
    #        Optimizer
    #----------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 50, 65, 80, 90], gamma=0.7) 
    criterion = MSELoss()
    writer = SummaryWriter("log")

    #----------------------------
    #      Start traning
    #----------------------------
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        try:
            # 学習
            model.train()
            train_loss = 0.0 
            validation_loss = 0.0 
            train_psnr = 0.0
            validation_psnr = 0.0
            for idx, (low_resolution_image, high_resolution_image ) in tqdm(enumerate(train_loader), desc=f"EPOCH[{epoch}] TRAIN", total=len(train_loader)):
                low_resolution_image = low_resolution_image.to(args.device)
                high_resolution_image = high_resolution_image.to(args.device)
                optimizer.zero_grad()
                output = model(low_resolution_image)
                loss = criterion(output, high_resolution_image)
                loss.backward()
                train_loss += loss.item() * low_resolution_image.size(0)
                for image1, image2 in zip(output, high_resolution_image):   
                    train_psnr += calc_psnr(image1, image2)
                optimizer.step()
            scheduler.step()

            average_train_loss = train_loss / len(train_dataset)
            average_train_psnr = train_psnr / len(train_dataset)
            
            # 検証
            model.eval()
            with torch.no_grad():
                for idx, (low_resolution_image, high_resolution_image ) in tqdm(enumerate(val_loader), desc=f"EPOCH[{epoch}] VALIDATION", total=len(val_loader)):
                    low_resolution_image = low_resolution_image.to(args.device)
                    high_resolution_image = high_resolution_image.to(args.device)
                    output = model(low_resolution_image)
                    loss = criterion(output, high_resolution_image)
                    validation_loss += loss.item() * low_resolution_image.size(0)
                    for image1, image2 in zip(output, high_resolution_image):   
                        validation_psnr += calc_psnr(image1, image2)

            average_validation_loss = validation_loss / len(validation_dataset)
            average_validation_psnr = validation_psnr / len(validation_dataset)

            writer.add_scalar("train/loss", train_loss / len(train_dataset), epoch)
            writer.add_scalar("train/psnr", train_psnr / len(train_dataset), epoch)
            writer.add_scalar("validation/loss", validation_loss / len(validation_dataset), epoch)
            writer.add_scalar("validation/psnr", validation_psnr / len(validation_dataset), epoch)
            writer.add_image("output", output[0], epoch)

            # メトリクスをコンソールに出力する
            print(f"Epoch {epoch+1} Training Loss: {average_train_loss:.7f}, Training PSNR: {average_train_psnr:.4f}")
            print(f"Epoch {epoch+1} Validation Loss: {average_validation_loss:.7f}, Validation PSNR: {average_validation_psnr:.4f}")
            # 5エポックごとにモデルを保存
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"model_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
        except Exception as ex:
            print(f"EPOCH[{epoch}] ERROR: {ex}")

    writer.close()

    # モデル生成
    torch.save(model.state_dict(), "model.pth")

    model.to(torch.device("cpu"))
    dummy_input = torch.randn(1, 3, 128, 128, device="cpu")
    torch.onnx.export(model, dummy_input, "model.onnx", 
                    opset_version=17,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {2: "height", 3:"width"}})
                
if __name__ == "__main__":
    run()