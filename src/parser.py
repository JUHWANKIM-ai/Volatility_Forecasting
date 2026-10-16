import argparse
from src.model import *

parser = argparse.ArgumentParser()
parser.add_argument("--input_size", type=int, default=2, help="input_size")
parser.add_argument("--hidden_size", type=int, default=64, help="hidden_size")
parser.add_argument("--num_layers", type=int, default=2, help="num_layers")
parser.add_argument("--output_size", type=int, default=1, help="output_size")

parser.add_argument("--retrain", type=bool, default=True, help="Retrain")
parser.add_argument("--Transfer", type=bool, default=True, help="Backbone 훈련이면 False 아니면 True")
parser.add_argument("--additional", type=bool, default=False, help="additional")
parser.add_argument("--bidirectional", type=bool, default=True, help="bidirectional")

parser.add_argument("--backbone1", type=bool, default=True, help="biLSTM")
parser.add_argument("--backbone2", type=bool, default=False, help="MLP")

opt_model = parser.parse_args()
print(opt_model)
# ---------------------------------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--model_saving_strategy", type=str, default='better', help="model_saving_strategy")
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument("--device", type=str, default='cuda', help="device")
parser.add_argument("--epochs", type=int, default=300, help="epochs")
parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--patience", type=int, default=10, help="patience")
parser.add_argument("--use_accelerator", type=bool, default=False, help="use_accelerator")
parser.add_argument("--use_wandb", type=bool, default=False, help="use_wandb")

parser.add_argument("--backbone_val_end", type=str, default='2005-01-01', help="date")
parser.add_argument("--transfer_test_end", type=str, default='2024-01-01', help="date")
parser.add_argument("--backbone_weight_path", type=str, default='weights/backbone/BiLSTM_SP.pth', help="weight")

opt_train = parser.parse_args()
print(opt_train)
# ---------------------------------------------------------------------------------------------------------------------#
backbone1 = single_biLSTM(input_size=opt_model.input_size, hidden_size=opt_model.hidden_size,
                          num_layers=opt_model.num_layers, output_size=opt_model.output_size,
                          additional=opt_model.additional, bidirectional=opt_model.bidirectional)

backbone2 = MLP(input_size=opt_model.input_size, hidden_size=opt_model.hidden_size, output_size=opt_model.output_size)
# ---------------------------------------------------------------------------------------------------------------------#
config = dict()
config['model'] = opt_model
config['train'] = opt_train

if opt_model.backbone1:
    if opt_model.Transfer:
        backbone1.load_state_dict(torch.load(opt_train.backbone_weight_path))
    config['structure'] = backbone1

if opt_model.backbone2:
    if opt_model.Transfer:
        backbone2.load_state_dict(torch.load(opt_train.backbone_weight_path))
    config['structure'] = backbone2
