import matplotlib.pyplot as plt
from src.metrics import *
from src.train import *
from torch.optim import Adam
from src.seed import seed_everything
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.dataset import CustomDataset, TestDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Run:
    def __init__(self, file_path, config):
        self.config = config
        self.file_path = file_path
        self.lr = self.config['train'].lr
        self.seed = self.config['train'].seed
        seed_everything(self.seed)

        self.epochs = self.config['train'].epochs
        self.device = self.config['train'].device
        self.batch_size = self.config['train'].batch_size
        self.train = pd.read_csv(self.file_path, index_col=0)

        self.model = self.config['structure']
        self.backbone_weight_path = self.config['train'].backbone_weight_path

        # Backbone Model 여부에 따라 weight 저장 위치를 달리함.
        if self.config['model'].backbone1:
            self.weight_path = f'weights/backbone/BiLSTM_{self.file_path[-10:-8]}.pth'
            self.saving_path = f'results/bilstm/additionalX_{self.file_path[-10:-8]}.csv'

        if self.config['model'].backbone2:
            self.weight_path = f'weights/backbone/MLP_{self.file_path[-10:-8]}.pth'
            self.saving_path = f'results/mlp/additionalX_{self.file_path[-10:-8]}.csv'

    def load_data(self, retraining):
        print('Loading data...')

        # 처음 훈련할 때 load_data 설정.
        if not retraining:
            train = self.train.loc[:self.config['train'].backbone_val_end]
            print(len(train))

            train_data, val_data = train_test_split(train, test_size=0.2, random_state=42, shuffle=False)
            train_dataset = CustomDataset(train_data)
            val_dataset = CustomDataset(val_data)
            dataloaders = {
                'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False),
                'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False),
            }

        # retraining 할때 설정.
        if retraining:
            # validation이 끝나는 시점까지 retrain data로 사용함.
            data = self.train.loc['2001-01-01':self.test_index]
            train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=False)
            train_dataset = CustomDataset(train_data)
            val_dataset = CustomDataset(val_data)
            dataloaders = {
                'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False),
                'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False),
            }

            '''retrain 주기만큼 예측이 끊나면 주기의 마지막 날짜 -20을 self.test_index에 저장하게 됨. 그 이유는 retrain 예측이 
            끊났을 때 마지막 날짜를 저장하게 되면 초반 20일은 항상 사용하지 않기 때문에 마지막 예측 날짜의 20일을 뺀 값을 사용해야
            예측이 연속적으로 이뤄지기 때문. 때문에 self.test_data는 마지막 예측 날짜의 -20일 부터 시작하는 것이 됨. '''
            self.test_data = self.train.loc[self.test_index:self.config['train'].transfer_test_end]
            self.test_dataset = TestDataset(self.test_data)
            self.test_dl = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        self.dataloaders = dataloaders

        print('Finished loading data!')

    def run_model(self, retraining):
        print(' ')
        print('Training model...')
        print(' ')

        self.load_data(retraining)

        # retraining이 True이면 마지막 레이어만 수정.
        if retraining:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True
            opt = Adam(self.model.fc.parameters(), lr=self.lr)
            self.model.to(self.device)

        # retraining이 False이면 전체 레이어 수정.
        if not retraining:
            opt = Adam(self.model.parameters(), lr=self.lr)
            self.model.to(self.device)

        lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.2, patience=self.config['train'].patience)
        parameters = {
            'num_epochs': self.epochs,
            'weight_path': self.weight_path,
            'backbone_weight_path': self.backbone_weight_path,

            'train_dl': self.dataloaders['train'],
            'val_dl': self.dataloaders['val'],

            'patience': self.config['train'].patience,
            'optimizer': opt,
            'lr_scheduler': lr_scheduler,
        }

        TM = Train_Module(self.device)
        self.model, self.loss_hist, self.metric_hist = TM.train_and_eval(self.model, parameters)

        print('Finished training model!')

    def check_validation(self):
        print(' ')
        print('Check loss and RMSE...')

        loss_hist_numpy = self.loss_hist.map(
            lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x)
        metric_hist_numpy = self.metric_hist.map(
            lambda x: x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x)

        # plot loss progress
        plt.title("Train-Val Loss")
        plt.plot(range(1, self.epochs + 1), loss_hist_numpy.iloc[:, 0], label="train")
        plt.plot(range(1, self.epochs + 1), loss_hist_numpy.iloc[:, 1], label="val")
        plt.ylabel("Loss")
        plt.xlabel("Training Epochs")
        plt.legend()
        plt.show()

        # plot accuracy progress
        plt.title("Train-Val R2")
        plt.plot(range(1, self.epochs + 1), metric_hist_numpy.iloc[:, 0], label="train")
        plt.plot(range(1, self.epochs + 1), metric_hist_numpy.iloc[:, 1], label="val")
        plt.ylabel("R2")
        plt.xlabel("Training Epochs")
        plt.legend()
        plt.show()

        print('Finished checking loss and R2!')

    def evaluate_testset(self, retrain):
        print(' ')
        print('Saving evaluations and predictions for a test set...')

        # 처음 while문 들어갈때는 test_dl이 없으므로 만들어 줘야 함. 또한 retrain을 하지 않을 때를 위해서 필요함.
        self.test_data = self.train.loc[self.config['train'].backbone_val_end:self.config['train'].transfer_test_end]
        self.test_dataset = TestDataset(self.test_data)
        self.test_dl = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        # retrain을 하기 때문에 pred dataframe을 while문 밖에서 저장할 필요가 있음.
        self.pred = pd.DataFrame(columns=['Predictions', 'Ground Truths'])
        pred_index = self.test_data.index[len(self.test_data) - len(self.test_dl):]

        '''2006-01-01부터 시작하는 첫 test_data의 index가 저장될 필요가 있음. self.test_data는 계속 변하기 때문에 
        self_data.index를 사용하면 안됨.'''
        retrain_index = self.test_data.index
        self.model.to(self.device)
        # self.model.load_state_dict(torch.load(self.weight_path))

        j = 0
        # 총 예측 가능 날짜만큼 pred가 쌓이면 종료 됨.
        while len(self.pred) < len(pred_index):

            # 처음은 retrain을 스킵해야 하기 때문에
            if retrain & (j > 0):
                # pretrained weight 에서 retrain을 진행하기 위해 필요.
                self.model.load_state_dict(torch.load(self.weight_path))
                self.run_model(True)
                # Needed to load best weights after retraining.
                # self.model.load_state_dict(torch.load(self.weight_path))

            self.model.eval()
            with ((torch.no_grad())):
                for X_train, gt in self.test_dl:
                    X_train = X_train.to(self.device)
                    output = self.model(X_train)
                    output = output.cpu().detach().numpy().tolist()
                    gt = gt.squeeze().detach().numpy().tolist()
                    self.pred.loc[len(self.pred)] = [output, gt]

                    # 마지막 시행은 retrain if 문안으로 들어가면 안됨. total로 방지.
                    total = (len(self.pred)) / len(pred_index) * 100
                    if retrain & (total < 100):
                        '''이 부분에서 self.test_index를 저장하는데 retrain_index를 사용해야 2006-01-01부터 100일 이후 날짜가
                        저장됨. <=> 주기의 마지막 날짜 -20 과 동치.'''
                        self.test_index = retrain_index[len(self.pred)]
                        print(self.test_index)
                        break
            j = 1
            print(f'\n{len(self.pred)}/{len(pred_index)}: {len(self.pred) / len(pred_index) * 100:.2f}%')
            time.sleep(2)

        self.pred.index = pred_index
        mae = calculate_mae(self.pred['Ground Truths'].values, self.pred['Predictions'].values)
        rmse = calculate_rmse(self.pred['Ground Truths'].values, self.pred['Predictions'].values)
        r2 = calculate_r2_score(self.pred['Ground Truths'].values, self.pred['Predictions'].values)
        self.pred.to_csv(self.saving_path)

        print(' ')
        print(f'Model: {self.backbone_weight_path}')
        print(f'MAE: {mae}')
        print(f'RMSE: {rmse}')
        print(f'R^2: {r2:.4f}')
        print(f"Saved Result in {self.saving_path}!")
