from glob import glob
from src.run import Run
from src.parser import config
from multiprocessing import set_start_method, Process

import os
import torch
import time


class Execution:
    def __init__(self, directory, saving_path):
        self.config = config
        self.directory = directory
        self.saving_path = saving_path

    def get_file_list(self):
        self.list = glob(f"{self.directory}/*.csv")
        if self.saving_path is not None:
            self.list = [file for file in self.list if not os.path.exists(f"{self.saving_path}/{file.split('/')[-1]}")]
        self.list.sort()
        return self.list

    def setup(self, rank):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        torch.cuda.set_device(rank)
        print(f"Process on GPU: {torch.cuda.current_device()}")
        # logging.info(f"Process on GPU: {torch.cuda.current_device()}")

    def process_file(self, rank, file):
        self.setup(rank)

        trainer = Run(file, self.config)
        trainer.run_model(False)
        trainer.check_validation()
        trainer.evaluate_testset(self.config['model'].retrain)

    # def main(self):
    #     set_start_method('spawn', force=True)
    #     world_size = 1  # 사용 가능한 GPU 수
    #     file_list = self.get_file_list()
    #
    #     for i in range(0, len(file_list), world_size):
    #
    #         current_batch = file_list[i:i + world_size]
    #         processes = []
    #         for j, file_path in enumerate(current_batch):
    #             rank = j % world_size
    #             p = Process(target=self.process_file, args=(rank, file_path))
    #             p.start()
    #             processes.append(p)
    #
    #         for p in processes:
    #             p.join()

    def main(self):
        if self.config['model'].Transfer:
            file_list = self.get_file_list()
        if not self.config['model'].Transfer:
            file_list = ['data/spot_std/SP_vol.csv']

        for file in file_list:
            print('')
            print(f'{file} will be started...')
            time.sleep(3)
            trainer = Run(file, self.config)
            if not self.config['model'].Transfer:
                trainer.run_model(False)
                trainer.check_validation()
            trainer.evaluate_testset(self.config['model'].retrain)


if __name__ == "__main__":
    E = Execution('data/future_std', None)
    E.main()
