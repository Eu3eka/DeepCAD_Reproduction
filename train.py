from collections import OrderedDict
from tqdm import tqdm
import argparse
from dataset.cad_dataset import get_dataloader
from config import ConfigAE
from utils import cycle
from trainer import TrainerAE


def main():
    # create experiment cfg containing all hyperparameters
    cfg = ConfigAE('train')

    # create network and training agent
    tr_agent = TrainerAE(cfg)

    # load from checkpoint if provided
    if cfg.cont:
        tr_agent.load_ckpt(cfg.ckpt)

    # create dataloader
    train_loader = get_dataloader('train', cfg)
    val_loader = get_dataloader('validation', cfg)
    val_loader_all = get_dataloader('validation', cfg)
    val_loader = cycle(val_loader)

    # start training
    clock = tr_agent.clock

    for e in range(clock.epoch, cfg.nr_epochs):
        # begin iteration
        pbar = tqdm(train_loader)
        #print(f"[train.py-main]:\n")
        for b, data in enumerate(pbar):
            # train step
            # 打印每个 batch 的大小和内容的 key
            #print(f"Batch {b}:")
            #print(f"Data keys: {data.keys()}")  # 打印数据的 keys，比如 'command' 和 'args'
            #print(f"Command shape: {data['command'].shape}")
            #print(f"Args shape: {data['args'].shape}")
            outputs, losses = tr_agent.train_func(data)

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            # validation step
            if clock.step % cfg.val_frequency == 0:
                data = next(val_loader)
                outputs, losses = tr_agent.val_func(data)

            clock.tick()

            tr_agent.update_learning_rate()

        if clock.epoch % 5 == 0:
            tr_agent.evaluate(val_loader_all)

        clock.tock()

        if clock.epoch % cfg.save_frequency == 0:
            tr_agent.save_ckpt()

        # if clock.epoch % 10 == 0:
        tr_agent.save_ckpt('latest')


if __name__ == '__main__':
    main()
