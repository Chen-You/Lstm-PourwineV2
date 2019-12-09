import os
import math
import time
import random

import numpy as np
import torch
import torchsnooper
import matplotlib.pyplot as plt
from torch import nn, optim
from tensorboardX import SummaryWriter

from config import params
from device import getDevice
from model import Model
from customDataset import dataPreprocess

def timeSince(start):
    s = time.time() - start
    m = math.floor(s/60)
    s -= m*60

    return f"{m:2d}m {s:2.1f}s"

# @torchsnooper.snoop()
def trainIters(task, device, model, train_loader, valid_loader, params):
    if params['mode'] == 'train':
        comment = "{}_n{}h{}p{}".format(task, params['num_layers'], params['hidden_size'], params['previous_state'])
        writer = SummaryWriter(comment=comment)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        criterion = nn.MSELoss()
        start = time.time()
        for epoch in range(params['epochs']):
            total_loss = {'train': 0.0, 'valid': 0.0}

            model.train()
            for idx, data in enumerate(train_loader):
                model.zero_grad()
                hidden_tensor = model.initHidden().to(device)

                loss = 0
                input_tensor = data[:, 0:1, :]
                input_tensor = input_tensor.repeat(1, params['previous_state'], 1)
                use_teacher_forcing = random.random() < params['teacher_forcing_ratio']
                for i in range(1, data.shape[1]):
                    input_tensor = input_tensor.to(device)
                    target_tensor = data[:, i:i+1, :]
                    target_tensor = target_tensor.to(device)
                    output_tensor, hidden_tensor = model(input_tensor.view(params['batch_size'], 1, -1), hidden_tensor)
                    loss += criterion(output_tensor, target_tensor[:,:,-7:])

                    if use_teacher_forcing:
                        input_tensor = torch.cat((input_tensor, target_tensor), axis=1)[:, 1:, :]
                    else:
                        tmp_tensor = torch.cat((target_tensor[:, :, :-7], output_tensor.detach()), axis=2)
                        input_tensor = torch.cat((input_tensor, tmp_tensor), axis=1)[:, 1:, :]

                loss.backward()
                optimizer.step()
                total_loss['train'] += loss/data.size(1)

            total_loss['train'] /= len(train_loader)
            print('[%d, %d][%s] train loss: %f'%(epoch+1, params['epochs'], timeSince(start), total_loss['train']), end=', ')
            print()

            model.eval()
            with torch.no_grad():
                for idx, data in enumerate(valid_loader):
                    hidden_tensor = model.initHidden().to(device)

                    loss = 0

                    input_tensor = data[:, 0:1, :]
                    input_tensor = input_tensor.repeat(1, params['previous_state'], 1)
                    for i in range(1, data.shape[1]):
                        input_tensor = input_tensor.to(device)
                        target_tensor = data[:, i:i+1, :]
                        target_tensor = target_tensor.to(device)
                        output_tensor, hidden_tensor = model(input_tensor.view(params['batch_size'], 1, -1), hidden_tensor)
                        loss += criterion(output_tensor, target_tensor[:,:,-7:])
                        # if i%250 == 0 and idx+1 == len(valid_loader):
                        #     print(i)
                        #     print('i', input_tensor[0])
                        #     print('t', target_tensor[0])
                        #     print('p', output_tensor[0])
                        #     print()

                        tmp_tensor = torch.cat((target_tensor[:, :, :-7], output_tensor.detach()), axis=2)
                        input_tensor = torch.cat((input_tensor, tmp_tensor), axis=1)[:, 1:, :]
                        # input_tensor = torch.cat((input_tensor, output_tensor.detach()), axis=1)[:, 1:, :]

                    total_loss['valid'] += loss.item()/data.size(1)

                total_loss['valid'] /= len(valid_loader)
                print('[%d, %d][%s] valid loss: %f'%(epoch+1, params['epochs'], timeSince(start), total_loss['valid']), end=', ')

            writer.add_scalars('data/loss', {'train_loss': total_loss['train'], 'valid_loss': total_loss['valid']}, epoch)

            print()
            # break

            if total_loss['valid'] < params['min_loss']:
                ckpt_name = "{}_{}_n{}h{}p{}.pt".format(task, epoch+1, params['num_layers'], params['hidden_size'], params['previous_state'])
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(params['model_path'], ckpt_name))
                print(f"Save {ckpt_name} under {params['model_path']}/")
                params['min_loss'] = total_loss['valid']
                params['curr_early_stop'] = params['early_stop']
            else:
                print(f"total_loss['valid']({total_loss['valid']:.6f}) > min_loss({params['min_loss']:.6f})")
                params['curr_early_stop'] -= 1
                print(f"curr_early_stop: {params['curr_early_stop']}")

            print()
            if params['curr_early_stop'] == 0:
                writer.close()
                break
    elif params['mode'] == 'eval':
        # evaluation
        eval_model = model
        eval_model = eval_model.to(device)
        eval_model.load_state_dict(torch.load(os.path.join(params['model_path'], params['ckpt_name']))['model_state_dict'])
        eval_model.eval()
        with torch.no_grad():
            for idx, data in enumerate(valid_loader):
                hidden_tensor = eval_model.initHidden()
                hidden_tensor = hidden_tensor.to(device)

                data = sample['data']
                loss = 0
                target_buf = []
                predict_buf = []

                input_tensor = data[:, 0:1, :]
                input_tensor = input_tensor.repeat(1, params['previous_state'], 1)
                print(data.shape[1])
                for i in range(1, data.shape[1]):
                    input_tensor = input_tensor.to(device)
                    target_tensor = data[:, i:i+1, :]
                    target_tensor = target_tensor.to(device)
                    output_tensor, hidden_tensor = model(input_tensor.view(params['batch_size'], 1, -1), hidden_tensor)
                    target_buf.append(target_tensor.cpu().numpy()[0][:, -7:])
                    predict_buf.append(output_tensor.cpu().detach().numpy()[0][:, -7:])
                    # if i%250 == 0 and idx+1 == len(valid_loader):
                    if i%250 == 0:
                        print('i', input_tensor[0])
                        print('t', target_tensor[0][0][-14:])
                        print('p', output_tensor[0])
                        print()

                    student_tensor = torch.cat((target_tensor[:, :, :-14], output_tensor.detach()), axis=2)
                    input_tensor = torch.cat((input_tensor[:, 1:, :], student_tensor), axis=1)
                    # input_tensor = torch.cat((input_tensor[:, 1:, :], output_tensor.detach()), axis=1)
                break


def main():
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    for k, v in params.items():
        print(f"{k}: {v}")
    print()

    device = getDevice(params['device_id'])

    train_loader, valid_loader, norm= dataPreprocess(params['task'], params['file_path'], params['valid_ratio'])
    model = Model(params).to(device)
    trainIters(params['task'], device, model, train_loader, valid_loader, params)

if __name__ == '__main__':
    main()

