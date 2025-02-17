# import wandb
import torch
import torch.nn as nn
import numpy as np
import time
import json
import sys
import os
from models import *
from utils import *
from entailment import *
import logging
import geoopt

def main_train(model, trainloader, opt, device = 'cpu', K = 0.1):
    model.train()
    avgloss = 0.
    acc = 0
    
    criterion = Entailment_loss(K = K)

    for bidx, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        y = y.squeeze()
        opt.zero_grad()
        out = model(x)
        loss = criterion(out, model.prototypes, y)
        loss.backward()
        avgloss += loss.item()
        with torch.no_grad():
            pred = predict_entailment(out, model.prototypes)
        
        pred = pred.squeeze()
        acc += (pred == y).sum().item() / len(y)  
        opt.step()
    return model, acc/(bidx+1), avgloss/(bidx+1)

def main_test(model, testloader, device):  
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data,y in testloader:
            data = data.to(device)
            y = y.to(device)
            
            y = y.squeeze()
            true_labels.append(y)

            output = model(data)

            pred = predict_entailment(output, model.prototypes)
            
            pred = pred.squeeze().to(device)
            
            predictions.append(pred)
    
    true_labels = torch.cat(true_labels, dim=0)
    predictions = torch.cat(predictions, dim=0)
    acc = (true_labels == predictions).sum().item() / len(true_labels)
    return acc, predictions, true_labels

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="classification")
    parser.add_argument('-device',dest='device', default='cpu', type = str, help='device')
    parser.add_argument('-config',dest='config', default='configs/config.json', type = str, help='device')
    parser.add_argument('-rank',dest='rank', default=0, type = int, help='ranking of the run')
    parser.add_argument('-temp',dest='temperature', default=0.01, type = float, help='geometry of the output')
    parser.add_argument('-dim', dest = 'dim', default=8, type = int, help = 'embedding dimension')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    with open(args.config) as json_file:
        config = json.load(json_file)

    torch.manual_seed(args.rank)
    config['device'] = args.device
    
    # run = wandb.init(project="your_project",config = config)


    name_file = f"{config['dataset']}_{args.rank}"
    logs_directory = f"logs_{config['dataset']}"
    
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)

    directory = f"results_{config['dataset']}"
    if not os.path.exists(directory):
        os.makedirs(directory)


    logging.basicConfig(filename=logs_directory+'/'+name_file+'.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    specific_name = f"/{name_file}"
    fileName_res = directory+specific_name

    logging.info(config)

    trainloader, testloader = load_dataset(config['dataset'], 
                                           config['batch_size'], 
                                           num_workers = 8, 
                                           val= config['validation'])
    
    manifold = geoopt.PoincareBallExact(c=1, learnable = False)

    model = load_backbone(config['dataset'], 
                          config['output_dim']) 
    
    model = Model(model,
                device = config['device'],
                dataset = config['dataset'],
                output_dim = config['output_dim'],
                temperature = config['temperature'],
                clipping = config['clipping'],
                manifold = manifold,
                prototypes_ray=config['prototypes_ray'],
                margin=config['margin'],
                seed = args.rank)

    model = model.to(config['device'])
    opt = load_optimizer(model.parameters(), *list(config['optimizer'].values()))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, config['lr_scheduler']['steps'], gamma=config['lr_scheduler']['entity'])
    
    total_start_time = time.time()
    for epoch in range(0, config['epochs']):

        sys.stdout.flush()
        t0 = time.time()
                    
        final_model, acc, loss_calculated  = main_train(model, 
                                                        trainloader, 
                                                        opt = opt, 
                                                        device = config['device'], 
                                                        K = config['K'])
        # wandb.log({"training_acc": acc, "loss": loss_calculated}, step = epoch)
        
        t1 = time.time()
        logging.info(f'Training at epoch {epoch}: Accuracy = {round(acc*100,4)} ; Loss = {round(loss_calculated,4)}; time = {round(t1-t0,2)}')
        print(f'Training at epoch {epoch}: Accuracy = {round(acc*100,4)} ; Loss = {round(loss_calculated,4)}; time = {round(t1-t0,2)}')
        scheduler.step()
        torch.save({
            'epoch': epoch,
            'model_state_dict': final_model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            },  fileName_res+'_model.pt')
    
    logging.info("Total time --- %s seconds ---" % (time.time() - total_start_time))
    test_acc, test_AHC, test_prediction, test_tl = main_test(model, testloader, device = config['device'])
    logging.info(f'test Accuracy = {round(100*test_acc,4)}')
    # wandb.log({"test_acc": test_acc})
    logging.info("saving the results")
    
    np.save(fileName_res+'.npy', test_prediction.cpu().numpy())
    np.save(fileName_res+'_tl.npy', test_tl.cpu().numpy())
    # wandb.finish()

