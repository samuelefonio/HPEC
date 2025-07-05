# import wandb
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import pickle as pkl
from utils import *
import time
import json
import sys
import os
from models import *
import logging
import geoopt
import copy
from utils import HPS_loss

def main_train(method, model, trainloader, opt, device = 'cpu', proto_opt = None):
    model.train()
    avgloss = 0.
    acc = 0

    if method in ['CHPS', 'HBL', 'ECL', 'XE', 'NF']:
        criterion = nn.CrossEntropyLoss()
    elif method == 'HPS':
        criterion  = HPS_loss(prototypes=model.prototypes)
    
    for bidx, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        y = y.squeeze()
        opt.zero_grad()
        if proto_opt is not None:
            proto_opt.zero_grad()
        out = model(x)

        loss = criterion(out, y)
        
        loss = loss
        loss.backward()
        
        avgloss += loss.item()

        pred = prediction(method, out, model.prototypes)
        
        pred = pred.squeeze()
        
        acc += (pred == y).sum().item() / len(y)  
        opt.step()
        if proto_opt is not None:
            proto_opt.step()
    return model, acc/(bidx+1), avgloss/(bidx+1)

def main_test(model, testloader, device, method):  
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

            pred = prediction(method, output, model.prototypes)
            
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
    parser.add_argument('-config',dest='config', default='config.json', type = str, help='device')
    parser.add_argument('-seed',dest='seed', default=0, type = int, help='seed of the run')
    parser.add_argument('-temp',dest='temperature', default=0.01, type = float, help='geometry of the output')
    parser.add_argument('-dim', dest = 'dim', default=8, type = int, help = 'embedding dimension')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    # Receiving the characteristics of the experiment
    args = parse_args()
    with open(args.config) as json_file:
        config = json.load(json_file)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    
    
    config['seed'] = args.seed
    # run = wandb.init(project="your_project", config = config)


    name_file = f"{config['method']}_{config['dataset']}_{args.seed}"
    logs_directory = f"logs_{config['dataset']}"
    
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)

    directory = f"results_{config['dataset']}"
    if not os.path.exists(directory):
        os.makedirs(directory)


    logging.basicConfig(filename=logs_directory+'/'+name_file+'.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    specific_name = f"/{name_file}"
    specific_name_proto = f"/proto_{name_file}"
    fileName_res = directory+specific_name
    fileName_proto = directory+specific_name_proto

    logging.info(config)

    trainloader, testloader = load_dataset(config['dataset'], 
                                           config['batch_size'], 
                                           num_workers = 8, 
                                           val= config['validation'])
    
    manifold = geoopt.PoincareBallExact(c=1, learnable = False)

    model = load_backbone(config['dataset'], 
                          config['output_dim']) 
    
    model = load_model(model, 
                       config['method'],
                       config['device'], 
                       config['dataset'], 
                       config['output_dim'], 
                       config['temperature'],
                       config['clipping'],
                       manifold,
                       config['seed'])
    
    model = model.to(config['device'])
    
    if config['method'] == 'HBL':
        filtered_parameters = [p for name, p in model.named_parameters() if 'proto' not in name]
        proto_params = [p for name, p in model.named_parameters() if 'proto' in name]
        optimizer_parameters = [{'params': filtered_parameters}]
        opt = load_optimizer(optimizer_parameters, *list(config['optimizer'].values()))
        proto_opt = geoopt.optim.RiemannianSGD(proto_params, lr = config['proto_lr'], momentum=0.9, dampening=0, weight_decay=0, nesterov=False, stabilize=None)
    else:
        opt = load_optimizer(model.parameters(), *list(config['optimizer'].values()))
        proto_opt = None
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, config['lr_scheduler']['steps'], gamma=config['lr_scheduler']['entity'])

    total_start_time = time.time()
    best_val = 0
    model_epoch = 0
    total_norm = []
    prototype_dict = {i:0 for i in range(config['epochs'])}
    checkpoint_epoch = 0
    for epoch in range(checkpoint_epoch, config['epochs']):

        sys.stdout.flush()
        t0 = time.time()
        
        prototype_dict[epoch] = model.prototypes.clone().detach().cpu().numpy() 
                    
        final_model, acc, loss_calculated = main_train(config['method'],
                                                                    model, 
                                                                    trainloader, 
                                                                    opt = opt,
                                                                    device = config['device'], 
                                                                    proto_opt = proto_opt)
        # wandb.log({"training_acc": acc, "loss": loss_calculated, "avg_norm": avg_norm}, step = epoch)
        if epoch%20 == 0:
            test_acc, test_prediction, test_tl = main_test(model, testloader, device = config['device'], method = config['method'])
            logging.info(f'valid Accuracy = {round(100*test_acc,4)}')
            # wandb.log({"valid_acc": test_acc}, step = epoch)
        t1 = time.time()
        logging.info(f'Training at epoch {epoch}: Accuracy = {round(acc*100,4)} ; Loss = {round(loss_calculated,4)} ; time = {round(t1-t0,2)}')
        print(f'Training at epoch {epoch}: Accuracy = {round(acc*100,4)} ; Loss = {round(loss_calculated,4)} ; time = {round(t1-t0,2)}')
        
        scheduler.step()
        torch.save({
            'epoch': epoch,
            'model_state_dict': final_model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            },  fileName_res+'_model.pt')
    
    logging.info("Total time --- %s seconds ---" % (time.time() - total_start_time))
    test_acc, test_prediction, test_tl = main_test(model, testloader, device = config['device'], method = config['method'])
    logging.info(f'test Accuracy = {round(100*test_acc,4)}')
    # wandb.log({"test_acc": test_acc})
    logging.info("saving the results")
    
    np.save(fileName_res+'.npy', test_prediction.cpu().numpy())
    np.save(fileName_res+'_tl.npy', test_tl.cpu().numpy())
    np.save(fileName_res+'norms.npy', np.array(total_norm))
    with open(fileName_proto+'.pkl', "wb") as pickle_file:
        pkl.dump(prototype_dict, pickle_file)
    logging.info("results_saved")
    # wandb.finish()

