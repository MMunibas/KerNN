#!/usr/bin/env python3

# general imports
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# torch imports
import torch
import torch.nn as nn

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# utils imports
from utils.plot_corr import plot_corr_efmu
from utils.neuralnets.FFNet import FFNet
from utils.distances import get_bond_length_hoxa
from utils.kernels import get_1D_kernels_k33

# define constants
evtokcal = 23.060541945

# define options
torch.set_printoptions(precision=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Running NN on", device)

# load actual data for h2o
data = np.load("datasets/hydrogen_oxalate_mp2_avtz_gen2_22110.npz")
ndata = len(data["E"])
natoms = len(data["R"][0])
nintdist = int(natoms *(natoms -1) /2)

"""


fig, ax = plt.subplots(1, 2)
for i in range(nintdist):
    ax[0].hist(k[:, i].detach().numpy())
    
for i in range(nintdist):
    ax[1].hist(indist[:, i].detach().numpy())

plt.show()
print(k)
quit()
"""


ntrains = [17800]
nvalid = 2200

for ntrain in ntrains:

    # define some variables
    n_input, n_hidden, n_out, batch_size, valid_batch_size, learning_rate = nintdist, 50, 1+natoms, 43, nvalid, 0.0001
    fweight, dweight, qweight = 10, 5, 2
    seeds = [42]#[41, 42, 43, 44]
    for seed in seeds:

        # do train/val/test split
        # random state parameter, such that random operations are reproducible if wanted
        random_state = np.random.RandomState(seed=seed)

        # create shuffled list of indices
        idx = random_state.permutation(np.arange(ndata))

        # store indices of training, validation and test data
        idx_train = idx[0:ntrain]
        idx_valid = idx[ntrain:ntrain + nvalid]
        idx_test = idx[ntrain + nvalid:]

        # load positions and calculate interatomic distances
        # do this in here to allow use of autodiff
        pos = torch.tensor(np.float32(data["R"]))

        # load Energies
        E = torch.from_numpy(np.float32(data["E"])).unsqueeze(1)
        
        # load Forces
        F = torch.from_numpy(np.float32(data["F"]))
        
        # load Dipole moments
        D = torch.from_numpy(np.float32(data["D"]))
        
        # load total charge
        Q = torch.from_numpy(np.float32(data["Q"]))


        #find minimum energy configuration
        minidx = torch.argmin(E)
        minpos = pos[minidx]
        minr = get_bond_length_hoxa(minpos, nintdist)
        #print("minR ", minr) 
        
        indist = get_bond_length_hoxa(pos, nintdist)

        k = get_1D_kernels_k33(indist, minr, 1) 
        kmean = torch.mean(k, axis=0)
        kstd = torch.std(k, axis=0)
        #print("kmean ", kmean) 
        #print("kstd ", kstd)
         
        #####################
        # define TRIAINING SET
        #####################
        pos_train = pos[idx_train]
        E_train = E[idx_train]
        F_train = F[idx_train]
        D_train = D[idx_train]
        Q_train = Q[idx_train]
        
        #####################
        # define VALIDATION SET
        #####################
        pos_valid = pos[idx_valid]
        E_valid = E[idx_valid]
        F_valid = F[idx_valid]
        D_valid = D[idx_valid]
        Q_valid = Q[idx_valid]
        
        #####################
        # define TEST SET
        #####################
        pos_test = pos[idx_test]
        E_test = E[idx_test]
        F_test = F[idx_test]
        D_test = D[idx_test]
        Q_test = Q[idx_test]

        # then calculate statistics
        meanE = torch.mean(E_train)
        stdE = torch.std(E_train)
        #print("meanE", meanE)
        #print("stdE", stdE)
        #quit()
        

        train_loader = torch.utils.data.DataLoader(list(zip(pos_train, E_train, F_train, D_train, Q_train)), batch_size=batch_size,
                                                   shuffle=True, drop_last=True, num_workers=0)
        valid_loader = torch.utils.data.DataLoader(list(zip(pos_valid, E_valid, F_valid, D_valid, Q_valid)), batch_size=batch_size,
                                                   shuffle=False, drop_last=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(list(zip(pos_test, E_test, F_test, D_test, Q_test)), batch_size=batch_size,
                                                  shuffle=False, drop_last=True)

        # build/define neural network architecture
        model = FFNet(n_input,n_hidden,n_out).to(device)


        # Assign Exponential Moving Average model
        #see https://github.com/fadel/pytorch_ema
        use_ema=True
        if use_ema:
            from torch_ema import ExponentialMovingAverage
            trainer_ema_model = ExponentialMovingAverage(
                model.parameters(),
                decay=0.999)


        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of parameters", count_parameters(model))


        # define loss function and optimizer
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)


        def train_one_epoch(epoch_index, tb_writer):
            running_loss = 0.
            running_eloss = 0.
            running_floss = 0.
            running_dloss = 0.
            running_qloss = 0.
            last_loss = 0.


            # Here, we use enumerate(training_loader) instead of
            # iter(training_loader) so that we can track the batch
            # index and do some intra-epoch reporting

            for i, data in enumerate(train_loader, 0):
                

                # Every data instance contains (batch of) pos, E, F
                pos_batch, E_batch, F_batch, D_batch, Q_batch = data[0].to(device), data[1][:, 0].to(device), data[2].to(device),  data[3].to(device),  data[4].to(device)

                pos_batch.requires_grad_()
                R_batch = get_bond_length_hoxa(pos_batch, nintdist)
                k_batch = ((get_1D_kernels_k33(R_batch, minr, 1) - kmean) / kstd).to(device)
                
                #print(pos_batch.device, E_batch.device, F_batch.device, k_batch.device, meanE.device, stdE.device)

                # Zero your gradients for every batch!
                optimizer.zero_grad(set_to_none=True)
                

                # Make predictions for this batch
                outputs = model(k_batch)
                energy = outputs[:, 0] * stdE + meanE #would be interesting to make these learnable.
                charges = outputs[:, 1:]

                qtot = torch.sum(charges, dim=1)
    
      
                #compute forces using autograd
                forces = -torch.autograd.grad(torch.sum(energy), pos_batch, create_graph=True)[0]

                dipoles = torch.einsum('sm,smh->sh', charges, pos_batch)
  

                # Compute the loss and its gradients
                eloss = loss_function(energy, E_batch)
                floss = loss_function(forces, F_batch)
                dloss = loss_function(dipoles, D_batch)
                qloss = loss_function(qtot, Q_batch)

                loss = eloss + fweight * floss + dweight * dloss + qweight * qloss
                loss.backward(retain_graph=True)


                # Adjust learning weights
                optimizer.step()
                

                # Apply Exponential Moving Average / Update the moving average with the new parameters from the last optimizer step
                if use_ema:
                    trainer_ema_model.update()

                # Gather data and report
                running_loss += loss.item()
                running_eloss += eloss.item()
                running_floss += floss.item()
                running_dloss += dloss.item()
                running_qloss += qloss.item()
                
                #here I want to calculate an average per-batch loss, i.e. at the end of the epoch.
                if i == (len(train_loader) - 1):

                    last_loss = running_loss / len(train_loader)  # loss per batch
                    last_eloss = running_eloss / len(train_loader)
                    last_floss = running_floss / len(train_loader)
                    last_dloss = running_dloss / len(train_loader)
                    last_qloss = running_qloss / len(train_loader)
                    
                    print('  batch {} loss: {}'.format(i + 1, last_loss))
                    step = epoch_index * len(train_loader) + i + 1

                    tb_writer.add_scalar('Loss/loss_train', last_loss, step)
                    tb_writer.add_scalar('ELoss/eloss_train', last_eloss, step)
                    tb_writer.add_scalar('FLoss/floss_train', last_floss, step)
                    tb_writer.add_scalar('DLoss/dloss_train', last_dloss, step)
                    tb_writer.add_scalar('QLoss/qloss_train', last_qloss, step)
                    running_loss = 0.

                
            return last_loss, last_eloss, last_floss, last_dloss, last_qloss

        Train = True
        if Train:
            # Initializing in a separate cell so we can easily add more epochs to the same run
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            writer = SummaryWriter('runs/trainer_{}'.format(timestamp))
            epoch_number = 0

            EPOCHS = 10
            stopper = 0

            best_vloss = 10000000
            deltat = 0
            for epoch in range(EPOCHS):
                start = time.time()
                print(epoch, deltat)
                print('EPOCH {}:'.format(epoch_number + 1))

                # Make sure gradient tracking is on, and do a pass over the data
                model.train(True)
                avg_loss, avg_eloss, avg_floss, avg_dloss, avg_qloss = train_one_epoch(epoch_number, writer)

                running_vloss = 0.0
                running_eloss_valid = 0.0
                running_floss_valid = 0.0
                running_dloss_valid = 0.0
                running_qloss_valid = 0.0
                
                # Set the model to evaluation mode, disabling dropout and using population
                # statistics for batch normalization.
                model.eval()
                
                if use_ema:

                    trainer_ema_model.store()
                    trainer_ema_model.copy_to()
                    for i, vdata in enumerate(valid_loader):

                        pos_valid, E_valid, F_valid, D_valid, Q_valid = vdata[0].to(device), vdata[1][:, 0].to(device), vdata[2].to(device), vdata[3].to(device), vdata[4].to(device)
                        pos_valid.requires_grad_()
                        R_valid = get_bond_length_hoxa(pos_valid, nintdist)
                        k_valid = ((get_1D_kernels_k33(R_valid, minr, 1) - kmean) / kstd).to(device)


                        voutputs = model(k_valid)
                        venergy = voutputs[:, 0]  * stdE + meanE
                        vcharges = voutputs[:, 1:]
                        
                        vqtot = torch.sum(vcharges, dim=1)
                        
                        
                        vforces = -torch.autograd.grad(torch.sum(venergy), pos_valid, create_graph=True)[0]
                        
                        vdipoles = torch.einsum('sm,smh->sh', vcharges, pos_valid)


                        # Compute the loss and its gradients
                        eloss_valid = loss_function(venergy, E_valid)
                        floss_valid = loss_function(vforces, F_valid)
                        dloss_valid = loss_function(vdipoles, D_valid)
                        qloss_valid = loss_function(vqtot, Q_valid)
                        
                        vloss = eloss_valid + fweight * floss_valid + dweight * dloss_valid + qweight *qloss_valid 
                        
                        running_vloss += vloss
                        running_eloss_valid += eloss_valid
                        running_floss_valid += floss_valid
                        running_dloss_valid += dloss_valid
                        running_qloss_valid += qloss_valid
                        
                    avg_vloss = running_vloss / (i + 1)
                    avg_eloss_valid = running_eloss_valid / (i + 1)
                    avg_floss_valid = running_floss_valid / (i + 1)
                    avg_dloss_valid = running_dloss_valid / (i + 1)
                    avg_qloss_valid = running_qloss_valid / (i + 1)
                    
                    print('EMA LOSS train {} valid {}'.format(avg_loss, avg_vloss))
                    
                    # Log the running loss averaged per batch
                    # for both training and validation
                    writer.add_scalars('Loss/Train vs. Valid Loss',
                                       {'loss_train': avg_loss, 'loss_valid': avg_vloss},
                                       epoch_number + 1)

                    writer.add_scalars('ELoss/Train vs. Valid Loss',
                                       {'eloss_train': avg_eloss, 'eloss_valid': avg_eloss_valid},
                                       epoch_number + 1)

                    writer.add_scalars('FLoss/Train vs. Valid Loss',
                                       {'floss_train': avg_floss, 'floss_valid': avg_floss_valid},
                                       epoch_number + 1)

                    writer.add_scalars('DLoss/Train vs. Valid Loss',
                                       {'dloss_train': avg_dloss, 'dloss_valid': avg_dloss_valid},
                                       epoch_number + 1)
                                       
                    writer.add_scalars('QLoss/Train vs. Valid Loss',
                                       {'qloss_train': avg_qloss, 'qloss_valid': avg_qloss_valid},
                                       epoch_number + 1)

                    writer.flush()
                    
                    # Track best performance, and save the model's state
                    print("stopper:", stopper)
                    if avg_vloss < best_vloss:
                        best_vloss = avg_vloss
                        model_path_ema = 'models_ntrain{}/model_{}_{}_seed{}'.format(ntrain, timestamp, epoch_number, seed) + "_ema"
                        torch.save(model.state_dict(), model_path_ema)
                        stopper = 0
                    else:
                        stopper += 1

                    if stopper > 2000:
                        print("The validation loss has not improved for", str(stopper), "epochs.")
                        print("Exiting")
                        break
                        
                        
                    trainer_ema_model.restore()
                    
                    

                else:
                    print("PLEASE USE EMA. Exiting...")
                    quit()

                epoch_number += 1
                deltat = time.time() - start

            
            
        eval_ema = True
        if eval_ema:
            device ="cpu"
            #model_path_ema="models2_ntrain3200/model_20240419_091538_999_seed141_ema"
            model.load_state_dict(torch.load(model_path_ema))
            
            pos_test.requires_grad_()
            R_test = get_bond_length_hoxa(pos_test, nintdist)
            k_test = ((get_1D_kernels_k33(R_test, minr, 1) - kmean) / kstd).to(device)

            model.to(device)
            toutputs = model(k_test)
            e_pred = toutputs[:, 0] * stdE + meanE
            tcharges = toutputs[:, 1:]
            print(torch.sum(tcharges, dim=1))

            
            forces = -torch.autograd.grad(torch.sum(e_pred), pos_test, create_graph=True)[0]

            tdipoles = torch.einsum("sm,smh->sh", tcharges, pos_test)

            e_pred = (e_pred).cpu().detach().numpy() * evtokcal
            e_true = E_test[:, 0].detach().numpy() * evtokcal


            plot_corr_efmu(e_true, e_pred, F_test.detach().numpy() * evtokcal, forces.detach().numpy() * evtokcal, D_test.detach().numpy(), tdipoles.detach().numpy(), model_path_ema)

