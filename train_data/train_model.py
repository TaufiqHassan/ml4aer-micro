import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from data_transformations import apply_transformation, inverse_standard_transform
from NN_models import SimpleNN    

# Set the device to MPS if available, otherwise default to CPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# print('Training On: ', ('mps' if torch.backends.mps.is_available() else 'cpu'))

class ModelTrainer(object):
    def __init__(self,data_path,**kwargs):
        self.data_path = data_path
        self.model_id = kwargs.get('model_id','None')
        self.natmos = kwargs.get('natmos', 20)
        self.batch_size = kwargs.get('batch_size', 256)
        self.depths = kwargs.get('depths', [2])
        self.widths = kwargs.get('widths', [128])
        self.learning_rate = kwargs.get('learning_rate', 0.0003)
        self.weight_decay = kwargs.get('weight_decay', 0.0)
        self.n_models = kwargs.get('n_models', 1)
        self.n_epochs = kwargs.get('n_epochs', 100)
        self.loss_func = kwargs.get('loss_func', nn.SmoothL1Loss())
        
        self.varlist = kwargs.get('varlist', ['bc_a1','bc_a4',\
                                              'pom_a1','pom_a4',\
                                              'so4_a1','so4_a2','so4_a3',\
                                              'mom_a1','mom_a2','mom_a4',\
                                              'ncl_a1','ncl_a2',\
                                              'soa_a1','soa_a2','soa_a3',\
                                              'num_a1','num_a2','num_a4',\
                                              'H2SO4','SOAG'])
            
        self.transformation_type = kwargs.get('transformation_type', 'standardization')
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') 
        
        # Load the data
        self._load_data()
        self._prep_data()

    def _load_data(self):
        """Load the training, validation, and testing datasets."""
        self.data_path = Path(self.data_path)
        # Training data
        self.Xtrain = np.load(self.data_path / 'X_train.npy')
        self.yTrain = np.load(self.data_path / 'y_train.npy')
        print('Training Data Shapes:\n')
        print(self.Xtrain.shape, self.yTrain.shape)
        
        # Validation data
        self.Xval = np.load(self.data_path / 'X_val.npy')
        self.yval = np.load(self.data_path / 'y_val.npy')
        print('Validation Data Shapes:\n')
        print(self.Xval.shape, self.yval.shape)
        
        # Testing data
        self.Xtest = np.load(self.data_path / 'X_test.npy')
        self.ytest = np.load(self.data_path / 'y_test.npy')
        print('Testing Data Shapes:\n')
        print(self.Xtest.shape, self.ytest.shape)
        
    def _prep_data(self):
        """Prepare the data by calculating targets and applying transformation."""
        self.yTrain -= self.Xtrain[:, self.natmos:]
        self.yval -= self.Xval[:, self.natmos:]
        self.ytest -= self.Xtest[:, self.natmos:]
        
        # Apply transformation or normalization
        print('\nApplying transformation: ',self.transformation_type)
        self.Xtrain_transformed, self.yTrain_transformed,\
        self.Xval_transformed, self.yval_transformed,\
        self.Xtest_transformed, self.ytest_transformed,\
        self.stats = apply_transformation(self.Xtrain,self.yTrain,self.Xval,self.yval,\
                                     self.Xtest,self.ytest,transformation_type=self.transformation_type)
            
        # Create DataLoader objects
        self.train_dataset = TensorDataset(torch.Tensor(self.Xtrain_transformed), torch.Tensor(self.yTrain_transformed))
        self.val_dataset = TensorDataset(torch.Tensor(self.Xval_transformed), torch.Tensor(self.yval_transformed))
        
        self.train_loader = DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=128, 
            pin_memory=True, 
            persistent_workers=True
        )
        
        self.val_loader = DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=128, 
            pin_memory=True, 
            persistent_workers=True
        )
        
    def _train_and_eval(self):
        for depth, width in zip(self.depths,self.widths):
            depth = depth + 1 # number of hidden layers == depth - 1
            print(f'\nTraining models with depth: {depth-1}, width: {width}')
            print(f'\nNumber of parameters: {(depth-1) * width**2}')
            
            all_preds = []
            all_r2s = []

            for model_run in range(self.n_models):
                print('\nModel Run: ',model_run+1)
                model = SimpleNN(self.Xtrain.shape[1], width, self.yTrain.shape[1], depth).to(self.device)

                optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                criterion = self.loss_func
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=5, verbose=False)

                # Train the model
                lossnpy = str(depth)+'_'+str(width)+'_'+str(model_run)+'_'+str(self.n_epochs)
                train_model(model, optimizer, scheduler, criterion, self.train_loader, self.val_loader, \
                            num_epochs=self.n_epochs, lossnpy=lossnpy)

                # Collect predictions for bias-variance analysis
                model = SimpleNN(self.Xtrain.shape[1], width, self.yTrain.shape[1], depth).to(self.device)
                model.load_state_dict(torch.load('best_model_'+lossnpy+'.pth'))
                preds = self._collect_predictions(model)
                all_preds.append(preds)
                
                r2s = self._calculate_r2(preds)
                all_r2s.append(r2s)
                
            # Analyze bias and variance for this configuration
            self.analyze_bias_variance(all_preds, all_r2s, depth, width)
    def _evalonly(self):
        model_id = self.model_id
        depth = int(model_id.split("_")[0])
        width = int(model_id.split("_")[1])
        # Collect predictions for bias-variance analysis
        model = SimpleNN(self.Xtrain.shape[1], width, self.yTrain.shape[1], depth).to(self.device)
        model.load_state_dict(torch.load('best_model_'+model_id+'.pth'))
        preds = self._collect_predictions(model)
        np.save('predictions.npy',np.array(preds))
        #np.save('Ytest_transformed.npy',self.ytest_transformed)
        with open("data_stats.p", 'wb') as fp:
            pickle.dump(self.stats, fp, protocol=pickle.HIGHEST_PROTOCOL)

                
    def _collect_predictions(self, model):
        """Collect predictions on the test set using the best model."""
        model.eval()
        with torch.no_grad():
            preds = model(torch.Tensor(self.Xtest_transformed).to(self.device))
            preds = preds.cpu().detach().numpy()
        predictions_np = inverse_standard_transform(self.stats, preds, 'ytrain_mean', 'ytrain_std')
        return predictions_np
    
    def _calculate_r2(self, predictions):
        """Calculate R2 score for each variable in the varlist."""
        r2s = []
        for i, vv in zip(range(len(self.varlist)), self.varlist):
            r2 = r2_score(predictions[:, i], self.ytest[:, i])
            print(f'{vv}: R2 = {r2}')
            r2s.append(r2)
        return r2s
    
    def analyze_bias_variance(self, all_preds, all_r2s, depth, width):
        """Compute and print the bias-variance trade-off."""
        all_preds = np.stack(all_preds, axis=0)
        mean_preds = np.mean(all_preds, axis=1)
        np.save('AllPreds.npy',all_preds)
        np.save('Ytest.npy',self.ytest)
        
        df = pd.DataFrame()
        df['target'] = self.varlist
        for i in range(len(all_r2s)):
            df['Pred_run_'+str(i)] = mean_preds[i]
            df['r2_run_'+str(i)] = all_r2s[i]
        
        # Compute bias and variance for this configuration
        mean_preds = np.mean(all_preds, axis=0)
        target_biases = np.mean((mean_preds - self.ytest) ** 2, axis=0)
        target_variance = np.mean(np.var(all_preds, axis=0),axis=0)

        # Store the results for further analysis
        mean_ytest = np.mean(self.ytest, axis=0)
        df['ytest'] = mean_ytest
        df['target_biases'] = target_biases
        df['target_variance'] = target_variance
        df.to_csv('bias_var_tradeoff'+str(depth)+'_'+str(width)+'_'+str(self.n_epochs)+'.csv',index=False)

def train_model(model, optimizer, scheduler, criterion, train_loader, val_loader, num_epochs=5, early_stop=False, patience=5,lossnpy=''):
    """Train and validate the model with optional early stopping."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    TrainLosses = []
    ValLosses = []
    best_val_loss = float('inf')
    patience_counter = 0
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training loop
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        TrainLosses.append(epoch_loss)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        ValLosses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}')

        # Step the learning rate scheduler if using ReduceLROnPlateau
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), 'best_model_'+lossnpy+'.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            print('patience_counter:',patience_counter)
            
        best_val_loss = np.minimum(best_val_loss, val_loss)

        if early_stop and patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training finished.")
    np.save('TrainLosses_'+lossnpy+'.npy',np.array(TrainLosses))
    np.save('ValLosses_'+lossnpy+'.npy',np.array(ValLosses))

