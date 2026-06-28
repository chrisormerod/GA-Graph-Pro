import torch
from torch import nn
import pandas as pd
from torch.utils.data import (TensorDataset,
                              DataLoader,
                              random_split)
import copy

class DataFrameEmbeddingModel(nn.Module):
    
    def __init__(self, 
                 df : pd.DataFrame,
                 dimension : int, 
                 hidden : int,
                 smoothing : float = 0.1,
                 drop : float = 0.4 ):
        super().__init__()
        self.index = df.index
        self.dimension = dimension
        self.data = df
        
        self.embedding = nn.Embedding(num_embeddings = len(self.index), 
                                      embedding_dim = self.dimension)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features = dimension, 
                      out_features = hidden),
            nn.ReLU(),
            nn.Dropout(p = drop),
            nn.Linear(in_features = hidden, 
                      out_features = len(self.data.columns)),
            nn.Sigmoid()
        )
        
    
    def forward(self, 
                X : torch.LongTensor,
                rows : torch.Tensor):
        embedding = self.embedding(X)
        pred = self.classifier(embedding)
        return pred
  
    def fit(self,
            lr: float,
            batch_size: int = 100,
            epochs: int = 10,
            validation_split: float = 0.1,
            early_stopping_patience: int = 3,
            verbose: bool = True):
        """
        Train the DataFrameEmbeddingModel on the DataFrame data.
        
        Parameters:
        -----------
        lr : float
            Learning rate for the optimizer
        batch_size : int, default=100
            Number of samples per batch
        epochs : int, default=10
            Number of training epochs
        validation_split : float, default=0.1
            Fraction of data to use for validation
        early_stopping_patience : int, default=3
            Number of epochs with no improvement after which training will be stopped
        verbose : bool, default=True
            Whether to print progress during training
            
        Returns:
        --------
        dict
            Training history containing loss values
        """
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer)
        
        # Convert DataFrame to tensors
        indices = torch.arange(len(self.index), dtype=torch.long)
        targets = torch.tensor(self.data.values, dtype=torch.float32)
        
        # Create dataset and split into training and validation
        dataset = TensorDataset(indices, targets)
        
        # Split dataset into training and validation sets
        val_size = int(validation_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_losses = []
            
            for batch_idx, (batch_indices, batch_targets) in enumerate(train_loader):
                loss = self.training_step((batch_indices, batch_targets), batch_idx)
                train_losses.append(loss.item())
                
            avg_train_loss = sum(train_losses) / len(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_idx, (batch_indices, batch_targets) in enumerate(val_loader):
                    # Forward pass
                    embeddings = self.embedding(batch_indices)
                    predictions = self.classifier(embeddings)
                    
                    # Calculate validation loss
                    criterion = nn.BCELoss()
                    val_loss = criterion(predictions, batch_targets)
                    val_losses.append(val_loss.item())
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            history['val_loss'].append(avg_val_loss)
            
            # Step the learning rate scheduler
            self.scheduler.step()
            
            # Print progress
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load the best model if early stopping occurred
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
        
        return history        
            
            
    def training_step(self, batch, batch_idx):
        """
        Execute a single training step for the DataFrameEmbeddingModel.
        
        Parameters:
        -----------
        batch : tuple
            A tuple containing (indices, target_values) where:
            - indices: LongTensor of sample indices
            - target_values: Tensor of target values from the DataFrame
        batch_idx : int
            The index of the current batch (not used in this implementation)
            
        Returns:
        --------
        torch.Tensor
            The computed loss value for this batch
        """
        # Unpack the batch
        indices, targets = batch
        
        # Get the corresponding row indices for the loss calculation
        row_indices = torch.arange(len(indices))
        
        # Forward pass
        embeddings = self.embedding(indices)
        predictions = self.classifier(embeddings)
        
        # Calculate loss with label smoothing
        # Binary cross entropy loss is appropriate for multi-label classification with sigmoid outputs
        criterion = nn.BCELoss()
        
        # Apply label smoothing if specified
        smoothed_targets = targets.clone()
        if hasattr(self, 'smoothing') and self.smoothing > 0:
            smoothed_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        loss = criterion(predictions, smoothed_targets)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss

    def to_dataframe(self) -> pd.DataFrame:
        with torch.no_grad():
            tensor = self.embedding(torch.arange(0,len(self.index)))
        arr = tensor.cpu().numpy()
        df = pd.DataFrame(arr, index= self.index)
        return df
        
        
            