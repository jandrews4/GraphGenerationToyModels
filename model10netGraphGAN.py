import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, knn_graph
from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import KNNGraph
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import numpy as np

class ModelNetDataset:
    def __init__(self, root, classes, transform=None):
        self.root = root.replace("\\", "/")
        self.classes = classes
        self.transform = transform

    def load_data(self):
        data_list = []
        for cls in self.classes:
            cls_path = os.path.join(self.root, cls).replace("\\", "/")
            for split in ['train', 'test']:
                split_path = os.path.join(cls_path, split).replace("\\", "/")
                print(f"Checking path: {split_path}")  # Debugging print
                if os.path.exists(split_path):
                    for file_name in os.listdir(split_path):
                        if file_name.endswith('.off'):
                            file_path = os.path.join(split_path, file_name).replace("\\", "/")
                            print(f"Processing file: {file_path}")  # Debugging print
                            vertices, faces = self.load_off(file_path)
                            data = self.process_data(vertices, cls)
                            if self.transform:
                                data.pos = data.x  # Ensure data has 'pos' attribute
                                data = self.transform(data)
                            data_list.append(data)
                else:
                    print(f"Path does not exist: {split_path}")  # Debugging print
        return data_list

    def load_off(self, file_path):
        with open(file_path, 'r') as f:
            if 'OFF' != f.readline().strip():
                raise('Not a valid OFF header')
            n_verts, n_faces, _ = map(int, f.readline().strip().split(' '))
            vertices = [list(map(float, f.readline().strip().split(' '))) for _ in range(n_verts)]
            faces = [list(map(int, f.readline().strip().split(' ')[1:])) for _ in range(n_faces)]
        return vertices, faces

    def process_data(self, vertices, cls):
        x = torch.tensor(vertices, dtype=torch.float)
        y = torch.tensor([self.classes.index(cls)], dtype=torch.long)
        edge_index = knn_graph(x, k=6)
        return Data(x=x, edge_index=edge_index, y=y)

# Specify dataset root and classes
root = 'C:/Users/Owner/Downloads/ModelNet10/ModelNet10'
classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

# Initialize and load dataset
dataset = ModelNetDataset(root, classes, transform=KNNGraph(k=6))
data_list = dataset.load_data()

# Debugging print
print(f"Total data samples loaded: {len(data_list)}")

# Create data loader
data_loader = DataLoader(data_list, batch_size=1, shuffle=True)

# Print to confirm the data loader is working as expected
for data in data_loader:
    print(data)
    break

# Define the Generator and Discriminator
class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(noise_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        z = F.relu(self.fc1(z))
        x = torch.sigmoid(self.fc2(z))
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = torch.sigmoid(self.conv2(x, edge_index))
        return x

# Define a pre-trained Graph Neural Network for FID computation
class PretrainedGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PretrainedGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Function to calculate FID
def calculate_fid(real_embeddings, fake_embeddings):
    mu_real = np.mean(real_embeddings, axis=0)
    sigma_real = np.cov(real_embeddings, rowvar=False)
    mu_fake = np.mean(fake_embeddings, axis=0)
    sigma_fake = np.cov(fake_embeddings, rowvar=False)
    
    ssdiff = np.sum((mu_real - mu_fake)**2.0)
    covmean = sqrtm(sigma_real.dot(sigma_fake))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return fid

# Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_dim = 16
hidden_dim = 64
output_dim = 3  # Dimension of node features
num_epochs = 100

generator = Generator(noise_dim, hidden_dim, output_dim).to(device)
discriminator = Discriminator(output_dim, hidden_dim).to(device)
pretrained_gnn = PretrainedGNN(output_dim, hidden_dim, 128).to(device)  # Output embedding dimension is 128

# Use DataParallel to leverage multiple GPUs if available
if torch.cuda.device_count() > 1:
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)
    pretrained_gnn = nn.DataParallel(pretrained_gnn)

optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)
criterion = nn.BCELoss()

scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

def train_gan(generator, discriminator, pretrained_gnn, data_loader, optimizer_g, optimizer_d, criterion, num_epochs):
    generator.train()
    discriminator.train()
    pretrained_gnn.eval()
    
    generator_losses = []
    discriminator_losses = []
    node_accuracies = []
    fid_scores = []
    
    for epoch in range(num_epochs):
        epoch_loss_g = 0
        epoch_loss_d = 0
        correct_nodes = 0
        total_nodes = 0
        
        real_embeddings = []
        fake_embeddings = []
        
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for data in data_loader:
                data = data.to(device)
                real_labels = torch.ones(data.x.size(0), 1).to(device)
                fake_labels = torch.zeros(data.x.size(0), 1).to(device)
                
                # Train Discriminator
                optimizer_d.zero_grad()
                
                with torch.cuda.amp.autocast():
                    # Real data
                    real_outputs = discriminator(data.x, data.edge_index)
                    loss_d_real = criterion(real_outputs, real_labels)
                    
                    # Extract real embeddings for FID
                    with torch.no_grad():
                        real_embedding = pretrained_gnn(data.x, data.edge_index)
                        real_embeddings.append(real_embedding.cpu().numpy())
                    
                    # Fake data
                    z = torch.randn(data.x.size(0), noise_dim).to(device)
                    fake_data = generator(z)
                    edge_index = knn_graph(fake_data, k=6)
                    fake_outputs = discriminator(fake_data, edge_index)
                    loss_d_fake = criterion(fake_outputs, fake_labels)
                    
                    # Extract fake embeddings for FID
                    with torch.no_grad():
                        fake_embedding = pretrained_gnn(fake_data, edge_index)
                        fake_embeddings.append(fake_embedding.cpu().numpy())
                    
                    # Total discriminator loss
                    loss_d = loss_d_real + loss_d_fake
                
                scaler.scale(loss_d).backward()
                scaler.step(optimizer_d)
                scaler.update()
                
                # Train Generator
                optimizer_g.zero_grad()
                
                with torch.cuda.amp.autocast():
                    z = torch.randn(data.x.size(0), noise_dim).to(device)
                    fake_data = generator(z)
                    edge_index = knn_graph(fake_data, k=6)
                    fake_outputs = discriminator(fake_data, edge_index)
                    
                    loss_g = criterion(fake_outputs, real_labels)
                
                scaler.scale(loss_g).backward()
                scaler.step(optimizer_g)
                scaler.update()
                
                # Metrics
                epoch_loss_g += loss_g.item()
                epoch_loss_d += loss_d.item()
                
                # Node-level accuracy
                predicted_labels = (real_outputs > 0.5).float()
                correct_nodes += (predicted_labels == real_labels).sum().item()
                total_nodes += real_labels.size(0)
                
                pbar.update(1)
        
        epoch_loss_g /= len(data_loader)
        epoch_loss_d /= len(data_loader)
        node_accuracy = 100. * correct_nodes / total_nodes
        
        generator_losses.append(epoch_loss_g)
        discriminator_losses.append(epoch_loss_d)
        node_accuracies.append(node_accuracy)
        
        # Calculate FID
        real_embeddings = np.concatenate(real_embeddings, axis=0)
        fake_embeddings = np.concatenate(fake_embeddings, axis=0)
        fid_score = calculate_fid(real_embeddings, fake_embeddings)
        fid_scores.append(fid_score)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {epoch_loss_g:.4f}, Discriminator Loss: {epoch_loss_d:.4f}, Node Accuracy: {node_accuracy:.2f}%, FID: {fid_score:.4f}")
    
    return generator_losses, discriminator_losses, node_accuracies, fid_scores

# Run the training loop
generator_losses, discriminator_losses, node_accuracies, fid_scores = train_gan(generator, discriminator, pretrained_gnn, data_loader, optimizer_g, optimizer_d, criterion, num_epochs)

# Plot the results
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(generator_losses, label='Generator Loss')
plt.plot(discriminator_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator and Discriminator Loss per Epoch')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(node_accuracies, label='Node Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Node Level Accuracy per Epoch')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(fid_scores, label='FID Score')
plt.xlabel('Epoch')
plt.ylabel('FID')
plt.title('Frechet Inception Distance (FID) per Epoch')
plt.legend()

plt.tight_layout()
plt.show()
