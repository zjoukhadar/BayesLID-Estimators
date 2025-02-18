import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from estimators import compute_lid_estimators
from pooling import arithmetic_pooling, harmonic_pooling, geometric_pooling
from sequential_update import one_step_update, accumulative_update
from knn import get_knn_distances

def main():
    # Hyperparameters
    batch_size = 128
    k = 10  # number of nearest neighbors (for LID estimation)
    num_epochs = 5  # number of epochs to simulate sequential updates
    tau = 0.9    # temperature parameter for sequential updates

    # Data transformation and CIFAR-10 dataset loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    # Load a pre-trained ResNet18, and remove its final FC layer to obtain feature vectors.
    model = torchvision.models.resnet18(pretrained=True)
    # Remove final FC layer (keep up to the average pooling layer for 512-D features)
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    # Initialize variables for sequential updates.
    one_step_prev_beta = None
    accum_alpha = 0.0
    accum_beta = 0.0

    # Lists to store estimates over epochs.
    epoch_mle_estimates = []
    epoch_stein_estimates = []
    epoch_brown_estimates = []
    epoch_one_step_estimates = []
    epoch_accum_estimates = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        # For simplicity, use one mini-batch per epoch to compute LID.
        data_iter = iter(trainloader)
        images, _ = next(data_iter)
        images = images.to(device)
        
        # Extract features using the pre-trained ResNet18.
        with torch.no_grad():
            feats = feature_extractor(images)
            # Flatten features from shape (B, 512, 1, 1) to (B, 512)
            feats = feats.view(feats.size(0), -1)
        
        # Compute pairwise kNN distances (excluding the self–distance).
        knn_distances_list = get_knn_distances(feats, k)
        
        # Compute LID estimators for each sample.
        mle_list = []
        stein_list = []
        brown_list = []
        beta_list = []  # to store beta (–Σln(x_i/w)) for sequential updates
        
        for knn in knn_distances_list:
            # This function implements the estimators as in Equation (3) and Table 1 of the paper.
            estimators = compute_lid_estimators(knn)
            mle_list.append(estimators["mle"])
            stein_list.append(estimators["stein"])
            brown_list.append(estimators["brown"])
            beta_list.append(estimators["beta"])
        
        # Pooling: using arithmetic pooling (linear pooling) for MLE,
        # harmonic pooling (logarithmic pooling) for Stein, and geometric pooling for Brown.
        pooled_mle_arith = arithmetic_pooling(mle_list)
        pooled_mle_log = harmonic_pooling(mle_list)
        pooled_stein = harmonic_pooling(stein_list)
        pooled_brown = geometric_pooling(brown_list)
        
        print(f"  Pooled MLE (arithmetic): {pooled_mle_arith:.3f}")
        print(f"  Pooled MLE (harmonic):   {pooled_mle_log:.3f}")
        print(f"  Pooled Stein estimator:  {pooled_stein:.3f}")
        print(f"  Pooled Brown estimator:  {pooled_brown:.3f}")
        
        # Average beta for the batch (beta = -sum_i ln(x_i/w)).
        current_beta = np.mean(beta_list)
        print(f"  Average beta from batch: {current_beta:.3f}")
        
        # Store pooled estimates for summary.
        epoch_mle_estimates.append(pooled_mle_arith)
        epoch_stein_estimates.append(pooled_stein)
        epoch_brown_estimates.append(pooled_brown)
        
        # ---------------------------
        # Sequential Update Methods
        # ---------------------------
        if epoch == 0:
            # Initialize one-step sequential update with the first beta value.
            one_step_prev_beta = current_beta
            one_step_estimate = k / one_step_prev_beta  # MLE from first batch
            # For accumulative update, initialize accumulators.
            accum_alpha = k
            accum_beta = current_beta
            accum_estimate = accum_alpha / accum_beta
        else:
            # One-Step Update: see Equation (15) in the paper.
            one_step_estimate, one_step_prev_beta = one_step_update(one_step_prev_beta, current_beta, k, tau)
            # Accumulative Update: see Equation (16) in the paper.
            accum_estimate, accum_alpha, accum_beta = accumulative_update(accum_alpha, accum_beta, current_beta, k, tau, epoch)
        
        epoch_one_step_estimates.append(one_step_estimate)
        epoch_accum_estimates.append(accum_estimate)
        
        print(f"  One-Step Sequential Estimate: {one_step_estimate:.3f}")
        print(f"  Accumulative Sequential Estimate: {accum_estimate:.3f}")
    
    # Summary of estimates over epochs.
    print("\nSummary of pooled estimates over epochs:")
    for i in range(num_epochs):
        print(f"Epoch {i+1}: MLE={epoch_mle_estimates[i]:.3f}, "
              f"Stein={epoch_stein_estimates[i]:.3f}, Brown={epoch_brown_estimates[i]:.3f}, "
              f"One-Step={epoch_one_step_estimates[i]:.3f}, Accumulative={epoch_accum_estimates[i]:.3f}")

if __name__ == '__main__':
    main()