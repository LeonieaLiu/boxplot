import torch
from tqdm import tqdm
import torch.nn as nn

def dice_coeff(input, target, epsilon=1e-6):
    input = (input > 0.5).float()
    intersection = (input * target).sum(dim=(1, 2, 3))
    union = input.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()

@torch.no_grad()
def test(model, test_loader, device, epoch, epochs):
    model.eval()
    epoch_dice_coeff = 0
    total_batches = len(test_loader)

    for images, masks, _ in tqdm(test_loader, desc=f"Testing Epoch {epoch + 1}/{epochs}"):
        images = images.to(device)
        masks = masks.to(device)

        outputs, _, _, _, _, _ = model(images)

        # Accumulate Dice coefficient 累加 Dice 系数
        outputs = torch.sigmoid(outputs)
        epoch_dice_coeff += dice_coeff(outputs, masks)

    # Calculate average Dice coefficient计算平均 Dice 系数
    avg_dice_coeff = epoch_dice_coeff / total_batches

    return avg_dice_coeff
