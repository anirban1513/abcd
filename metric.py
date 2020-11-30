import torch 

def dice_value(pred, target, threshold = 0.5, smooth = 1e-6):
    if threshold is not None:
        pred = (torch.sigmoid(pred) > threshold).float()
    else:
        pred = torch.sigmoid(pred)
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum(dim=1)
    dice_score = (2. * intersection + smooth) / (m1.sum(dim=1) + m2.sum(dim=1) + smooth) 
    return dice_score.mean()

def dice_coeff(pred, target, threshold = None, smooth = 1e-6):
    pred = torch.sigmoid(pred)
    if threshold:
        pred = (pred>threshold).float()
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

