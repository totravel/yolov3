import argparse
import ast
import os
import torch
import torch.optim as optim
from utils import get_device
from yolo_model import YOLOv3
from yolo_loss import YOLOv3Loss
from yolo_data import voc_loader


def parse_args():
  parser = argparse.ArgumentParser(description='YOLOv3 Training')

  parser.add_argument('--voc_root', type=str, default='data/VOCdevkit', help='VOC root directory')
  parser.add_argument('--img_size', type=int, default=416, help='input image size')

  parser.add_argument('--weights', type=str, default='data/darknet53_256_c2ns-3aeff817.pth', help='path to pretrained ImageNet-1k weights')

  parser.add_argument('--save_dir', type=str, default='out/checkpoints', help='directory to save model checkpoints')
  parser.add_argument('--ckpt', type=ast.literal_eval, default=True, help='whether to load from latest checkpoint if available')

  parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
  parser.add_argument('--batch_size', type=int, default=32, help='batch size')

  parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
  parser.add_argument('--mo', type=float, default=0.9, help='momentum')
  parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')

  parser.add_argument('--patience', type=int, default=5, help='patience')
  parser.add_argument('--log_interval', type=int, default=10, help='log print interval')

  parser.add_argument('--force', action='store_true', help='force override lr')

  return parser.parse_args()


def train_one_epoch(model, train_loader, criterion, optimizer, device, log_interval: int=5):
  total_loss = .0
  bbox_loss = .0
  obj_loss = .0
  cls_loss = .0
  noobj_loss = .0

  model.train()
  num_batches = len(train_loader)

  for batch, (imgs, labels) in enumerate(train_loader, 1):
    print(f'batch {batch}/{num_batches}...', end='\r', flush=True)

    imgs = imgs.to(device)
    labels = [(bboxes.to(device), classes.to(device)) for (bboxes, classes) in labels]

    preds = model(imgs)
    loss, *loss_components = criterion(preds, labels)
    
    optimizer.zero_grad()
    loss.backward()

    has_invalid_grad = False
    for param in model.parameters():
      if param.grad is not None:
        if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
          has_invalid_grad = True
          break
    if has_invalid_grad:
      print(f'⚠️ invalid gradients (NaN/Inf) detected in batch {batch}, skipping update')
      optimizer.zero_grad()
      continue

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
    optimizer.step()

    total_loss += loss.item()
    bbox_loss += loss_components[0]
    obj_loss += loss_components[1]
    cls_loss += loss_components[2]
    noobj_loss += loss_components[3]

    if batch % log_interval == 0:
      avg_loss = total_loss / batch
      avg_bbox = bbox_loss / batch
      avg_obj = obj_loss / batch
      avg_cls = cls_loss / batch
      avg_noobj = noobj_loss / batch
      print(f'batch {batch}/{num_batches}, loss {avg_loss:.4f}', 
            f'(bbox {avg_bbox:.4f}, obj {avg_obj:.4f}, cls {avg_cls:.4f}, noobj {avg_noobj:.4f})')
  
  return total_loss / num_batches


@torch.no_grad()
def validate(model, val_loader, criterion, device, log_interval: int=5):
  total_loss = .0

  model.eval()
  num_batches = len(val_loader)

  for batch, (imgs, labels) in enumerate(val_loader, 1):
    print(f'batch {batch}/{num_batches}...', end='\r', flush=True)
    
    imgs = imgs.to(device)
    labels = [(bboxes.to(device), classes.to(device)) for (bboxes, classes) in labels]

    preds = model(imgs)
    loss, *_ = criterion(preds, labels)
    
    total_loss += loss.item()

    if batch % log_interval == 0:
      avg_loss = total_loss / batch
      print(f'batch {batch}/{num_batches}, loss {avg_loss:.4f} ')
  
  return total_loss / num_batches


def main():
  args = parse_args()
  os.makedirs(args.save_dir, exist_ok=True)

  device = get_device()
  print(f'🧰 device: {device}')

  model = YOLOv3(pretrained=True, weights=args.weights).to(device)
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mo, weight_decay=args.wd)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
  criterion = YOLOv3Loss(device)
  
  start_epoch = 0
  best_val_loss = float('inf')
  pth_path = os.path.join(args.save_dir, 'yolo_best.pth')
  ckpt_path = os.path.join(args.save_dir, 'yolo_last.ckpt')

  if args.ckpt and os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    last_epoch = ckpt['epoch']
    start_epoch = last_epoch + 1
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    best_val_loss = ckpt['val_loss']
    print(f'💾 checkpoint loaded (last_epoch {last_epoch+1}, best_val_loss {best_val_loss:.4f})')
  
    if args.force:
      for pg in optimizer.param_groups:
        pg['lr'] = args.lr

  train_loader, val_loader = voc_loader(args.voc_root, args.img_size, args.batch_size)
  patience_counter = 0

  for epoch in range(start_epoch, args.epochs):
    curr_lr = ','.join([f'{lr:.6f}' for lr in scheduler.get_last_lr()])
    print(f'================ epoch {epoch+1}/{args.epochs} [{curr_lr}] ================ ')

    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.log_interval)
    val_loss = validate(model, val_loader, criterion, device, args.log_interval)
    scheduler.step()

    print(f'train_loss {train_loss:.4f}, val_loss {val_loss:.4f}')

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      patience_counter = 0
      torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'val_loss': best_val_loss
      }, pth_path)
      print(f'✅ best model saved (best_val_loss {best_val_loss:.4f})')
    else:
      patience_counter += 1
      if patience_counter >= args.patience:
        print(f'❌ no better model for {args.patience} epochs, early stopping')
        break
      print(f'⭕️ no better model (best_val_loss {best_val_loss:.4f}, patience {patience_counter}/{args.patience})')

    torch.save({
      'epoch': epoch,
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'scheduler': scheduler.state_dict(),
      'val_loss': best_val_loss
    }, ckpt_path)
    print(f'💾 latest model saved')
  print(f'🎉 all done')
    

if __name__ == '__main__':
  main()
