from util import *
import sklearn.metrics
from sklearn.metrics import roc_auc_score


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:0')

df = pd.read_csv('/home/tanxin/Kaggle/df_instancenumber')
df_box = pd.read_csv('active_extravasation_bounding_boxes.csv')

df = df_box.merge(df, on=['series_id','instance_number'], how='left')

def modify(num, bias=0):
    if np.isnan(num):
        return num
    else:
        return str(int(num) + bias).zfill(4)

df = df[df['x2'] <= 512]
df = df[df['y2'] <= 512]
df = df[df['png_number']!=0]  # should drop
df['image_path'] = '/disk1/tanxin/train_data_npy/' + df['pid'].astype(str) + '_' + df['series_id'].astype(str) + '_' + df['png_number'].apply(modify) + '.npy'
df = df[df['image_path']!= '/disk1/tanxin/train_data_npy/31474_20619_0106.npy']
df.rename(columns={'pid':'patient_id'}, inplace=True)
df = df.groupby('patient_id').agg(lambda x: x.tolist())
df2 = pd.read_csv('train.csv')
df = df.merge(df2, on=['patient_id'], how='left')
df = df.drop(columns=['filename', 'bowel_healthy', 'bowel_injury', 'extravasation_healthy', 'kidney_healthy', 'kidney_low', 'kidney_high',
       'liver_healthy', 'liver_low', 'liver_high', 'spleen_healthy', 'spleen_low', 'spleen_high', 'any_injury', 'Unnamed: 0',
       'ImagePositionPatient', 'path', 'reverse', 'min', 'instance_number'])
df['has_bbox'] = 1
df['flag'] = 1
# ['patient_id', 'x1', 'y1', 'x2', 'y2', 'series_id', 'width', 'height',
#        'image_path', 'extravasation_injury', 'has_bbox']




# df = pd.concat([df, df_noise], ignore_index=True)


df2 = df2[df2['extravasation_healthy']==1]
df3 = pd.read_csv('train_series_meta.csv')
df2 = df2.merge(df3, on=['patient_id'], how='left').reset_index(drop=True)
df2 = df2.drop(columns=['bowel_healthy', 'bowel_injury', 'extravasation_healthy', 'kidney_healthy', 'kidney_low', 'kidney_high',
       'liver_healthy', 'liver_low', 'liver_high', 'spleen_healthy', 'spleen_low', 'spleen_high', 'any_injury', 'aortic_hu',
       'incomplete_organ'])
df2['image_path'] = '/disk1/tanxin/train_data_npy/' + df2['patient_id'].astype(str) + '_' + df2['series_id'].astype(str) + '_*.npy'
df2['split'] = (df2.index // 10).values
df2 = df2.groupby('split').agg(lambda x: x.tolist())
df2['x1'] = df2['y1'] = 0
df2['x2'] = df2['y2'] = 512
df2['extravasation_injury'] = df2['has_bbox'] = df2['flag'] = 0
df2 = df2.reset_index(drop=True)

df = pd.concat([df, df2], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
pd.set_option('display.max_columns',None)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df['fold'] = -1
for fold, (train_idx, valid_idx) in enumerate(kf.split(df, df['extravasation_injury'])):
    df.loc[valid_idx, 'fold'] = fold

df_noise = df[df['extravasation_injury']==1].copy()
df_noise['flag'] = df_noise['has_bbox'] = 0
df_noise['x1'] = df_noise['y1'] = 0
df_noise['x2'] = df_noise['y2'] = 512
df = pd.concat([df, df_noise], ignore_index=True)

def score(submission, solution):
    y_true = solution / np.sum(solution, axis=1, keepdims=True)
    y_pred = np.concatenate([1-submission, submission], axis=1)
    y_pred = y_pred / np.sum(y_pred, axis=1, keepdims=True)
    score = sklearn.metrics.log_loss(
        y_true=y_true,
        y_pred=y_pred,
        sample_weight=np.sum(y_true*np.array([[1., 6.]]), axis=1)
    )
    return score

def train_func(model, loader_train, optimizer, scaler=None):
    model.train()
    true = []
    prob = []
    train_loss = []
    bbox_loss = []
    cls_loss = []
    bar = tqdm(loader_train)
    for batch in bar:
        batch['image'] = batch['image'].cuda()
        batch['labels'] = batch['labels'].cuda()
        optimizer.zero_grad()
        with amp.autocast():
            output_dict = model(batch)
            loss = output_dict['loss']
            loss_bbox = output_dict['loss_bbox']
            loss_cls = output_dict['loss_cls']

        prob.append(output_dict['preds'][:, -1])
        true.append(batch['labels'][:, -1])
        train_loss.append(loss.item())
        bbox_loss.append(loss_bbox.item())
        cls_loss.append(loss_cls.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bar.set_description(f'loss:{np.mean(train_loss[-30:]):.4f} loss_bb:{np.mean(bbox_loss[-30:]):.4f} loss_cls:{np.mean(cls_loss[-30:]):.4f}')
    true = torch.concat(true)
    # true = torch.stack([1-true, true], dim=1)
    sc = roc_auc_score(true.cpu().detach().numpy(), torch.concat(prob).cpu().detach().numpy())
    # sc = score(torch.concat(prob).cpu().detach().numpy()[:, np.newaxis], true.cpu().detach().numpy())
    return np.mean(train_loss), np.mean(bbox_loss), np.mean(cls_loss), sc


def valid_func(model, loader_valid):
    model.eval()
    true = []
    prob = []
    valid_loss = []
    bbox_loss = []
    cls_loss = []
    bar = tqdm(loader_valid)
    with torch.no_grad():
        for batch in bar:
            batch['image'] = batch['image'].cuda()
            batch['labels'] = batch['labels'].cuda()
            output_dict = model(batch)

            prob.append(output_dict['preds'][:, -1])
            true.append(batch['labels'][:, -1])

            loss = output_dict['loss']
            loss_bbox = output_dict['loss_bbox']
            loss_cls = output_dict['loss_cls']
            valid_loss.append(loss.item())
            bbox_loss.append(loss_bbox.item())
            cls_loss.append(loss_cls.item())
            bar.set_description(f'loss:{np.mean(valid_loss[-30:]):.4f} loss_bb:{np.mean(bbox_loss[-30:]):.4f} loss_cls:{np.mean(cls_loss[-30:]):.4f}')
    true = torch.concat(true)
    # true = torch.stack([1-true, true], dim=1)
    sc = roc_auc_score(true.cpu().detach().numpy(), torch.concat(prob).cpu().detach().numpy())
    # sc = score(torch.concat(prob).cpu().detach().numpy()[:, np.newaxis], true.cpu().detach().numpy())
    return np.mean(valid_loss), np.mean(bbox_loss), np.mean(cls_loss), sc


def run(fold):
    backbone = ''
    log_file = os.path.join(log_dir, f'seg_ex1_{backbone}.txt')
    model_file = os.path.join(model_dir, f'seg_ex1_{backbone}_fold{fold}_best.pth')

    train_ = df[df['fold'] != fold].reset_index(drop=True)

    valid_ = df[df['fold'] == fold].reset_index(drop=True)



    dataset_train = SEGDatasetEx(train_, 'train', transform=transforms_train_ex)
    dataset_valid = SEGDatasetEx(valid_, 'valid', transform=transforms_valid_ex)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=16)
    loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=16, shuffle=False, num_workers=16)

    model = SegNetEx(pretrained=True,backbone=backbone)
    model = model.to(device)
    loss_min = 0.5
    load_ckpt = False
    if load_ckpt:
        model.load_state_dict(torch.load(f'/home/tanxin/Kaggle/models/seg_ex1_{backbone}_fold{fold}_last.pth')['model_state_dict'])
        loss_min = torch.load(f'/home/tanxin/Kaggle/models/seg_ex1_{backbone}_fold{fold}_last.pth')['score_best']
    optimizer = optim.AdamW(model.parameters(), lr=0.0002)
    scaler = torch.cuda.amp.GradScaler()
    from_epoch = 0
    metric_best = 0.

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 200)

    print(len(dataset_train), len(dataset_valid))

    for epoch in range(1, 200+1):
        scheduler_cosine.step(epoch - 1)
        print(time.ctime(), 'Epoch:', epoch)

        train_loss, train_loss_bbox, train_loss_cls, score_train = train_func(model, loader_train, optimizer, scaler)
        valid_loss, valid_loss_bbox, valid_loss_cls, score_valid = valid_func(model, loader_valid)

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, ' \
                                       f'loss: {train_loss:.5f}|{valid_loss:.5f}, ' \
                                       f'bbox: {train_loss_bbox:.5f}|{valid_loss_bbox:.5f}, ' \
                                       f'cls: {train_loss_cls:.5f}|{valid_loss_cls:.5f}, ' \
                                       f'score: {score_train:.5f}|{score_valid:.5f}.'
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

        if score_valid > loss_min:
            print(f'metric_best ({loss_min:.6f} --> {score_valid:.6f}). Saving model ...')
            torch.save(model.state_dict(), model_file)
            loss_min = score_valid

        # Save Last
        if not DEBUG:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'score_best': loss_min,
                },
                model_file.replace('_best', '_last')
            )

    del model
    torch.cuda.empty_cache()
    gc.collect()

# run(0)
# run(1)
# run(2)
# run(3)
run(4)