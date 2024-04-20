import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from util import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')
df = pd.read_csv('train_series_meta.csv')
mask_files = os.listdir(f'segmentations')
df_mask = pd.DataFrame({
    'mask_file': mask_files,
})
df_mask['series_id'] = df_mask['mask_file'].apply(lambda x: int(x.split('.')[0]))
df_mask['mask_file'] = df_mask['mask_file'].apply(lambda x: os.path.join('segmentations', x))
# print(df_mask.dtypes)
df = df.merge(df_mask, on='series_id', how='left')
df['image_path'] = 'train_data/'+ df['patient_id'].astype(str)+'_'+df['series_id'].astype(str)
df['mask_file'].fillna('', inplace=True)

df_to_seg = df.query('mask_file == ""').reset_index(drop=True)
fold = 1
log_file = os.path.join(log_dir, f'{kernel_type}.txt')
model_file = os.path.join(model_dir, f'{kernel_type}_fold{fold}_best.pth')
test_ = df_to_seg.reset_index(drop=True)
dataset_test = SEGDataset(test_, 'test', transform=transforms_valid)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = TimmSegModel(backbone, pretrained=True)
model = convert_3d(model)
model = model.to(device)
model.load_state_dict(torch.load(model_file))
model.eval()

def seg_func(model, loader_valid):
    model.eval()


    bar = tqdm(loader_valid)
    with torch.no_grad():
        for images, series_id in bar:
            images = images.cuda()
            logits = model(images)

            pred = (logits.sigmoid() > 0.5).float().detach()
            for i in range(len(series_id)):
                np.save(f"our_seg/{series_id[i]}.npy",pred[i].cpu())
    return 0

# Image.open('train_data/12930_34757_0135.png')
seg_func(model, loader_test)

