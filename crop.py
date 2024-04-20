from util import *


def load_png_512(path):
    img = Image.open(path)
    data = np.array(img)
    data = cv2.resize(data, (512, 512), interpolation = cv2.INTER_LINEAR)
    return data

def load_png_line_par_512(path):
    t_paths = sorted(glob(path+"_*.png"),
                     key=lambda x: int(x.split('/')[-1].split(".")[0]))

    n_scans = len(t_paths)
    indices = np.quantile(list(range(n_scans)), np.linspace(0., 1., image_sizes[2])).round().astype(int)
    t_paths = [t_paths[i] for i in indices]

    images = []
    for filename in t_paths:
        images.append(load_png_512(filename))
    images = np.stack(images, -1)

    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)
    images = (images * 255).astype(np.uint8)

    return images

def search_crop_box(mask):
    z_dim = mask.shape[0]
    # rows = np.any(mask, axis=2)
    # cols = np.any(mask, axis=1)
    left = lower = 128
    right = upper = 0
    for z in range(z_dim):
        rows = np.any(mask[:, :, z], axis=1)
        cols = np.any(mask[:, :, z], axis=0)
        if not len(np.where(rows)[0]):
            continue
        else:
            left_, right_  = np.where(rows)[0][[0, -1]]
            lower_, upper_ = np.where(cols)[0][[0, -1]]

            if left_ < left: left = left_
            if right_ > right: right = right_
            if lower_ < lower: lower = lower_
            if upper_ > upper: upper = upper_

    return left, right, lower, upper


def get_crop(image, mask_512, mask):
    left, right, lower, upper = search_crop_box(mask)
    h = 512. / 128
    left, right, lower, upper = round(left * h), round(right * h), round(lower * h), round(upper * h)
    print(left, right, lower, upper)
    return image[left:right, lower:upper, :], mask_512[left:right, lower:upper, :]


def crop(df):
    for n in tqdm(range(0, len(df))):
        item = df.loc[n]
        if item['series_id'] != 51678:
            continue
        print(item['patient_id'])
        if item['mask_file']:
            mask_org = nib.load(item['mask_file']).get_fdata()
            shape = mask_org.shape
            mask_org = mask_org.transpose(1, 0, 2)[::-1, :, :]  # (d, w, h)
            mask = np.zeros((out_dim, shape[1], shape[0], shape[2]))
            for cid in range(out_dim):
                mask[cid] = (mask_org == (cid + 1))
            mask = mask.astype(np.uint8)
            mask = R(mask).numpy()
        else:
            mask = np.load('our_seg/' + str(item['series_id']) + '.npy')

        mapping = ['liver', 'spleen', 'left_kidney', 'right_kidney', 'bowel']
        assert out_dim == 5
        image_sizes = [512, 512, 128]
        R2 = Resize(image_sizes)
        mask_512 = R2(mask).numpy().astype(np.uint8)

        for i in range(1):
            image = load_png_line_par_512(item['image_path'])
            image_crop, mask_512_crop = get_crop(image, mask_512[i], mask[i])
            image_crop = np.concatenate([image_crop, mask_512_crop], axis=2)
            np.save(f"crop_{mapping[i]}/{item['series_id']}_{mapping[i]}.npy", image_crop)


def check(image_path):
    image = np.load(image_path)[:, :, 218]
    print(image.shape)
    image = np.expand_dims(image, 0).repeat(3, 0)
    f, axarr = plt.subplots()
    axarr.imshow(torch.tensor(image).transpose(0, 1).transpose(1,2).squeeze())
    plt.show()


if __name__ == '__main__':
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
    df['image_path'] = 'train_data/' + df['patient_id'].astype(str) + '_' + df['series_id'].astype(str)
    df['mask_file'].fillna('', inplace=True)
    # df_to_seg = df.query('mask_file == ""').reset_index(drop=True)

    crop(df)

    # check('crop_liver/51678_liver.npy') #31781