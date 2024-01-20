from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import imagenet
from utils import *


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')

    model_student = create_student()
    model_student = model_student.to(device)

    weights_path = './evaluation/model_state2.pth'
    checkpoint = torch.load(weights_path, map_location=device)
    model_student.load_state_dict(checkpoint)
    model = model_student.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    data_folder = './data/test'
    imagenet_data = imagenet.ImageNet(data_folder, transform)
    train_dataloader = DataLoader(imagenet_data, batch_size=1, shuffle=True, generator=torch.Generator(device=device))


    topil = transforms.ToPILImage()
    with torch.no_grad():
        for i, image in tqdm(enumerate(train_dataloader), total=20):
            image = transforms.functional.resize(image, (90, 90), antialias=True)
            image = image.to(device)
            output = model_student(image)
            output = output.reshape(14, 14)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(image[0].permute(1, 2, 0).cpu().detach())
            axes[1].imshow(output.cpu().detach())
            plt.show()
            plt.close(fig)


