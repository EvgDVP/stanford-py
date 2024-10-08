import os
import zipfile
from io import BytesIO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

# Определение устройства (GPU или CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class PalmDataset(Dataset):
    def __init__(self, csv_file, zip_file, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.transform = transform
        self.zip_file = zip_file

        # Открываем ZIP-файл
        self.archive = zipfile.ZipFile(zip_file, 'r')

        # Фильтрация данных только для ладоней
        self.labels = self.labels[self.labels['aspectOfHand'].str.contains('palmar', case=False, na=False)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Получаем имя изображения
        img_name = self.labels.iloc[idx, 7]  # imageName
        img_path = f"Hands/{img_name}"  # Путь к файлу внутри архива

        # Извлекаем изображение из архива
        with self.archive.open(img_path) as img_file:
            img = Image.open(BytesIO(img_file.read())).convert("RGB")

        # Применяем трансформации, если они заданы
        if self.transform:
            img = self.transform(img)

        # Получаем метки
        age = self.labels.iloc[idx, 1]  # возраст
        skin_color = self.labels.iloc[idx, 3].lower()  # цвет кожи
        accessories = self.labels.iloc[idx, 4]  # наличие аксессуаров

        skin_color_mapping = {'very fair': 0, 'fair': 1, 'medium': 2, 'dark': 3}  # Маппинг цвета кожи
        skin_color_label = skin_color_mapping.get(skin_color, -1)

        # Преобразование меток в числовой формат (например, one-hot)
        label = torch.tensor([age, skin_color_label, accessories], dtype=torch.float32)

        return img, label

# Путь к CSV файлу с метками и путь к ZIP-файлу с изображениями
csv_file = 'content/dataset/HandInfo.csv'
zip_file = 'content/dataset/images/Hands.zip'

df = pd.read_csv(csv_file)
print(df.columns)

# Определяем преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Изменение размера изображения до 512x512
    transforms.ToTensor(),  # Преобразование изображения в тензор
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Нормализация
])

# Создаем датасет с использованием класса PalmDataset
dataset = PalmDataset(csv_file=csv_file, zip_file=zip_file, transform=transform)

# Создаем DataLoader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super(Generator, self).__init__()

        self.init_size = 4  # Начальный размер изображения 4x4
        self.l1 = nn.Sequential(nn.Linear(latent_dim + condition_dim, 512 * self.init_size * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4x4 -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32x32 -> 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 64x64 -> 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 8, 4, 2, 1),  # 128x128 -> 256x256
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.ConvTranspose2d(8, 3, 4, 2, 1),  # 256x256 -> 512x512
            nn.Tanh()  # Для нормализации значений пикселей в диапазоне от -1 до 1
        )

    def forward(self, z, condition):
        # Конкатенируем латентный вектор и условие
        x = torch.cat([z, condition], dim=1)

        # Преобразуем через полносвязный слой в начальный тензор
        out = self.l1(x)
        out = out.view(out.size(0), 512, self.init_size, self.init_size)  # Превращаем в тензор 4x4

        # Пропускаем через транспонированные свёртки для увеличения разрешения
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, condition_dim):
        super(Discriminator, self).__init__()

        # Сверточные блоки для обработки изображения
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 8, 4, 2, 1),  # 512x512 -> 256x256
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8, 16, 4, 2, 1),  # 256x256 -> 128x128
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, 4, 2, 1),  # 128x128 -> 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),  # 64x64 -> 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),  # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),  # 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1)  # 8x8 -> 4x4
        )

        # Полносвязный слой для объединения признаков изображения с условием
        self.fc_layer = nn.Sequential(
            nn.Linear(512 * 4 * 4 + condition_dim, 1)  # Увеличиваем входную размерность на condition_dim
        )

    def forward(self, img, condition):
        # Пропускаем изображение через сверточные блоки
        out = self.conv_blocks(img)

        # Разворачиваем тензор в вектор перед подачей на полносвязный слой
        out = out.view(out.size(0), -1)  # Размер: (batch_size, 512 * 4 * 4)

        # Конкатенируем признаки изображения с условием (например, возраст, цвет кожи, аксессуары)
        out = torch.cat([out, condition], dim=1)  # Объединяем с условием

        # Пропускаем через полносвязный слой для классификации
        validity = self.fc_layer(out)

        return validity

def compute_gradient_penalty(discriminator, real_samples, fake_samples, conditions):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = discriminator(interpolates, conditions)
    fake = torch.ones(real_samples.shape[0], 1).to(real_samples.device)  # Признак для реальных изображений

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()  # L2-норма
    return gradient_penalty

# Установка параметров для обучения
latent_dim = 100  # Размер латентного пространства
condition_dim = 3  # Размерность условных данных
num_epochs = 100
n_critic = 5  # Количество шагов для дискриминатора перед обновлением генератора
lr = 0.0001  # Начальная скорость обучения
weight_clip = 0.01  # Объектная функция для WGAN

# Инициализация моделей
generator = Generator(latent_dim, condition_dim).to(device)
discriminator = Discriminator(condition_dim).to(device)

# Оптимизаторы
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Обучение
for epoch in range(num_epochs):
    for i, (real_images, conditions) in enumerate(dataloader):
        real_images = real_images.to(device)
        conditions = conditions.to(device)

        # Обучение дискриминатора
        for _ in range(n_critic):
            optimizer_D.zero_grad()

            # Генерация случайного латентного вектора
            z = torch.randn(real_images.size(0), latent_dim).to(device)

            # Генерация фейковых изображений
            fake_images = generator(z, conditions)

            # Потеря для дискриминатора
            real_validity = discriminator(real_images, conditions)
            fake_validity = discriminator(fake_images.detach(), conditions)
            gradient_penalty = compute_gradient_penalty(discriminator, real_images, fake_images, conditions)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

            # Объектная функция WGAN
            for p in discriminator.parameters():
                p.data.clamp_(-weight_clip, weight_clip)

        # Обучение генератора
        optimizer_G.zero_grad()

        fake_validity = discriminator(fake_images, conditions)
        g_loss = -torch.mean(fake_validity)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Сохранение моделей каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        torch.save(generator.state_dict(), f'generator_epoch_{epoch + 1}.pth')
        torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch + 1}.pth')

print("Обучение завершено!")
