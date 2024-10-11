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
print(f"Используемое устройство: {device}")

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
        age = self.labels.iloc[idx, 1] / 100  # возраст
        skin_color = self.labels.iloc[idx, 3].lower()  # цвет кожи
        accessories = self.labels.iloc[idx, 4]  # наличие аксессуаров

        skin_color_mapping = {'very fair': 0, 'fair': 1, 'medium': 2, 'dark': 3}  # Маппинг цвета кожи
        skin_color_label = skin_color_mapping.get(skin_color, -1)

        # Преобразование меток в числовой формат
        label = torch.tensor([age, skin_color_label, accessories], dtype=torch.float32)

        return img, label

class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super(Generator, self).__init__()

        self.init_size = 4  # Начальный размер изображения 4x4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 1024 * self.init_size * self.init_size)
        )

        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            # 256x256 -> 512x512
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Tanh()  # Для нормализации значений пикселей в диапазоне от -1 до 1
        )

    def forward(self, z, condition):
        # Конкатенируем латентный вектор и условие
        x = torch.cat([z, condition], dim=1)

        # Преобразуем через полносвязный слой в начальный тензор
        out = self.l1(x)
        out = out.view(out.size(0), 1024, self.init_size, self.init_size)  # Превращаем в тензор 4x4

        # Пропускаем через транспонированные свёртки для увеличения разрешения
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, condition_dim):
        super(Discriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3 + condition_dim, 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Используем AdaptiveAvgPool для получения фиксированного размера выхода
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Выход будет 512x1x1
        self.fc = nn.Sequential(
            nn.Linear(512, 1),
        )

    def forward(self, img, condition):
        # Изменяем размер условия для совпадения с размером изображения
        condition_expanded = condition.view(condition.size(0), condition.size(1), 1, 1).expand(-1, -1, img.size(2),
                                                                                               img.size(3))

        # Конкатенируем условие как дополнительный канал к изображению
        img_input = torch.cat([img, condition_expanded], dim=1)

        # Обработка изображения через свёрточные слои
        img_features = self.conv_blocks(img_input)

        # Применяем адаптивное среднее значение для получения фиксированного размера
        img_features = self.pool(img_features)  # Получаем размер 512x1x1
        img_features = img_features.view(img_features.size(0), -1)  # Сглаживаем в 512

        # Пропускаем через полносвязный слой для получения оценки реальности изображения
        validity = self.fc(img_features)

        return validity

# Путь к CSV файлу с метками и путь к ZIP-файлу с изображениями
csv_file = 'content/dataset/HandInfo.csv'
zip_file = 'content/dataset/images/Hands.zip'

df = pd.read_csv(csv_file)
print(f"Столбцы CSV файла: {df.columns}")

# Определяем преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.RandomRotation(5, fill=(255, 255, 255))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Создаем датасет с использованием класса PalmDataset
dataset = PalmDataset(csv_file=csv_file, zip_file=zip_file, transform=transform)

# Создаем DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

os.makedirs('input_images', exist_ok=True)
os.makedirs('generated_images', exist_ok=True)

# Инициализация моделей
latent_dim = 1024  # Размер латентного пространства
condition_dim = 3  # Размерность условных данных
generator = Generator(latent_dim, condition_dim).to(device)
discriminator = Discriminator(condition_dim).to(device)

# Оптимизаторы
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

def save_input_image(tensor, img_idx):
    """Сохраняет входное изображение в формате PNG."""
    img = transforms.ToPILImage()(tensor.cpu().squeeze(0))
    img.save(f'input_images/input_image_{img_idx}.png')
    print(f'Входное изображение сохранено как input_images/input_image_{img_idx}.png')

def save_generated_image(tensor, img_idx):
    """Сохраняет сгенерированное изображение в формате PNG."""
    tensor = (tensor + 1) / 2  # Переводим значения в диапазон [0, 1]
    img = transforms.ToPILImage()(tensor.cpu().squeeze(0))
    img.save(f'generated_images/generated_image_{img_idx}.png')
    print(f'Сгенерированное изображение сохранено как generated_images/generated_image_{img_idx}.png')

# Функция для label smoothing (применяется только к реальным меткам)
def smooth_labels(labels, smoothing=0.1):
    return labels - smoothing

# Проводим обучение на одном батче
generator.train()
discriminator.train()

for i, (real_images, conditions) in enumerate(dataloader):
    real_images = real_images.to(device)
    conditions = conditions.to(device)
    save_input_image(real_images, i)

    # Выводим входные изображения и метки
    print(f"Входное изображение (тензор): {real_images}")
    print(f"Метки (тензор): {conditions}")

    # Обучение дискриминатора
    optimizer_D.zero_grad()

    # Генерация случайного латентного вектора
    z = torch.randn(real_images.size(0), latent_dim).to(device)

    # Генерация фейковых изображений
    fake_images = generator(z, conditions)

    # Выводим фейковые изображения
    print(f"Сгенерированные изображения (тензор): {fake_images}")

    # Валидность для дискриминатора
    real_validity = discriminator(real_images, conditions)
    fake_validity = discriminator(fake_images.detach(), conditions)

    # Применяем label smoothing к реальной валидности
    real_validity_smoothed = smooth_labels(real_validity, smoothing=0.1)

    # Расчет потерь дискриминатора
    d_loss = -torch.mean(real_validity_smoothed) + torch.mean(fake_validity)
    d_loss.backward()  # Потеря для дискриминатора
    optimizer_D.step()

    # Обучение генератора
    optimizer_G.zero_grad()

    # Генерация фейковых изображений
    fake_images = generator(z, conditions)
    save_generated_image(fake_images, i)
    fake_validity = discriminator(fake_images, conditions)
    g_loss = -torch.mean(fake_validity)  # Потеря для генератора
    g_loss.backward()
    optimizer_G.step()

    print(f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Останавливаем после одного батча
    break

