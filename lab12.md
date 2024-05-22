# Wstęp

Metody uczenia maszynowego możemy podzielić na dwie główne kategorie (pomijając uczenie ze wzmocnieniem): nadzorowane i nienadzorowane. Uczenie **nadzorowane** (ang. *supervised*) to jest uczenie z dostępnymi etykietami dla danych wejściowych. Na parach danych uczących $dataset= \{(x_0,y_0), (x_1,y_1), \ldots, (x_n,y_n)\}$ model ma za zadanie nauczyć się funkcji $f: X \rightarrow Y$. Z kolei modele uczone w sposób **nienadzorowany** (ang. *unsupervised*) wykorzystują podczas trenowania dane nieetykietowane tzn. nie znamy $y$ z pary $(x, y)$.

Dość częstą sytuacją, z jaką mamy do czynienia, jest posiadanie małego podziobioru danych etykietowanych i dużego nieetykietowanych. Często annotacja danych wymaga ingerencji człowieka - ktoś musi określić co jest na obrazku, ktoś musi powiedzieć czy dane słowo jest rzeczownkiem czy czasownikiem itd.

Jeżeli mamy dane etykietowane do zadania uczenia nadzorowanego (np. klasyfikacja obrazka), ale także dużą ilość danych nieetykietowanych, to możemy wtedy zastosować techniki **uczenia częściowo nadzorowanego** (ang. *semi-supervised learning*). Te techniki najczęściej uczą się funkcji $f: X \rightarrow Y$, ale jednocześnie są w stanie wykorzystać informacje z danych nieetykietowanych do poprawienia działania modelu.

## Cel ćwiczenia

Celem ćwiczenia jest nauczenie modelu z wykorzystaniem danych etykietowanych i nieetykietowanych ze zbioru STL10 z użyciem metody [Bootstrap your own latent](https://arxiv.org/abs/2006.07733).

Metoda ta jest relatywnie "lekka" obliczeniowo, a także dość prosta do zrozumienia i zaimplementowania, dlatego też na niej się skupimy na tych laboratoriach.

# Zbiór STL10

Zbiór STL10 to zbiór stworzony i udostępniony przez Stanford [[strona]](https://ai.stanford.edu/~acoates/stl10/) [[papier]](https://cs.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf) a inspirowany przez CIFAR-10. Obrazy zostały pozyskane z [ImageNet](https://image-net.org/). Szczegóły można doczytać na ich stronie. To co jest ważne to to, że autorzy zbioru dostarczają predefiniowany plan eksperymentalny, żeby móc porównywać łatwo wyniki eksperymentów. Nie będziemy go tutaj stosować z uwagi na jego czasochłonność (10 foldów), ale warto pamiętać o tym, że często są z góry ustalone sposoby walidacji zaprojetowanych przez nas algorytmów na określonych zbiorach referencyjnych.

Korzystając z `torchvision.datasets` ***załaduj*** 3 podziały zbioru danych STL10: `train`, `test`, `unlabeled` oraz utwórz z nich instancje klasy `DataLoader`. Korzystając z Google Colab rozważ użycie Google Drive do przechowyania zbioru w calu zaoszczędzenia czasu na wielokrotne pobieranie.


```python
import torchvision
from torch.utils.data import DataLoader

train_data = torchvision.datasets.STL10(
    root="./data",
    split="train",
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
test_data = torchvision.datasets.STL10(
    root="./data",
    split="test",
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
unlabelled_data = torchvision.datasets.STL10(
    root="./data",
    split="unlabeled",
    download=True,
    transform=torchvision.transforms.ToTensor(),
)


train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
unlabelled_loader = DataLoader(unlabelled_data, batch_size=64, shuffle=True)
```

    Downloading http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz to ./data/stl10_binary.tar.gz


    100%|██████████| 2640397119/2640397119 [07:44<00:00, 5679391.68it/s] 


    Extracting ./data/stl10_binary.tar.gz to ./data
    Files already downloaded and verified
    Files already downloaded and verified



```python
train_data.labels
```




    array([1, 5, 1, ..., 1, 7, 5], dtype=uint8)



# Uczenie nadzorowane

Żeby porównać czy metoda BYOL przynosi nam jakieś korzyści musimy wyznaczyć wartość bazową metryk(i) jakości, których będziemu używać (np. dokładność).

***Zaimplementuj*** wybraną metodę uczenia nadzorowanego na danych `train` z STL10. Możesz wykorzystać predefiniowane architektury w `torchvision.models` oraz kody źródłowe z poprzednich list.


```python
from torchvision.models import alexnet, AlexNet, AlexNet_Weights
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the pre-trained model
model = alexnet(weights=AlexNet_Weights.DEFAULT)

# Modify the final layer to match the number of classes in STL-10 (10 classes)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(
    model: AlexNet,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = 10,
):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # print(images)
            # print(labels)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")


def evaluate(model: AlexNet, test_loader: DataLoader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy: {accuracy}")


def evaluate_classification_report(model: AlexNet, test_loader: DataLoader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=test_data.classes))


train(model, train_loader, criterion, optimizer, num_epochs=1)

evaluate(model, test_loader)
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[38], line 81
         76             y_pred.extend(predicted.cpu().numpy())
         78     print(classification_report(y_true, y_pred, target_names=test_data.classes))
    ---> 81 train(model, train_loader, criterion, optimizer, num_epochs=1)
         83 evaluate(model, test_loader)


    Cell In[38], line 42, in train(model, train_loader, criterion, optimizer, num_epochs)
         40     loss = criterion(outputs, labels)
         41     loss.backward()
    ---> 42     optimizer.step()
         43     running_loss += loss.item()
         45 print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")


    File ~/projects/ai/gsn-l/venv/lib/python3.11/site-packages/torch/optim/optimizer.py:385, in Optimizer.profile_hook_step.<locals>.wrapper(*args, **kwargs)
        380         else:
        381             raise RuntimeError(
        382                 f"{func} must return None or a tuple of (new_args, new_kwargs), but got {result}."
        383             )
    --> 385 out = func(*args, **kwargs)
        386 self._optimizer_step_code()
        388 # call optimizer step post hooks


    File ~/projects/ai/gsn-l/venv/lib/python3.11/site-packages/torch/optim/optimizer.py:76, in _use_grad_for_differentiable.<locals>._use_grad(self, *args, **kwargs)
         74     torch.set_grad_enabled(self.defaults['differentiable'])
         75     torch._dynamo.graph_break()
    ---> 76     ret = func(self, *args, **kwargs)
         77 finally:
         78     torch._dynamo.graph_break()


    File ~/projects/ai/gsn-l/venv/lib/python3.11/site-packages/torch/optim/adam.py:166, in Adam.step(self, closure)
        155     beta1, beta2 = group['betas']
        157     has_complex = self._init_group(
        158         group,
        159         params_with_grad,
       (...)
        163         max_exp_avg_sqs,
        164         state_steps)
    --> 166     adam(
        167         params_with_grad,
        168         grads,
        169         exp_avgs,
        170         exp_avg_sqs,
        171         max_exp_avg_sqs,
        172         state_steps,
        173         amsgrad=group['amsgrad'],
        174         has_complex=has_complex,
        175         beta1=beta1,
        176         beta2=beta2,
        177         lr=group['lr'],
        178         weight_decay=group['weight_decay'],
        179         eps=group['eps'],
        180         maximize=group['maximize'],
        181         foreach=group['foreach'],
        182         capturable=group['capturable'],
        183         differentiable=group['differentiable'],
        184         fused=group['fused'],
        185         grad_scale=getattr(self, "grad_scale", None),
        186         found_inf=getattr(self, "found_inf", None),
        187     )
        189 return loss


    File ~/projects/ai/gsn-l/venv/lib/python3.11/site-packages/torch/optim/adam.py:316, in adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, foreach, capturable, differentiable, fused, grad_scale, found_inf, has_complex, amsgrad, beta1, beta2, lr, weight_decay, eps, maximize)
        313 else:
        314     func = _single_tensor_adam
    --> 316 func(params,
        317      grads,
        318      exp_avgs,
        319      exp_avg_sqs,
        320      max_exp_avg_sqs,
        321      state_steps,
        322      amsgrad=amsgrad,
        323      has_complex=has_complex,
        324      beta1=beta1,
        325      beta2=beta2,
        326      lr=lr,
        327      weight_decay=weight_decay,
        328      eps=eps,
        329      maximize=maximize,
        330      capturable=capturable,
        331      differentiable=differentiable,
        332      grad_scale=grad_scale,
        333      found_inf=found_inf)


    File ~/projects/ai/gsn-l/venv/lib/python3.11/site-packages/torch/optim/adam.py:439, in _single_tensor_adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, grad_scale, found_inf, amsgrad, has_complex, beta1, beta2, lr, weight_decay, eps, maximize, capturable, differentiable)
        437         denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
        438     else:
    --> 439         denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        441     param.addcdiv_(exp_avg, denom, value=-step_size)
        443 # Lastly, switch back to complex view


    KeyboardInterrupt: 



```python
evaluate_classification_report(model, test_loader)
```

                  precision    recall  f1-score   support
    
        airplane       0.10      1.00      0.18       800
            bird       0.00      0.00      0.00       800
             car       0.00      0.00      0.00       800
             cat       0.00      0.00      0.00       800
            deer       0.00      0.00      0.00       800
             dog       0.00      0.00      0.00       800
           horse       0.00      0.00      0.00       800
          monkey       0.00      0.00      0.00       800
            ship       0.00      0.00      0.00       800
           truck       0.00      0.00      0.00       800
    
        accuracy                           0.10      8000
       macro avg       0.01      0.10      0.02      8000
    weighted avg       0.01      0.10      0.02      8000
    


      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
      _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))



```python
print(len(test_data))
print(test_loader.dataset.classes)
print(test_data[0][0].size())
print(test_data[0][1])
```

    8000
    ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    torch.Size([3, 96, 96])
    6


# Bootstrap your own latent

Metoda [Bootstrap your own latent](https://arxiv.org/abs/2006.07733) jest opisana w rodziale 3.1 papieru a także w dodatku A. Składa się z dwóch etapów:


1.   uczenia samonadzorowanego (ang. *self-supervised*)
2.   douczania nadzorowanego (ang. *fine-tuning*)

## Uczenie samonadzorowane

Architektura do nauczania samonadzorowanego składa się z dwóch sieci: (1) *online* i (2) *target*. W uproszczeniu cała architektura działa tak:


1.   Dla obrazka $x$ wygeneruj dwie różne augmentacje $v$ i $v'$ za pomocą funkcji $t$ i $t'$.
2.   Widok $v$ przekazujemy do sieci *online*, a $v'$ do *target*.
3.   Następnie widoki przekształacamy za pomocą sieci do uczenia reprezentacji (np. resnet18 lub resnet50) do reprezentacji $y_\theta$ i $y'_\xi$.
4.   Potem dokonujemy projekcji tych reprezentacji w celu zmniejszenia wymiarowości (np. za pomocą sieci MLP).
5.   Na sieci online dokonujmey dodatkowo predykcji pseudo-etykiety (ang. *pseudolabel*)
6.   Wyliczamy fukncję kosztu: MSE z wyjścia predyktora sieci *online* oraz wyjścia projekcji sieci *target* "przepuszczonej" przez predyktor sieci *online* **bez propagacji wstecznej** (*vide Algorithm 1* z papieru).
7.   Dokonujemy wstecznej propagacji **tylko** po sieci *online*.
8.   Aktualizujemy wagi sieci *target* sumując w ważony sposób wagi obu sieci $\xi = \tau\xi + (1 - \tau)\theta$ ($\tau$ jest hiperprametrem) - jest to ruchoma średnia wykładnicza (ang. *moving exponential average*).

Po zakończeniu procesu uczenia samonadzorowanego zostawiamy do douczania sieć kodera *online* $f_\theta$. Cała sieć *target* oraz warstwy do projekcji i predykcji w sieci *online* są "do wyrzucenia".

### Augmentacja

Dodatek B publikacji opisuje augmentacje zastosowane w metodzie BYOL. Zwróć uwagę na tabelę 6 w publikacji. `torchvision.transforms.RandomApply` może być pomocne.

***Zaimeplementuj*** augmentację $\tau$ i $\tau'$.



```python
import random
from torch import nn
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

# Parameter T T′
# Random crop probability 1.0 1.0 x
# Flip probability 0.5 0.5 x
# Color jittering probability 0.8 0.8 x

# Brightness adjustment max intensity 0.4 0.4
# Contrast adjustment max intensity 0.4 0.4
# Saturation adjustment max intensity 0.2 0.2
# Hue adjustment max intensity 0.1 0.1

# Color dropping probability 0.2 0.2 x
# Gaussian blurring probability 1.0 0.1 x
# Solarization probability 0.0 0.2


def get_t1_aug():
    transform = T.Compose(
        [
            T.RandomResizedCrop(size=(96, 96)),
            T.RandomHorizontalFlip(p=0.5),
            RandomApply(
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                p=0.8,
            ),
            RandomApply(T.Grayscale(num_output_channels=3), p=0.2),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        ]
    )
    return transform


def get_t2_aug():
    transform = T.Compose(
        [
            T.RandomResizedCrop(size=(94, 94)),
            T.RandomHorizontalFlip(p=0.5),
            RandomApply(
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                p=0.8,
            ),
            RandomApply(T.Grayscale(num_output_channels=3), p=0.2),
            RandomApply(T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)), p=0.1),
            RandomApply(F.solarize, p=0.2),
        ]
    )
    return transform


class RandomApply(nn.Module):
    def __init__(self, fn: nn.Module, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)
```

### Implementacja uczenia samonadzorowanego

***Zaprogramuj*** proces uczenia samonadzorowanego na danych nieetykietowanych ze zbioru STL10.

Wskazówki do realizacji polecenia:

1. Proces uczenia może trwać bardzo długo dlatego zaleca się zastsowanie wczesnego zatrzymania lub uczenia przez tylko jedną epokę. Mimo wszystko powinno się dać osiągnąć poprawę w uczeniu nadzorowanym wykorzystując tylko zasoby z Google Colab.
2. Dobrze jest pominąć walidację na zbiorze treningowym i robić ją tylko na zbiorze walidacyjnym - zbiór treningowy jest ogromny i w związku z tym narzut czasowy na walidację też będzie duży.
3. Walidację modelu można przeprowadzić na zbiorze `train` lub całkowicie ją pominąć, jeżeli uczymy na stałej ilości epok.
4. Rozważ zastosowanie tylko jednej augmentacji - augmentacja $\tau'$ jest bardziej czasochłonna niż $\tau$.
5. Poniżej jest zaprezentowany zalążek kodu - jest on jedynie wskazówką i można na swój sposób zaimplementować tę metodę


```python
from copy import deepcopy
from json import encoder
from torch import le, nn
from torch import nn, Tensor
from torch.nn import functional as F

from src.ssl_base import SSLBase
import copy


class SmallConvnet(nn.Module):
    """Small ConvNet (avoids heavy computation)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 21 * 21, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 16 * 21 * 21)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, plain_last: bool = False
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        if not plain_last:
            self.net.append(nn.BatchNorm1d(output_dim))
            self.net.append(nn.ReLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def mlp(dim: int, projection_size: int = 256) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, projection_size),
        nn.ReLU(),
        nn.BatchNorm1d(projection_size),
        nn.ReLU(),
    )


class BYOLModel(SSLBase):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        tau: float,
        out_channels: int = 10,
    ):
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            out_channels=out_channels,
        )

        # Initialize online network
        # funkcja f
        self.online_encoder = SmallConvnet()

        # funkcja g
        self.online_projector = MLP(84, 84, 84, plain_last=False)

        # funkcja q
        self.online_predictor = MLP(84, 84, 84, plain_last=True)
        self.online_net = nn.Sequential(
            self.online_encoder,
            self.online_projector,
            self.online_predictor,
        )

        # Initialize target network with frozen weights
        self.target_encoder = self.copy_and_freeze_module(self.online_encoder)
        self.target_projector = self.copy_and_freeze_module(self.online_projector)
        self.target_net = nn.Sequential(self.target_encoder, self.target_projector)

        # Initialize augmentations
        self.aug_1 = get_t1_aug()
        self.aug_2 = get_t1_aug()

        self.tau = tau

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        t = self.aug_1(x)
        t_prim = self.aug_2(x)

        q = self.online_net(t)
        q_sym = self.online_net(t_prim)

        with torch.no_grad():
            z_prim = self.target_net(t_prim)
            z_prim_sym = self.target_net(t)

        q = torch.cat([q, q_sym], dim=0)
        z_prim = torch.cat([z_prim, z_prim_sym], dim=0)

        return q, z_prim

    def forward_repr(self, x: Tensor) -> Tensor:
        return self.online_encoder(x)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, _ = batch
        q, z_prim = self.forward(x)
        loss = self.byol_loss(q=q, z_prim=z_prim)

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def byol_loss(self, q: Tensor, z_prim: Tensor) -> Tensor:
        q = F.normalize(q, dim=-1, p=2)
        z_prim = F.normalize(z_prim, dim=-1, p=2)
        return (2 - 2 * (q * z_prim).sum(dim=-1)).mean()

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.update_target_network()

    @torch.no_grad()
    def update_target_network(self) -> None:
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data = (
                self.tau * target_param.data + (1 - self.tau) * online_param.data
            )

    @staticmethod
    def copy_and_freeze_module(model: nn.Module) -> nn.Module:
        mode_copy = copy.deepcopy(model)
        for param in mode_copy.parameters():
            param.requires_grad = False

        return mode_copy


aug_1 = get_t1_aug()
aug_2 = get_t1_aug()

model = BYOLModel(
    learning_rate=1e-3,
    weight_decay=1e-5,
    tau=0.99,
    out_channels=10,
)
```


```python
from cgitb import small
from lightning import Trainer

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
TAU = 0.99
EPOCHS = 200
ACCELERATOR = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "logs"

# logger = TensorBoardLogger(save_dir=OUT_DIR, default_hp_metric=False)

x = next(iter(train_loader))[0].size()


trainer = Trainer(
    default_root_dir=OUT_DIR,
    max_epochs=EPOCHS,
    # logger=logger,
    accelerator=ACCELERATOR,
    num_sanity_val_steps=0,
    log_every_n_steps=10,
)

trainer.fit(model, train_loader)
trainer.test(model, test_loader)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    
      | Name             | Type         | Params
    --------------------------------------------------
    0 | online_encoder   | SmallConvnet | 859 K 
    1 | online_projector | MLP          | 14.6 K
    2 | online_predictor | MLP          | 14.4 K
    3 | online_net       | Sequential   | 888 K 
    4 | target_encoder   | SmallConvnet | 859 K 
    5 | target_projector | MLP          | 14.6 K
    6 | target_net       | Sequential   | 874 K 
    --------------------------------------------------
    888 K     Trainable params
    874 K     Non-trainable params
    1.8 M     Total params
    7.054     Total estimated model params size (MB)


    Epoch 0:   0%|          | 0/79 [00:00<?, ?it/s] 

    Testing: |          | 0/? [00:00<?, ?it/s]2,  0.41it/s, v_num=11, train/loss=1.980]


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[63], line 26
         16 trainer = Trainer(
         17     default_root_dir=OUT_DIR,
         18     max_epochs=EPOCHS,
       (...)
         22     log_every_n_steps=10,
         23 )
         25 trainer.fit(model, train_loader)
    ---> 26 trainer.test(model, test_loader)


    File ~/projects/ai/gsn-l/venv/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py:754, in Trainer.test(self, model, dataloaders, ckpt_path, verbose, datamodule)
        752 self.state.status = TrainerStatus.RUNNING
        753 self.testing = True
    --> 754 return call._call_and_handle_interrupt(
        755     self, self._test_impl, model, dataloaders, ckpt_path, verbose, datamodule
        756 )


    File ~/projects/ai/gsn-l/venv/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py:44, in _call_and_handle_interrupt(trainer, trainer_fn, *args, **kwargs)
         42     if trainer.strategy.launcher is not None:
         43         return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
    ---> 44     return trainer_fn(*args, **kwargs)
         46 except _TunerExitException:
         47     _call_teardown_hook(trainer)


    File ~/projects/ai/gsn-l/venv/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py:794, in Trainer._test_impl(self, model, dataloaders, ckpt_path, verbose, datamodule)
        790 assert self.state.fn is not None
        791 ckpt_path = self._checkpoint_connector._select_ckpt_path(
        792     self.state.fn, ckpt_path, model_provided=model_provided, model_connected=self.lightning_module is not None
        793 )
    --> 794 results = self._run(model, ckpt_path=ckpt_path)
        795 # remove the tensors from the test results
        796 results = convert_tensors_to_scalars(results)


    File ~/projects/ai/gsn-l/venv/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py:987, in Trainer._run(self, model, ckpt_path)
        982 self._signal_connector.register_signal_handlers()
        984 # ----------------------------
        985 # RUN THE TRAINER
        986 # ----------------------------
    --> 987 results = self._run_stage()
        989 # ----------------------------
        990 # POST-Training CLEAN UP
        991 # ----------------------------
        992 log.debug(f"{self.__class__.__name__}: trainer tearing down")


    File ~/projects/ai/gsn-l/venv/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py:1026, in Trainer._run_stage(self)
       1023 self.lightning_module.zero_grad(**zero_grad_kwargs)
       1025 if self.evaluating:
    -> 1026     return self._evaluation_loop.run()
       1027 if self.predicting:
       1028     return self.predict_loop.run()


    File ~/projects/ai/gsn-l/venv/lib/python3.11/site-packages/lightning/pytorch/loops/utilities.py:182, in _no_grad_context.<locals>._decorator(self, *args, **kwargs)
        180     context_manager = torch.no_grad
        181 with context_manager():
    --> 182     return loop_run(self, *args, **kwargs)


    File ~/projects/ai/gsn-l/venv/lib/python3.11/site-packages/lightning/pytorch/loops/evaluation_loop.py:114, in _EvaluationLoop.run(self)
        112     return []
        113 self.reset()
    --> 114 self.on_run_start()
        115 data_fetcher = self._data_fetcher
        116 assert data_fetcher is not None


    File ~/projects/ai/gsn-l/venv/lib/python3.11/site-packages/lightning/pytorch/loops/evaluation_loop.py:245, in _EvaluationLoop.on_run_start(self)
        243 self._on_evaluation_model_eval()
        244 self._on_evaluation_start()
    --> 245 self._on_evaluation_epoch_start()


    File ~/projects/ai/gsn-l/venv/lib/python3.11/site-packages/lightning/pytorch/loops/evaluation_loop.py:326, in _EvaluationLoop._on_evaluation_epoch_start(self, *args, **kwargs)
        324 hook_name = "on_test_epoch_start" if trainer.testing else "on_validation_epoch_start"
        325 call._call_callback_hooks(trainer, hook_name, *args, **kwargs)
    --> 326 call._call_lightning_module_hook(trainer, hook_name, *args, **kwargs)


    File ~/projects/ai/gsn-l/venv/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py:157, in _call_lightning_module_hook(trainer, hook_name, pl_module, *args, **kwargs)
        154 pl_module._current_fx_name = hook_name
        156 with trainer.profiler.profile(f"[LightningModule]{pl_module.__class__.__name__}.{hook_name}"):
    --> 157     output = fn(*args, **kwargs)
        159 # restore current_fx when nested context
        160 pl_module._current_fx_name = prev_fx_name


    File ~/projects/ai/gsn-l/src/ssl_base.py:39, in SSLBase.on_test_epoch_start(self)
         35 def on_test_epoch_start(self) -> None:
         36     """Before computing reprs and scores for test set,
         37      updates downstream model with train reprs.
         38     """
    ---> 39     self._update_downstream_model_with_train_representations()


    File ~/projects/ai/gsn-l/src/ssl_base.py:57, in SSLBase._update_downstream_model_with_train_representations(self)
         54 """Resets state of the downstream model and computes representations of the train set."""
         55 self.downstream_model.reset()
    ---> 57 for batch in self.trainer.datamodule.train_dataloader():  # type: ignore
         58     # Iterating data_loader manually requires manual device change:
         59     x, y = batch
         60     x, y = x.to(self.device), y.to(self.device)


    AttributeError: 'NoneType' object has no attribute 'train_dataloader'


## Douczanie nadzorowane

***Zaimplementuj*** proces douczania kodera z poprzedniego polecenia na danych etykietowanych ze zbioru treningowego. Porównaj jakość tego modelu z modelem nauczonym tylko na danych etykietownaych. Postaraj się wyjaśnić różnice.


```python
state_dict = model.encoder_online.state_dict()
encoder = ___
encoder.load_state_dict(state_dict)

___
```


```python
class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 21 * 21, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = x.reshape(-1, 16 * 21 * 21)
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        return x


model = Test()
x = torch.randn(1, 3, 96, 96)
model(x)
```

    torch.Size([1, 6, 46, 46])
    torch.Size([1, 16, 21, 21])
    torch.Size([1, 7056])
    torch.Size([1, 120])
    torch.Size([1, 84])





    tensor([[0.0586, 0.0000, 0.0000, 0.0271, 0.0128, 0.0000, 0.0532, 0.0754, 0.0000,
             0.0000, 0.2171, 0.1203, 0.1092, 0.0385, 0.0964, 0.0000, 0.0000, 0.1251,
             0.0408, 0.0000, 0.0000, 0.0065, 0.0000, 0.0007, 0.0890, 0.0884, 0.2195,
             0.0000, 0.0000, 0.1868, 0.0000, 0.0000, 0.1056, 0.0000, 0.0484, 0.0000,
             0.0000, 0.0000, 0.0000, 0.0566, 0.0795, 0.0000, 0.0276, 0.0652, 0.0065,
             0.1108, 0.0288, 0.0423, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0147,
             0.0000, 0.0402, 0.0000, 0.0000, 0.0307, 0.0000, 0.2457, 0.0000, 0.0347,
             0.0000, 0.0000, 0.0138, 0.0000, 0.0000, 0.1716, 0.0000, 0.0000, 0.1002,
             0.1924, 0.0319, 0.0000, 0.0508, 0.1076, 0.0672, 0.0000, 0.0000, 0.0879,
             0.0000, 0.0661, 0.0000]], grad_fn=<ReluBackward0>)




```python
class SmallConvnet(nn.Module):
    """Small ConvNet (avoids heavy computation)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = x.reshape(-1, 16 * 4 * 4)
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        return x


x = torch.randn(64, 1, 28, 28)

small_convnet = SmallConvnet()
small_convnet(x)
```

    torch.Size([64, 6, 12, 12])
    torch.Size([64, 16, 4, 4])
    torch.Size([64, 256])
    torch.Size([64, 120])
    torch.Size([64, 84])





    tensor([[0.0527, 0.0961, 0.0742,  ..., 0.0455, 0.0819, 0.0000],
            [0.0785, 0.1323, 0.0717,  ..., 0.0489, 0.1071, 0.0000],
            [0.0464, 0.1324, 0.0548,  ..., 0.0360, 0.0688, 0.0000],
            ...,
            [0.1048, 0.1357, 0.0536,  ..., 0.0753, 0.1009, 0.0000],
            [0.0809, 0.1380, 0.0169,  ..., 0.0427, 0.0688, 0.0000],
            [0.0822, 0.1866, 0.0576,  ..., 0.0437, 0.1325, 0.0000]],
           grad_fn=<ReluBackward0>)


