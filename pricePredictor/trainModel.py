import torch
from torch.utils.data import DataLoader
from dataset import PricePredictorDataset
from predictorModel import PricePredictor


def createDataloader(train_percentage=0.95):
    house_dataset = PricePredictorDataset("../data/data.csv")
    train_size = int(train_percentage * len(house_dataset))
    eval_size = len(house_dataset) - train_size
    train_subset, eval_subset = torch.utils.data.random_split(
        house_dataset, [train_size, eval_size])

    train_data_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
    eval_data_loader = DataLoader(eval_subset, batch_size=1, shuffle=True)

    return train_data_loader, eval_data_loader


def trainTest(model, criterion, optimizer, train_loader, test_loader):
    for train_batch in train_loader:
        data = train_batch[0][0]
        targets = train_batch[1][0]


if __name__ == "__main__":
    train_data_loader, eval_data_loader = createDataloader()

    model = PricePredictor()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 20
    for epoch in range(epochs):
        trainTest(model, criterion, optimizer,
                  train_data_loader, eval_data_loader)
