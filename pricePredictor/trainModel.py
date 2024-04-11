import os
import torch
from torch.utils.data import DataLoader
from dataset import PricePredictorDataset
from predictorModel import PricePredictor


def createDataloader(train_percentage=0.97):
    house_dataset = PricePredictorDataset("../data/data.csv")
    train_size = int(train_percentage * len(house_dataset))
    eval_size = len(house_dataset) - train_size
    train_subset, eval_subset = torch.utils.data.random_split(
        house_dataset, [train_size, eval_size])

    train_data_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
    eval_data_loader = DataLoader(eval_subset, batch_size=1, shuffle=True)

    return train_data_loader, eval_data_loader


def trainTest(model, criterion, optimizer, train_loader, test_loader):
    train_loss = 0
    for train_batch in train_loader:
        data = train_batch[0][0]
        targets = train_batch[1][0]

        inputs = data[:, 0:1]
        added_features = data[:, 1:]

        for iteration in range(inputs.shape[0] - 1):
            iteration_input = inputs[:iteration + 1, :]
            iteration_features = added_features[:iteration + 1, :]
            iteration_target = targets[iteration].unsqueeze(0) + 3

            optimizer.zero_grad()
            hidden_state = torch.tensor([[0.0] * model.d_model])
            prediction = None
            for row in range(iteration_input.shape[0]):
                current_input = iteration_input[row, :].unsqueeze(0)
                start_idx = max(row - 4, 0)
                end_idx = min(row + 1, iteration_features.shape[0])
                current_features = torch.mean(
                    iteration_features[start_idx: end_idx, :], dim=0, keepdim=True)
                prediction, hidden_state = model(
                    current_input, hidden_state, current_features)
            loss = criterion(prediction, iteration_target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

    avg_train_loss = int(train_loss / (len(train_loader) * 9))
    print("Average train loss: ", avg_train_loss)

    test_loss = 0
    for test_batch in test_loader:
        data = test_batch[0][0]
        targets = test_batch[1][0]

        inputs = data[:, 0:1]
        added_features = data[:, 1:]

        target = targets[-2].unsqueeze(0) + 3
        hidden_state = torch.tensor([[0.0] * model.d_model])
        prediction = None
        for row in range(inputs.shape[0] - 1):
            current_input = inputs[row, :].unsqueeze(0)
            start_idx = max(row - 4, 0)
            end_idx = min(row + 1, iteration_features.shape[0])
            current_features = torch.mean(
                added_features[start_idx: end_idx, :], dim=0, keepdim=True)
            prediction, hidden_state = model(
                current_input, hidden_state, current_features)
        print(
            f"Actual avg_mean_ppsf: {target[0].item()}, Predicted avg_mean_ppsf: {prediction[0].item()}")
        loss = criterion(prediction, target)
        test_loss += loss.item()

    avg_test_loss = int(test_loss / len(test_loader))
    print("Average tests loss: ", avg_test_loss)

    return avg_train_loss, avg_test_loss


if __name__ == "__main__":
    # train_data_loader, eval_data_loader = createDataloader()

    # model = PricePredictor()
    # if os.path.exists("predictor_original.pth"):
    #     print("Loading model from file")
    #     model.load_state_dict(torch.load("predictor_original.pth"))

    # criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    # epochs = 5000
    # for epoch in range(epochs):
    #     print("Epoch: ", epoch)
    #     avg_train_loss, avg_test_loss = trainTest(model, criterion, optimizer,
    #                                               train_data_loader, eval_data_loader)
    #     if avg_train_loss < 4000 and avg_test_loss < 50:
    #         print("Local minima found; Breaking")
    #         break
    # torch.save(model.state_dict(), "predictor_avg.pth")

    house_dataset = PricePredictorDataset("../data/data.csv")
    house_dataset.state_ppsf_stats("predictor_avg.pth")
