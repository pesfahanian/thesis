from config import Config

from data import data

from runtime import Runtime


def main():
    runtime = Runtime(data=data,
                      hidden_channels=Config.hidden_channels)

    for epoch in range(Config.epochs):
        loss = runtime.train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    test_acc = runtime.test()
    print(f'Test Accuracy: {test_acc:.4f}')


if __name__ == '__main__':
    main()
