import torch


class Trainer():
    """Trainer class to train and evaluate a model."""

    def __init__(
            self,
            model,
            optimizer,
            device):
        """
        :param model: the model we want to train.
        :param loss_function: the loss_function to minimize.
        :param optimizer: the optimizer used to minimize the loss_function.
        :param device: torch.device("cuda") or torch.device("cpu")
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device


    def train(self, train_dataset, valid_dataset, epochs: int):
        """Train over a dataset.

        :param train_dataset: a Dataset or DatasetLoader instance containing
            the training instances.
        :param valid_dataset: a Dataset or DatasetLoader instance used to
            evaluate learning progress.
        :param epochs: the number of times to iterate over train_dataset.

        :return avg_train_loss: the average training loss on train_dataset over
                epochs.
        """
        print('Training...')

        train_loss = 0.0
        for epoch in range(epochs):
            print(' Epoch {:03d}'.format(epoch + 1))

            epoch_loss = 0.0

            for step, sample in enumerate(train_dataset):
                inputs = sample[0].type(torch.float32).to(self.device)
                labels = sample[1].to(self.device)

                # we need to set the gradients to zero before starting to do
                # backpropragation because PyTorch accumulates the gradients
                # on subsequent backward passes
                self.optimizer.zero_grad()

                sample_loss = self.model.loss(inputs, labels)

                sample_loss.backward()
                self.optimizer.step()

                # sample_loss is a Tensor, tolist returns a float
                # (alternative: use float() instead of .tolist())
                epoch_loss += sample_loss.tolist()
                if step % 1000 == 0:
                    print('    [E: {:2d} @ step {}] current avg loss = {:0.4f}'.format(
                        epoch, step, epoch_loss / (step + 1)))

            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss
            print('  [E: {:2d}] train loss = {:0.4f}'.format(
                epoch, avg_epoch_loss))

            valid_loss = self.evaluate(valid_dataset)

            print('  [E: {:2d}] valid loss = {:0.4f}'.format(
                epoch, valid_loss))

        print('... Done!')

        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss, valid_loss

    def evaluate(self, valid_dataset, wandb = None):
        """
        Args:
            valid_dataset: the dataset to use to evaluate the model.

        Returns:
            avg_valid_loss: the average validation loss over valid_dataset.
        """
        valid_loss = 0.0

        # no gradient updates here
        with torch.no_grad():
            for sample in valid_dataset:
                inputs = sample[0].type(
                    torch.float32).unsqueeze(0).to(self.device)
                labels = sample[1].unsqueeze(0).to(self.device)

                predictions = self.model(inputs)
                sample_loss = self.model.loss(inputs, labels)
                valid_loss += sample_loss.tolist()

        return valid_loss / len(valid_dataset)

    def predict(self, x):
        """
        Returns: hopefully the right prediction.
        """
        return self.model(x).tolist()
