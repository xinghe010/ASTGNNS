import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def train(epoch, data_loader, model, optimizer, device, recorder):
    recorder.info('------starting {} epoch training------'.format(epoch))
    model.train()
    train_loss = 0.0
    total_targets = []
    total_pred_targets = []

    for i, batch in enumerate(data_loader, 1):
        optimizer.zero_grad()
        batch.to(device=device)
        loss, targets, pred_targets = model(batch)
        loss.backward(retain_graph=True)
        optimizer.step()
        total_targets.extend(targets.tolist())
        total_pred_targets.extend(pred_targets.tolist())
        train_loss += loss.cpu().item()
    accuracy = accuracy_score(total_targets, total_pred_targets)
    f1 = f1_score(total_targets, total_pred_targets)
    recall = recall_score(total_targets, total_pred_targets)
    precision = precision_score(total_targets, total_pred_targets)
    log = "train epoch[{}] end! train loss: {:.4f} train accuarcy: {:.2f}% train f1: {:.2f}% train recall: {:.2f}% train precision: {:.2f}%".format(
        epoch, train_loss / i, accuracy * 100, f1 * 100, recall * 100, precision * 100)
    recorder.info(log)
    return train_loss / i, accuracy, f1, recall, precision


def valid(epoch, data_loader, model, device, recorder):
    recorder.info('------starting {} epoch valid------'.format(epoch))
    model.eval()
    valid_loss = 0.0
    total_targets = []
    total_pred_targets = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader, 1):
            batch.to(device=device)
            loss, targets, pred_targets = model(batch)
            total_targets.extend(targets.tolist())
            total_pred_targets.extend(pred_targets.tolist())
            valid_loss += loss.cpu().item()

    accuracy = accuracy_score(total_targets, total_pred_targets)
    f1 = f1_score(total_targets, total_pred_targets)
    recall = recall_score(total_targets, total_pred_targets)
    precision = precision_score(total_targets, total_pred_targets)
    log = "valid epoch[{}] end! valid loss: {:.4f} valid accuarcy: {:.2f}% valid f1: {:.2f}% valid recall: {:.2f}% valid precision: {:.2f}%".format(
        epoch, valid_loss / i, accuracy * 100, f1 * 100, recall * 100, precision * 100)
    recorder.info(log)
    return valid_loss / i, accuracy, f1, recall, precision


def test(data_loader, model, device, recorder):
    recorder.info('------starting test------')
    model.eval()
    test_loss = 0.0
    total_targets = []
    total_pred_targets = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader, 1):
            batch.to(device=device)
            loss, targets, pred_targets = model(batch)
            total_targets.extend(targets.tolist())
            total_pred_targets.extend(pred_targets.tolist())
            test_loss += loss.cpu().item()
    accuracy = accuracy_score(total_targets, total_pred_targets)
    f1 = f1_score(total_targets, total_pred_targets)
    recall = recall_score(total_targets, total_pred_targets)
    precision = precision_score(total_targets, total_pred_targets)
    log = "test end! test loss: {:.4f} test accuarcy: {:.2f}% test f1: {:.2f}% test recall: {:.2f}% test precision: {:.2f}%".format(
        test_loss / i, accuracy * 100, f1 * 100, recall * 100, precision * 100)
    recorder.info(log)
    return test_loss / i, accuracy, f1, recall, precision
