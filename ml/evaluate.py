# evaluate.py
import torch
import torch.nn as nn

def evaluate(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()
    test_loss, test_correct = 0.0, 0
    class_correct = [0] * 6
    class_total   = [0] * 6

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss    += loss.item()
            predictions   = outputs.argmax(1)
            test_correct += (predictions == labels).sum().item()
            for label, pred in zip(labels, predictions):
                class_correct[label] += (pred == label).item()
                class_total[label]   += 1

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * test_correct / len(test_loader.dataset)
    print(f"\nResultats sur le jeu de test :")
    print(f"   Loss     : {avg_loss:.4f}")
    print(f"   Accuracy : {accuracy:.2f}%")
    print(f"\nPrecision par classe :")
    for i in range(6):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"   Classe {i} : {acc:.2f}%  ({class_correct[i]}/{class_total[i]})")
    return accuracy
