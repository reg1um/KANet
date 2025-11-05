import torch

from kanet.core.kanet import KANet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def target_function(x):
    result = x[:, 0] ** 2 + x[:, 1] ** 2
    return result.unsqueeze(1)


def toy(x):
    return torch.sin(torch.pi * x[:, 0]) + x[:, 1] ** 2


model = KANet(
    layers_config=[
        {"in_feat": 2, "out_feat": 2, "num_knots": 10, "grid_range": (-1, 1)},
        {"in_feat": 2, "out_feat": 1, "num_knots": 10, "grid_range": (-1, 1)},
        {"in_feat": 1, "out_feat": 1, "num_knots": 10, "grid_range": (-1, 1)},
    ]
).to(device)

# optimizer = torch.optim.LBFGS(
#    model.parameters(),
#    lr=1,
#    line_search_fn='strong_wolfe',
#    tolerance_grad=1e-07,
#    tolerance_change=1e-09,
#    max_iter=10,
#    )


optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=5,
    max_eval=10,
    tolerance_grad=1e-5,
    tolerance_change=1e-7,
    history_size=10,
    line_search_fn="strong_wolfe",
)


num_epochs = 60
grid_extension_epochs = [20, 40]


X_train = torch.empty(2048, 2, device=device).uniform_(-1, 1)
X_test = torch.empty(512, 2, device=device).uniform_(-1, 1)
y_train = toy(X_train)
y_test = toy(X_test)

# target = toy(x)


def make_closure(optimizer, model, X_train, y_train):
    # To avoid binding warnings
    def closure():
        optimizer.zero_grad()
        y = model(X_train)
        loss = torch.sqrt(torch.nn.functional.mse_loss(y, y_train.unsqueeze(1)) + 1e-6)
        loss.backward()
        return loss

    return closure


for epoch in range(num_epochs):
    # Grid extension AFTER optimization step
    if epoch in grid_extension_epochs:
        print(f"\n>>> Grid extension at epoch {epoch}")
        model.grid_extension(X_train)
        # model.grid_widening(k_extend=4)
        # Reinitialize optimizer after structural change
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=5,
            max_eval=10,
            tolerance_grad=1e-5,
            tolerance_change=1e-7,
            history_size=10,
            line_search_fn="strong_wolfe",
        )
        optimizer.state.clear()

    closure = make_closure(optimizer, model, X_train, y_train)

    # torch.manual_seed(0)   # mimic the custom LBFGS
    loss = optimizer.step(closure)
    print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# Test the trained model
print("\nTesting trained model:")
with torch.no_grad():
    test_output = model(X_test)
    test_loss = torch.sqrt(
        torch.nn.functional.mse_loss(test_output, y_test.unsqueeze(1)) + 1e-6
    )
    print(f"Test Loss: {test_loss.item():.6f}")
    for i in range(min(10, X_test.shape[0])):
        print(
            f"Input: {X_test[i].cpu().numpy()}, Predicted: \
                {test_output[i].item():.6f}, Target: {y_test[i].item():.6f}"
        )
