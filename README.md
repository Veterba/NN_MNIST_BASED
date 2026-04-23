# First MNIST Model project

##  Goal
- Make crucial start point model which based on recognizing numbers

##  Tasks
- [x] Find dataset
- [x] Write roadmap for entire project
- [ ] Phase 1: Preparing
- [ ] Phase 2: Building the Model
- [ ] Phase 3: Training
- [ ] Phase 4: Evaluation
- [ ] Phase 5: Experiments
- [ ] Phase 6: CNN
- [ ] Phase 7: Wrap Up

##  Notes
- TrochVision is a library for computer vision tasks based on PyTorch. It provides a suit of tools like, datasets, pre-trained models and other tools: [main documentation](https://docs.pytorch.org/vision/main/)

### Roadmap

---

**1. Preparing**

Dependencies: `torch`, `torchvision`, `matplotlib`

- [ ] **Load MNIST dataset with TorchVision**
  *Why first:* Before you build anything, you need to see and understand your data. Every ML project starts here — the data dictates everything: input shape, number of classes, preprocessing.
  *What it means:* `datasets.MNIST(download=True, transform=transforms.ToTensor())` downloads 70,000 grayscale digit images. `transforms.ToTensor()` converts each image from a PIL image (0–255 ints) to a PyTorch tensor with floats in [0, 1] — neural nets work better with small normalized numbers. You split into `train_data` (60k images the model learns from) and `test_data` (10k images you test on — the model never sees these during training, so they measure real performance).
  *What to check:* `len(train_data)`, `len(test_data)`, `train_data[0][0].shape` → should be `(1, 28, 28)` = 1 channel, 28 height, 28 width. `train_data[0][1]` → an integer label (0–9).

- [ ] **Wrap data in DataLoader**
  *Why here:* You can't feed 60,000 images at once — too much memory, and the math works better in small batches. DataLoader handles batching, shuffling, and iteration.
  *What it means:* `DataLoader(train_data, batch_size=64, shuffle=True)` groups images into batches of 64. `shuffle=True` randomizes order each epoch so the model doesn't memorize sequence. Test loader doesn't shuffle — order doesn't matter for evaluation.
  *What to check:* grab one batch with `images, labels = next(iter(train_loader))`, print `images.shape` → `(64, 1, 28, 28)` = 64 images in a batch.

- [ ] **Visualize 10 random samples with matplotlib**
  *Why here:* You should always eyeball your data before modeling. Are the images clean? Are labels correct? What does a "hard" digit look like?
  *What to do:* `plt.subplot` grid, show 10 images with their labels as titles. Look at the variety — some 4s look like 9s, some 1s are slanted. These are the cases your model will struggle with later.

---

**2. Building the Model**

- [ ] **Define DigitNet as nn.Module** (base architecture from [[MNIST]])
  *Why now:* You've seen the data and know the shapes. Now you build the function that takes an image and outputs a prediction. This is the core of the whole project.
  *What it means, layer by layer:*
  - `nn.Flatten()` — reshapes `(batch, 1, 28, 28)` → `(batch, 784)`. FC layers need a 1D vector, not a 2D image. You're throwing away spatial structure here (CNN fixes this later).
  - `nn.Linear(784, 128)` — a fully-connected layer. Mathematically: `output = input @ weights + bias`. 784 inputs, 128 outputs = 784×128 = 100,352 learnable parameters. Each of these 128 neurons "looks at" all 784 pixels and learns a different pattern.
  - `nn.ReLU()` — activation function: `max(0, x)`. Without it, stacking two Linear layers is mathematically identical to one Linear layer (matrix multiplication is linear). ReLU adds non-linearity, letting the network learn complex patterns. This is what makes a neural network more powerful than linear regression.
  - `nn.Linear(128, 10)` — output layer. 10 neurons = 10 digit classes (0–9). Each neuron outputs a score (logit) for its class.

- [ ] **Run a forward pass on one batch BEFORE training**
  *Why before training:* This is a crucial sanity check AND a learning moment. You need to see what the model outputs when its weights are random — this is your baseline. Training only makes sense once you understand what "before" looks like.
  *What it means:* `output = model(images)` runs the forward pass. `output.shape` = `(64, 10)` — for each of 64 images, 10 raw logits. These are NOT probabilities yet — they can be any real number, positive or negative.
  *What to check:* `output.argmax(dim=1)` picks the highest logit per image as the prediction. Compare with actual labels — accuracy should be ~10% (random guessing across 10 classes). `torch.softmax(output, dim=1)` converts logits to probabilities that sum to 1 — you'll see roughly uniform ~0.1 per class.

---

**3. Training**

- [ ] **Define loss function and optimizer**
  *Why separate step:* These two pieces control HOW the model learns. The loss says "how wrong are you" and the optimizer says "how to fix it." Choosing them is a design decision.
  *What it means:*
  - `nn.CrossEntropyLoss()` — the standard loss for classification. Internally it applies softmax to your logits, then computes negative log likelihood. Do NOT apply softmax yourself before this — it's built in, and doing it twice ruins training. Higher loss = worse predictions.
  - `torch.optim.Adam(model.parameters(), lr=1e-3)` — Adam optimizer. `model.parameters()` gives it all the weights/biases to update. `lr=1e-3` is the learning rate — how big each update step is. Adam adapts the learning rate per-parameter, which is why it's the default choice over plain SGD.

- [ ] **Write the training loop**
  *Why this is the heart of the project:* Everything before was setup. This is where the model actually learns. You'll run this loop thousands of times and the weights gradually shift from random to useful.
  *What it means, line by line:*
  - `model.train()` — puts model in training mode (matters when you add dropout/batchnorm later).
  - `for images, labels in train_loader:` — iterate through all 60k images in batches of 64.
  - `preds = model(images)` — forward pass: input → prediction.
  - `loss = loss_fn(preds, labels)` — compute how wrong the predictions are.
  - `optimizer.zero_grad()` — **CRITICAL**: clear old gradients. PyTorch accumulates gradients by default. If you skip this, gradients from previous batches mix in and training goes wrong.
  - `loss.backward()` — backpropagation: compute the gradient of the loss w.r.t. every weight. This is the chain rule applied automatically by PyTorch's autograd (see [[Gradient Descent]]).
  - `optimizer.step()` — update every weight by `weight -= lr * gradient`. This is one step of gradient descent.
  - One full pass through all batches = one **epoch**.

- [ ] **Add loss tracking — collect `loss.item()` per batch, plot with matplotlib**
  *Why here:* The loss curve is your main diagnostic tool. It tells you if the model is learning (loss going down), stuck (flat), or broken (NaN, going up). You'll use this in every future project.
  *What to check:* loss should start around 2.3 (= -ln(0.1), random guessing on 10 classes) and drop below 0.1.

- [ ] **Add accuracy tracking per epoch**
  *Why here:* Loss is the model's internal score. Accuracy is what you actually care about — what percentage of digits does it get right? They usually correlate but not always.
  *What to check:* should climb from ~85% (epoch 1) to ~97% (epoch 3).

- [ ] **Run training — expect ~97% after 3 epochs**

---

**4. Evaluation**

- [ ] **Evaluate with `model.eval()` + `torch.no_grad()`**
  *Why a separate phase:* Training accuracy is optimistic — the model has seen those images. Test accuracy on unseen data is the real measure. If test accuracy is much lower than train accuracy, you're overfitting.
  *What it means:*
  - `model.eval()` — switches off training-specific behavior. Right now your model has no dropout or batchnorm so it doesn't matter yet, but it WILL matter in Phase 5 when you add dropout. Build the habit now.
  - `torch.no_grad()` — tells PyTorch "I'm not going to call backward()". Skips gradient tracking, saves memory and runs faster. Always use during evaluation.

- [ ] **Build a confusion matrix**
  *Why here:* Overall accuracy hides per-class behavior. Maybe the model is 99% on digit 1 but 90% on digit 8. The confusion matrix is a 10×10 grid where `[i][j]` = how many times the model predicted `j` when the answer was `i`. Diagonal = correct, off-diagonal = mistakes.
  *What to check:* which digit pairs have the highest off-diagonal values? Likely 4↔9, 3↔8, 7↔2 — digits that share visual features.

- [ ] **Display 10 misclassified images with predicted vs actual labels**
  *Why here:* This is where you develop ML intuition. Looking at failure cases tells you more than accuracy numbers. Are the mistakes "reasonable" (even a human would hesitate) or "dumb" (the model clearly hasn't learned basic shapes)?

---

**5. Experiments**

The real learning phase. Change ONE variable at a time, re-train, record results in a table. This builds intuition you can't get from reading.

- [ ] **Layer width: try 32, 128, 256, 512 neurons in hidden layer**
  *Why this first:* It's the simplest change and teaches the most fundamental concept — model capacity. A wider layer can represent more patterns but uses more memory and may overfit.
  *What to observe:* 32 neurons may underfit (~94%), 512 may not improve much over 128 but trains slower. There's a point of diminishing returns. Fill in a results table with accuracy for each.

- [ ] **Depth: add a hidden layer → 784→128→64→10**
  *Why here:* After width, the natural question is "what about adding more layers?" Deeper networks can learn hierarchical features — first layer sees edges, second sees shapes.
  *What to observe:* might slightly improve accuracy. But the key lesson: deeper ≠ always better. With 2-3 layers on MNIST, depth barely matters. It matters a lot on harder problems.

- [ ] **Activation: replace ReLU with Sigmoid, then Tanh**
  *Why here:* You've been using ReLU without questioning it. Now see what happens with older activations.
  *What to observe:* Sigmoid trains noticeably slower — vanishing gradient problem, gradients near 0 or 1 are tiny. Tanh is better than Sigmoid but still slower than ReLU. This is why ReLU became the default in modern deep learning.

- [ ] **Learning rate: try 0.1, 1e-3, 1e-6**
  *Why here:* The single most important hyperparameter. Every practitioner needs to feel what too-high and too-low looks like.
  *What to observe:* `lr=0.1` — loss oscillates wildly or goes NaN (steps too big, overshooting the minimum). `lr=1e-6` — loss barely moves (steps too small). `lr=1e-3` is the sweet spot for Adam. This is why Adam is popular: it adapts lr per-parameter, making the choice less fragile.

- [ ] **Batch size: try 16, 64, 256, 1024**
  *Why here:* Batch size affects training dynamics and speed. Smaller = noisier gradients but more updates per epoch. Larger = smoother gradients but fewer updates.
  *What to observe:* training time per epoch changes significantly. Final accuracy may vary slightly.

- [ ] **Epochs: train for 1, 3, 5, 10, 20**
  *Why here:* When should you stop training? This experiment shows overfitting in action.
  *What to observe:* at some point, train accuracy keeps improving but test accuracy plateaus or drops. That's overfitting — the model memorizes training data instead of learning general patterns. On MNIST it's subtle, but visible.

- [ ] **Dropout: add `nn.Dropout(0.2)` after ReLU layers**
  *Why last:* Dropout is a regularization technique — it randomly zeroes 20% of neurons during training, forcing the network not to rely on any single neuron. It's the natural response to the overfitting you observed in the epochs experiment.
  *What to observe:* train accuracy drops slightly (dropout makes training harder on purpose), but test accuracy stays the same or improves. The gap between train and test accuracy shrinks — that's regularization working.

---

**6. CNN**

- [ ] **Build a simple CNN: Conv2d → ReLU → MaxPool → Conv2d → ReLU → MaxPool → Flatten → Linear → Linear**
  *Why after FC experiments:* Your FC model flattens the image to 784 pixels and treats each independently — pixel 0 has no relationship to pixel 1. A CNN preserves 2D structure and learns local patterns (edges, curves, loops). This is why CNNs dominate image tasks.
  *What it means:*
  - `nn.Conv2d(1, 16, kernel_size=3)` — slides a 3×3 filter across the image. Output: 16 feature maps, each detecting a different local pattern. The kernel weights are learned during training.
  - `nn.MaxPool2d(2)` — takes every 2×2 block and keeps the max. Halves spatial dimensions. Makes the model translation-invariant (a 3 in the top-left looks the same as a 3 in the center).
  - After 2 conv+pool blocks, Flatten and pass through Linear layers — same as before.
  *What to calculate:* input `(1, 28, 28)` → after conv1+pool `(16, 13, 13)` → after conv2+pool `(32, 5, 5)` → flatten = 32×5×5 = 800 → `Linear(800, 128)` → `Linear(128, 10)`. Print shapes after each layer to verify.

- [ ] **Train CNN and compare accuracy to FC model**
  *What to observe:* CNN should hit ~99% vs FC's ~97%. That 2% gap represents the power of spatial awareness. On harder datasets (CIFAR-10, ImageNet), the gap becomes enormous.

- [ ] **Print tensor shape after each layer**
  *Why:* Understanding how dimensions transform through conv and pooling is essential for building any CNN. If you get a shape mismatch, this is how you debug it.

---

**7. Wrap Up**

- [ ] **Save best model with `torch.save(model.state_dict(), 'best_model.pth')`**
  *Why:* You've trained multiple models. Save the best one so you can load it later without retraining. `state_dict()` saves only the weights, not the architecture — load it back with `model.load_state_dict(torch.load(...))`.

- [ ] **Fill in Results section below**
  What to include: final accuracy for FC and CNN, most interesting experiment result, confusion matrix insight.

- [ ] **Write reflection: what surprised you, which experiment taught the most, what to try next**

- [ ] **Link forward: [[ML Roadmap]] next steps** — Fashion-MNIST, CIFAR-10, calculus essentials, classical ML models

---

### Results

| Model | Test Accuracy | Notes |
|---|---|---|
| FC (784→128→10) | | |
| CNN (2× Conv+Pool) | | |

Best experiment:

What surprised me:

Next:

##  Links
- [[MNIST]]
- [[PyTorch]]
- [[Gradient Descent]]
- [[Models]]
- [[Neural networks]]
- [[How to build a model]]
- [[ML Roadmap]]

