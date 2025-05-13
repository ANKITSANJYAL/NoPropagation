# No-Propagation Diffusion Transformer (NoPropDT) on MNIST

This repository contains a clean and complete PyTorch implementation of the **No-Propagation Diffusion Transformer (NoPropDT)** model as described in the paper by researchers from the University of Oxford.
Paper Link:
https://arxiv.org/pdf/2503.24322

---

## 🚀 Purpose

The goal of this project was to go **from scratch to paper-accurate** implementation of the No-Propagation learning algorithm — a backpropagation-free method for training deep models using local layer-wise targets and diffusion-based denoising.

This implementation specifically focuses on MNIST to replicate the experimental setup shown in the original paper and validate that the model can reach high accuracy using purely local updates.

---

## 💡 Intuition

Traditional neural networks use backpropagation to update weights layer by layer using gradients. While powerful, it's:

* Biologically implausible
* Not memory efficient
* Hard to parallelize layer-wise

NoPropDT replaces backprop with a stack of **denoising blocks**. Each block performs a denoising operation that refines a noisy class embedding toward the correct label embedding. The entire network learns through **sample reuse and local MSE** — not gradients passed through the whole network.

The intuition:

* Start with a noisy guess for class embedding.
* Use CNN + MLP blocks to clean (denoise) it toward the correct label.
* Repeat this T times.
* At the end, classify the denoised embedding with a linear head.

---

## 🧠 How the Codebase Works

The code is modular and broken into clear parts:

### 1. `models/no_prop_dt.py`

Defines the **NoPropDT model**:

* `DenoiseBlock`: CNN + MLP that processes image + noisy embedding
* `NoPropDT`: Stack of denoising blocks + classifier + cosine noise schedule

### 2. `trainer/train_nopropdt.py`

* Trains the model using **layer-wise local losses** (no backprop)
* Final step uses cross-entropy + KL for stability
* Evaluates accuracy at each epoch

### 3. `data/mnist_loader.py`

* Loads MNIST dataset with `ToTensor()`
* Returns train/test DataLoaders

### 4. `experiments/run_mnist_dt.py`

* Loads data, builds model, sets hyperparams, calls trainer
* Clean separation for future dataset extensions

### 5. `main.py`

* Entrypoint script that runs the MNIST experiment

### 🔄 Code Flow Diagram:

```
main.py
 └─→ run_mnist_dt.py
     ├─→ get_mnist_loaders() → data/mnist_loader.py
     ├─→ model = NoPropDT(...) → models/no_prop_dt.py
     └─→ train_nopropdt(...) → trainer/train_nopropdt.py
```

---

## ✅ Results

On MNIST, the model achieves **\~99% validation accuracy by epoch 7**, matching or exceeding what the paper reports.

### 📈 Accuracy Curve

![Training and Validation Accuracy](assets/MnistNoProp.png)


On CIFR-10, the model achieved **\~75% validation accuracy by epoch 50**, whilst it had been trained for just 50 epochs and seems like it can still go on learning

![Training and Validation Accuracy](assets/CIFR_50.png)

And all of this is done **without using backpropagation**.

---

## 🤝 How to Contribute

Want to experiment with this? You can:

* Clone the repo
* Add loaders under `data/` for your dataset (CIFAR-10, SVHN, etc.)
* Copy `run_mnist_dt.py` and modify it into `run_cifar_dt.py`
* Make sure your image size and input channels match the CNN in `DenoiseBlock`

If you have improvements, open a PR — clean modularity is maintained intentionally.

---

## 🔭 Next Steps for Me

This was the MNIST milestone. Next:

1. Add support for **CIFAR-10** and **SVHN**
2. Add **ReLU/decoder** variants from the full paper (for nonlinear denoising)
3. Add **backprop baselines** for comparison
4. Add **WandB tracking** for clean experiment logging
5. (Optional) Benchmark training speed and memory vs backprop

---

Thanks for checking out the project — if you're into optimization, bio-inspired learning, or just curious about alternatives to backprop, you’ll enjoy digging into this.

PRs and stars are always welcome :)

---
## 🙏 Credits

This implementation was restructured and extended from a notebook version shared by the community. Special thanks for the inspiration!
https://github.com/ashishbamania/Tutorials-On-Artificial-Intelligence/blob/main/Training%20Without%20Backpropagation/NoPropDT_on_MNIST.ipynb
