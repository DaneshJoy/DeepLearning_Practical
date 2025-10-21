# Local Installation


## 1- Install WSL in Windows

- Install the latest NVidia driver for your graphics card.

- Run this command in PowerShell (run as administrator):

  ```bash
  wsl --install
  ```

- The rest of the document is inside the WSL Linux (e.g. Ubunu)

---

## 2- Install Anaconda or Miniconda (for Linux)

- https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2

- ```bash
  mkdir -p ~/miniconda3
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
  bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
  rm ~/miniconda3/miniconda.sh
  ```

### Create Virtual Environment

```bash
conda create -n <env_name> python=3.12
conda activate <env_name>
```

---

## 3- Install Keras

- https://keras.io/getting_started/

- ```bash
  pip install --upgrade keras
  ```


---

## 4- Install Backend(s)

### Tensorflow

- https://www.tensorflow.org/install/pip

- **Install**

```bash
python3 -m pip install tensorflow[and-cuda]
```

- Test GPU Support

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### PyTorch

- https://pytorch.org/get-started/locally/

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

### JAX

- https://docs.jax.dev/en/latest/installation.html

```bash
pip install -U "jax[cuda12]"
```

---

## 5- Install and Run the Notebook (jupyter-lab)

```bash 
conda activate <env_name>
pip install jupyter
jupyter-lab --no-browser
```

> [!NOTE]
>
> Use `--no-browser` in WSL and open the notebook in browser by Ctrl+Click on the provided URL or copy/paste it inside your browser.
>
> This is not needed in Windows CMD or PowerShell where the browser will open automatically.
