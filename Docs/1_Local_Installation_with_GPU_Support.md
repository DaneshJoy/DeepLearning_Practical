# Local Installation


## 1- Install WSL in Windows

- Install the latest NVidia driver for your graphics card.

- Run this command in Windows **Terminal** (PowerShell or CMD)(run as administrator):

  ```bash
  wsl --install
  ```

- Restart the Terminal

## 2- Install CUDA on WSL

- Update the WSL if already installed:

  ```bash
  wsl --update
  ```

- Open the **Ubuntu** shell and update the packages:

  ```bash
  sudo apt update && sudo apt upgrade -y
  ```

- Install the repository key for CUDA (inside the **Ubuntu** shell):

  ```bash
  # remove the old GPG key if exists
  sudo apt-key del 7fa2af80
  
  # Download and install repository key
  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  ```
  
  > [!TIP]
  >
  > You can manually download the key file and put the downloaded `.deb` file in the /home/<username> path inside the Linux/Ubuntu directory and just run the `dpkg`  command
  >
  > > you can find the Linux/Ubuntu directory in the explorer sidebar after installing WSL

- Install the CUDA (inside the **Ubuntu** shell):

  ```bash
  sudo apt-get -y install cuda-toolkit-12-6
  ```

  > [!NOTE]
  >
  > The cuda-toolkit version should be changed based on the currently supported or installed version of cuda in TensorFlow / PyTorch / JAX

- Help: https://ubuntu.com/wsl/docs/latest/howto/gpu-cuda/#install-nvidia-cuda-on-ubuntu

---

> [!NOTE]
>
> The rest of the document will be done inside the WSL environment (e.g. **Ubuntu** shell)

---

## 2- Install UV on WSL:

https://docs.astral.sh/uv/getting-started/installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

Then restart the shell to apply the changes.

---

## 3- Create UV env

Create a directory

Create the uv venv inside the project directory

```bash
mkdir <project_dir_name>
cd <project_dir_name>
uv venv
```

---

## 4- Install Keras

> Install pip if is not installed: `sudo apt install python3-pip`

- https://keras.io/getting_started/

- ```bash
  uv pip install --upgrade keras
  ```


---

## 5- Install Backend(s)

### Tensorflow

- https://www.tensorflow.org/install/pip

- **Install**

    ```bash
    uv pip install tensorflow[and-cuda]
    ```

    with اینترنت ملی

    ```bash
    uv pip install tensorflow[and-cuda] -i https://mirror-pypi.runflare.com/simple
    ```

- Test GPU Support

    ```bash
    uv run python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```

### PyTorch

- https://pytorch.org/get-started/locally/

    ```bash
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
    ```

### JAX

- https://docs.jax.dev/en/latest/installation.html

    ```bash
    uv pip install -U "jax[cuda12]"
    ```

---

## 6- Install and Run jupyter

```bash
uv pip install jupyter
```

```bash
uv run jupyter-lab --no-browser
```

> [!NOTE]
>
> Use `--no-browser` in WSL and open the notebook in browser by Ctrl+Click on the provided URL or copy/paste it inside your browser.
>
> This is not needed in Windows CMD or PowerShell where the browser can open automatically.
