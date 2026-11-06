# نصب محلی با پشتیبانی GPU

<div dir="rtl">

## ۱- نصب WSL در ویندوز

- آخرین نسخه درایور NVidia مناسب کارت گرافیک خود را نصب کنید.

- دستور زیر را در PowerShell (با دسترسی administrator) اجرا کنید:

</div>

```bash
wsl --install
```

<div dir="rtl">

> بقیه مراحل داخل محیط WSL (مثلاً Ubuntu) انجام می‌شود.

---

## ۲- نصب UV روی WSL

</div>

https://docs.astral.sh/uv/getting-started/installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

<div dir="rtl">یا</div>

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

<div dir="rtl">

سپس ترمینال را ری‌استارت کنید تا تغییرات اعمال شوند.

---

## ۳- ساخت محیط مجازی با UV

یک دایرکتوری بسازید و داخل آن محیط مجازی (venv) ایجاد کنید:

</div>

```bash
mkdir <نام_پروژه>
cd <نام_پروژه>
uv venv
```

---

<div dir="rtl">

## ۴- نصب Keras

> اگر pip نصب نیست، ابتدا آن را نصب کنید: `sudo apt install python3-pip`

</div>

https://keras.io/getting_started/

```bash
uv pip install --upgrade keras
```

---

<div dir="rtl">
## ۵- نصب Backend(s)

### نصب CUDA در محیط WSL

```bash
# حذف کلید GPG قدیمی (در صورت وجود)
sudo apt-key del 7fa2af80

# دانلود و نصب کلید ریپازیتوری
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# به روزرسانی ریپازیتوری ها و نصب CUDA Toolkit
sudo apt update
sudo apt install cuda-toolkit-12-6
```



### TensorFlow

</div>

https://www.tensorflow.org/install/pip

<div dir="rtl">

**نصب (اینترنت آزاد):**

</div>

```bash
uv pip install tensorflow[and-cuda]
```

<div dir="rtl">

**نصب با اینترنت ملی:**

</div>

```bash
uv pip install tensorflow[and-cuda] -i https://mirror-pypi.runflare.com/simple
```

<div dir="rtl">

**تست پشتیبانی GPU:**

</div>

```bash
uv run python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

<div dir="rtl">

### PyTorch

</div>

https://pytorch.org/get-started/locally/

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

<div dir="rtl">

### JAX

</div>

https://docs.jax.dev/en/latest/installation.html

```bash
uv pip install -U "jax[cuda12]"
```

---

<div dir="rtl">

## ۶- نصب و اجرای Jupyter

</div>

```bash
uv pip install jupyter
```

```bash
uv run jupyter-lab --no-browser
```

<div dir="rtl">

> [!NOTE]
>
> در محیط WSL از گزینه `--no-browser` استفاده کنید و نوت‌بوک را با Ctrl+Click روی لینک نمایش‌داده‌شده یا با کپی‌پیست آن در مرورگر باز کنید.
>
> در CMD یا PowerShell ویندوز، این گزینه لازم نیست و مرورگر به‌صورت خودکار باز می‌شود.

</div>



## نصب Tensorboard

```bash
uv pip install tensorboard
```

اجرای TensorBoard در یک ترمینال جدا بر روی مسیر تعریف شده در کد:

```bash
tensorboard --logdir='<your_log_dir>'
```



اگر موقع اجرای دستور tensorboard با خطای ModuleNotFoundError: No 

module named 'pkg_resources' مواجه شدید، دستورات زیر را انجام دهید:

```bash
uv pip uninstall setuptools
uv pip install "setuptools<70.0.0"
```

